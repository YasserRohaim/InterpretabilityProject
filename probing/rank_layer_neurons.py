import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------
# Utilities to locate Gemma blocks
# ----------------------------

def get_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise RuntimeError(
        "Could not locate transformer layers. Inspect model attributes and adjust get_layers()."
    )


def mlp_modules(layer) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    if not hasattr(layer, "mlp"):
        raise RuntimeError("Layer has no .mlp. Inspect the model block and adjust mlp_modules().")
    mlp = layer.mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        if not hasattr(mlp, name):
            raise RuntimeError(f"MLP missing {name}. Inspect mlp and adjust mlp_modules().")
    return mlp.gate_proj, mlp.up_proj, mlp.down_proj


def mlp_act_fn(layer):
    mlp = layer.mlp
    if hasattr(mlp, "act_fn"):
        return mlp.act_fn
    raise RuntimeError("MLP has no act_fn. Inspect model code and adjust mlp_act_fn().")


# ----------------------------
# Few-shot prompt building
# ----------------------------

DEFAULT_FEW_SHOT = [
    {"sentence": "The dog chased the cat.", "label": 1},
    {"sentence": "Dog the cat chased.", "label": 0},
    {"sentence": "She is reading a book.", "label": 1},
    {"sentence": "Reading she book a is.", "label": 0},
]


def label_to_text(label: int) -> str:
    # Keep output tokens simple and consistent.
    return "acceptable" if int(label) == 1 else "unacceptable"


def load_few_shot(path: str) -> List[Dict]:
    """
    Load few-shot examples from a JSON file.

    Expected formats:
    1) A list:
       [{"sentence": "...", "label": 0/1}, ...]
    2) A dict with "examples":
       {"examples": [{"sentence": "...", "label": 0/1}, ...]}

    Labels should match CoLA: 1=acceptable, 0=unacceptable.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "examples" in data:
        data = data["examples"]
    if not isinstance(data, list):
        raise ValueError("Few-shot JSON must be a list or a dict with an 'examples' list.")
    for ex in data:
        if "sentence" not in ex or "label" not in ex:
            raise ValueError("Each few-shot example must have 'sentence' and 'label'.")
        if int(ex["label"]) not in (0, 1):
            raise ValueError("Few-shot labels must be 0 or 1.")
    return data


def build_few_shot_prefix(
    few_shot_examples: List[Dict],
    k: int,
    template: str = "Sentence: {sentence}\nAcceptability: {answer}\n",
) -> str:
    """
    Build a prefix containing k few-shot examples.
    """
    k = min(k, len(few_shot_examples))
    parts = []
    for ex in few_shot_examples[:k]:
        parts.append(
            template.format(
                sentence=ex["sentence"],
                answer=label_to_text(int(ex["label"])),
            )
        )
    return "\n".join(parts).strip() + ("\n\n" if parts else "")


def build_prompt(
    sentence: str,
    few_shot_prefix: str = "",
    query_template: str = "Sentence: {sentence}\nAcceptability:",
) -> str:
    """
    Final prompt = [few-shot prefix] + [query]
    """
    return few_shot_prefix + query_template.format(sentence=sentence)


def pick_position(input_ids: torch.Tensor, attention_mask: torch.Tensor, which: str) -> torch.Tensor:
    if which == "cls":
        return torch.zeros((input_ids.size(0),), dtype=torch.long, device=input_ids.device)
    if which == "last":
        lengths = attention_mask.sum(dim=1) - 1
        return lengths.to(torch.long)
    raise ValueError(f"Unknown position mode: {which}")


# ----------------------------
# Streaming stats for predictivity scoring
# ----------------------------

@dataclass
class RunningStats:
    sum_pos: List[np.ndarray]
    sumsq_pos: List[np.ndarray]
    n_pos: int
    sum_neg: List[np.ndarray]
    sumsq_neg: List[np.ndarray]
    n_neg: int


def init_stats(num_layers: int, intermediate_sizes: List[int]) -> RunningStats:
    sum_pos = [np.zeros((m,), dtype=np.float64) for m in intermediate_sizes]
    sumsq_pos = [np.zeros((m,), dtype=np.float64) for m in intermediate_sizes]
    sum_neg = [np.zeros((m,), dtype=np.float64) for m in intermediate_sizes]
    sumsq_neg = [np.zeros((m,), dtype=np.float64) for m in intermediate_sizes]
    return RunningStats(sum_pos, sumsq_pos, 0, sum_neg, sumsq_neg, 0)


def update_stats(stats: RunningStats, acts_per_layer: List[torch.Tensor], labels: torch.Tensor):
    """
    IMPORTANT FIX:
    - Cast to float32 before converting to numpy (bfloat16 isn't supported by numpy).
    """
    labels_np = labels.detach().to(torch.long).cpu().numpy()
    pos_idx = np.where(labels_np == 1)[0]
    neg_idx = np.where(labels_np == 0)[0]

    if len(pos_idx) > 0:
        stats.n_pos += len(pos_idx)
    if len(neg_idx) > 0:
        stats.n_neg += len(neg_idx)

    for l, a in enumerate(acts_per_layer):
        a_np = a.detach().to(torch.float32).cpu().numpy().astype(np.float64)  # (B, M)

        if len(pos_idx) > 0:
            ap = a_np[pos_idx]
            stats.sum_pos[l] += ap.sum(axis=0)
            stats.sumsq_pos[l] += (ap * ap).sum(axis=0)

        if len(neg_idx) > 0:
            an = a_np[neg_idx]
            stats.sum_neg[l] += an.sum(axis=0)
            stats.sumsq_neg[l] += (an * an).sum(axis=0)


def compute_scores(stats: RunningStats, eps: float = 1e-12) -> List[np.ndarray]:
    if stats.n_pos == 0 or stats.n_neg == 0:
        raise RuntimeError("Need both positive and negative samples to compute predictivity scores.")

    scores = []
    for l in range(len(stats.sum_pos)):
        mu_p = stats.sum_pos[l] / stats.n_pos
        mu_n = stats.sum_neg[l] / stats.n_neg

        var_p = (stats.sumsq_pos[l] / stats.n_pos) - (mu_p * mu_p)
        var_n = (stats.sumsq_neg[l] / stats.n_neg) - (mu_n * mu_n)

        denom = np.sqrt(0.5 * (np.maximum(var_p, 0.0) + np.maximum(var_n, 0.0)) + eps)
        s = np.abs(mu_p - mu_n) / denom
        scores.append(s)
    return scores


# ----------------------------
# Activation extraction (Gemma gated MLP)
# ----------------------------

@torch.no_grad()
def extract_mlp_act_outputs(
    model,
    tokenizer,
    sentences: List[str],
    device: str,
    few_shot_prefix: str,
    query_template: str,
    position_mode: str = "last",
    max_length: int = 256,
) -> List[torch.Tensor]:
    model.eval()

    enc = tokenizer(
        [build_prompt(s, few_shot_prefix=few_shot_prefix, query_template=query_template) for s in sentences],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    pos = pick_position(input_ids, attention_mask, position_mode)  # (B,)

    layers = get_layers(model)

    gate_outputs: Dict[int, torch.Tensor] = {}
    hooks = []

    for li, layer in enumerate(layers):
        gate_proj, _, _ = mlp_modules(layer)

        def make_hook(layer_index: int):
            def hook(module, inp, out):
                gate_outputs[layer_index] = out
            return hook

        hooks.append(gate_proj.register_forward_hook(make_hook(li)))

    _ = model(input_ids=input_ids, attention_mask=attention_mask)

    for h in hooks:
        h.remove()

    acts_per_layer = []
    for li, layer in enumerate(layers):
        if li not in gate_outputs:
            raise RuntimeError(f"Missing gate_proj hook output for layer {li}.")
        gate = gate_outputs[li]  # (B, T, M)
        act_fn = mlp_act_fn(layer)
        act = act_fn(gate) if callable(act_fn) else act_fn(gate)

        b_idx = torch.arange(act.size(0), device=act.device)
        act_sel = act[b_idx, pos, :]  # (B, M)
        acts_per_layer.append(act_sel)

    return acts_per_layer


def get_intermediate_sizes(model) -> List[int]:
    layers = get_layers(model)
    sizes = []
    for layer in layers:
        gate_proj, _, _ = mlp_modules(layer)
        sizes.append(gate_proj.out_features)
    return sizes


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="google/gemma-2-2b", help="HF model id")
    ap.add_argument("--dataset", type=str, default="nyu-mll/glue", help="HF dataset id (CoLA is in glue)")
    ap.add_argument("--subset", type=str, default="cola", help="dataset subset/config for GLUE")
    ap.add_argument("--split", type=str, default="validation", help="split to score neurons on")
    ap.add_argument("--max_samples", type=int, default=4000, help="cap samples for speed")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--position", type=str, choices=["last", "cls"], default="last")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--top_k", type=int, default=1000)
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--dtype", type=str, choices=["float16", "bfloat16", "float32"], default="bfloat16")

    # Few-shot options
    ap.add_argument("--few_shot_k", type=int, default=0,
                    help="Number of few-shot examples to prepend (0 disables few-shot).")
    ap.add_argument("--few_shot_path", type=str, default="",
                    help="Path to JSON file with few-shot examples. If empty and --few_shot_default is set, uses built-in examples.")
    ap.add_argument("--few_shot_default", action="store_true",
                    help="Use built-in tiny few-shot examples (only if --few_shot_k > 0 and --few_shot_path is empty).")
    ap.add_argument("--few_shot_template", type=str,
                    default="Sentence: {sentence}\nAcceptability: {answer}\n",
                    help="Template for each few-shot example. Fields: {sentence}, {answer}.")
    ap.add_argument("--query_template", type=str,
                    default="Sentence: {sentence}\nAcceptability:",
                    help="Template for the query example. Field: {sentence}.")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    # Build few-shot prefix once (reused for all examples)
    few_shot_prefix = ""
    if args.few_shot_k > 0:
        if args.few_shot_path:
            few_shot_examples = load_few_shot(args.few_shot_path)
        elif args.few_shot_default:
            few_shot_examples = DEFAULT_FEW_SHOT
        else:
            raise ValueError(
                "Few-shot enabled (--few_shot_k > 0) but no examples provided. "
                "Use --few_shot_path or --few_shot_default."
            )
        few_shot_prefix = build_few_shot_prefix(
            few_shot_examples=few_shot_examples,
            k=args.few_shot_k,
            template=args.few_shot_template,
        )

    print(f"[info] loading model={args.model} dtype={args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model.to(device)

    print(f"[info] loading dataset={args.dataset} subset={args.subset} split={args.split}")
    ds = load_dataset(args.dataset, args.subset, split=args.split)

    if "sentence" not in ds.column_names or "label" not in ds.column_names:
        raise RuntimeError(f"Unexpected columns: {ds.column_names} (expected sentence,label)")

    n = min(len(ds), args.max_samples)
    ds = ds.select(range(n))
    print(f"[info] using n={len(ds)} examples")

    intermediate_sizes = get_intermediate_sizes(model)
    num_layers = len(intermediate_sizes)
    print(f"[info] num_layers={num_layers}, intermediate_size[0]={intermediate_sizes[0]}")

    if few_shot_prefix:
        # Useful to confirm prompt size behavior
        print(f"[info] few-shot enabled: k={args.few_shot_k}, prefix_chars={len(few_shot_prefix)}")

    stats = init_stats(num_layers, intermediate_sizes)

    for start in range(0, len(ds), args.batch_size):
        end = min(start + args.batch_size, len(ds))
        batch = ds.select(range(start, end))
        sentences = batch["sentence"]
        labels = torch.tensor(batch["label"], dtype=torch.long, device=device)

        acts_per_layer = extract_mlp_act_outputs(
            model=model,
            tokenizer=tokenizer,
            sentences=sentences,
            device=device,
            few_shot_prefix=few_shot_prefix,
            query_template=args.query_template,
            position_mode=args.position,
            max_length=args.max_length,
        )

        update_stats(stats, acts_per_layer, labels)

        if (start // args.batch_size) % 25 == 0:
            print(f"[progress] {end}/{len(ds)}")

    print(f"[info] n_pos={stats.n_pos}, n_neg={stats.n_neg}")
    scores_per_layer = compute_scores(stats)

    all_items = []
    for l, s in enumerate(scores_per_layer):
        for i, val in enumerate(s):
            all_items.append((l, i, float(val)))

    all_items.sort(key=lambda x: x[2], reverse=True)
    top_items = all_items[: args.top_k]

    layer_counts = [0] * num_layers
    for l, i, _ in top_items:
        layer_counts[l] += 1

    csv_path = os.path.join(args.out_dir, "top_neurons.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("rank,layer,neuron,score\n")
        for r, (l, i, sc) in enumerate(top_items, start=1):
            f.write(f"{r},{l},{i},{sc}\n")

    json_path = os.path.join(args.out_dir, "layer_counts.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "dataset": f"{args.dataset}/{args.subset}",
                "split": args.split,
                "position": args.position,
                "top_k": args.top_k,
                "n_examples": len(ds),
                "n_pos": stats.n_pos,
                "n_neg": stats.n_neg,
                "few_shot_k": args.few_shot_k,
                "few_shot_path": args.few_shot_path,
                "few_shot_default": bool(args.few_shot_default),
                "layer_counts": layer_counts,
            },
            f,
            indent=2,
        )

    print("\n===Results ===")
    print(f"Saved top neurons: {csv_path}")
    print(f"Saved layer counts: {json_path}")
    print("\nLayer-wise counts (layer -> #top_neurons):")
    for l, c in enumerate(layer_counts):
        print(f"  layer {l:02d}: {c}")

    max_layer = int(np.argmax(np.array(layer_counts)))
    print(f"\nMost top neurons in layer {max_layer} (count={layer_counts[max_layer]})")


if __name__ == "__main__":
    main()