import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# -----------------------------
# 0) Repro / Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 1) Data: CoLA + balanced sampling
# -----------------------------
def load_cola():
    ds = load_dataset("glue", "cola")
    return ds["train"], ds["validation"]


def balanced_sample_train(train_ds, per_label: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for ex in train_ds:
        by_label[int(ex["label"])].append(ex)

    for y in [0, 1]:
        rng.shuffle(by_label[y])
        if len(by_label[y]) < per_label:
            raise ValueError(
                f"Not enough examples for label={y}. "
                f"Have {len(by_label[y])}, need {per_label}."
            )
        by_label[y] = by_label[y][:per_label]

    sampled = by_label[0] + by_label[1]
    rng.shuffle(sampled)
    return sampled


# -----------------------------
# 1b) Few-shot: balanced demonstrations
# -----------------------------
def build_few_shot_prefix(train_ds, k_per_label: int, seed: int, demo_template: str) -> str:
    """
    Build a single string containing balanced demonstrations: k_per_label from each class.
    Interleaves labels: 0,1,0,1,...
    """
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for ex in train_ds:
        by_label[int(ex["label"])].append(ex)

    for y in [0, 1]:
        rng.shuffle(by_label[y])
        if len(by_label[y]) < k_per_label:
            raise ValueError(f"Not enough examples for label={y} to build few-shot prefix.")
        by_label[y] = by_label[y][:k_per_label]

    demos = []
    for i in range(k_per_label):
        for y in [0, 1]:
            demos.append(demo_template.format(label=y, sentence=by_label[y][i]["sentence"]))

    return "".join(demos)


def apply_few_shot(text: str, prefix: str, target_template: str) -> str:
    """
    Final prompt = demonstrations + target query.
    """
    return prefix + target_template.format(sentence=text)


# -----------------------------
# 1c) Tokenization / batching
# -----------------------------
@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def make_dataloader(
    examples: List[dict],
    tokenizer,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    few_shot_prefix: str = "",
    few_shot_target_template: str = "Sentence: {sentence}\nLabel:",
) -> DataLoader:
    """
    Tokenize and batch examples for a causal LM input.
    CoLA input text lives in the 'sentence' field.

    If few_shot_prefix != "":
      prompt = prefix + target_template(sentence=...)
    else:
      prompt = raw sentence
    """

    def collate(batch_ex: List[dict]) -> Batch:
        texts_raw = [ex["sentence"] for ex in batch_ex]
        if few_shot_prefix:
            texts = [apply_few_shot(t, few_shot_prefix, few_shot_target_template) for t in texts_raw]
        else:
            texts = texts_raw

        ys = torch.tensor([int(ex["label"]) for ex in batch_ex], dtype=torch.long)

        tok = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return Batch(
            input_ids=tok["input_ids"],
            attention_mask=tok["attention_mask"],
            labels=ys,
        )

    return DataLoader(examples, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


# -----------------------------
# 2) Hooking MLP activations
# -----------------------------
class LayerActivationCatcher:
    """
    Captures the output tensor of a chosen activation module for ONE layer.
    We attach a forward hook and store the last batch activation.

    Stored tensor shape typically: [batch, seq_len, hidden]
    """
    def __init__(self):
        self.last: Optional[torch.Tensor] = None

    def hook_fn(self, module, inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        self.last = out


def find_transformer_layers(model) -> List[torch.nn.Module]:
    """
    Tries common attribute paths for decoder-only transformer blocks.
    Gemma models typically store layers in model.model.layers
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError("Could not locate transformer layers. Inspect model attributes.")


def find_mlp_activation_module(layer: torch.nn.Module) -> torch.nn.Module:
    """
    Find the activation module inside a layer's MLP.
    Heuristic:
      - submodule name contains 'act' or 'activation'
      - or module class looks like gelu/silu/relu/swish
    """
    candidates = []
    for name, mod in layer.named_modules():
        lname = name.lower()
        cls = mod.__class__.__name__.lower()

        looks_like_act_name = ("act" in lname) or ("activation" in lname)
        looks_like_act_type = any(k in cls for k in ["gelu", "silu", "relu", "swish"])
        if looks_like_act_name or looks_like_act_type:
            candidates.append((name, mod))

    if not candidates:
        raise ValueError("Could not find an activation module in this layer's MLP.")

    def score(item):
        name, _ = item
        lname = name.lower()
        depth = lname.count(".")
        bonus = 0
        if lname.endswith("act") or "act_fn" in lname or "activation" in lname:
            bonus -= 2
        return (depth + bonus, len(lname))

    candidates.sort(key=score)
    return candidates[0][1]


@torch.no_grad()
def extract_features_for_layer(
    model,
    dataloader: DataLoader,
    layer_idx: int,
    positions: List[str],
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    For a single layer:
      - attach a hook to that layer's MLP activation module
      - run the dataloader
      - collect features at requested positions

    positions:
      - "first": token position 0
      - "last": last non-pad token (attention_mask-based)
      - "mean": mean over non-pad tokens

    Returns:
      feats_by_pos: dict pos -> [N, hidden]
      labels: [N]
    """
    layers = find_transformer_layers(model)
    layer = layers[layer_idx]
    act_mod = find_mlp_activation_module(layer)

    catcher = LayerActivationCatcher()
    handle = act_mod.register_forward_hook(catcher.hook_fn)

    feats = {p: [] for p in positions}
    ys_all = []

    for batch in dataloader:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        ys = batch.labels.cpu().numpy()

        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        if catcher.last is None:
            raise RuntimeError("Hook did not capture activation output. Check module selection.")

        A = catcher.last  # [B, T, H]
        B, T, H = A.shape

        lengths = attention_mask.sum(dim=1)  # [B]
        last_idx = torch.clamp(lengths - 1, min=0)  # [B]

        for p in positions:
            if p == "first":
                f = A[:, 0, :]
            elif p == "last":
                f = A[torch.arange(B, device=device), last_idx, :]
            elif p == "mean":
                mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
                denom = mask.sum(dim=1).clamp(min=1)
                f = (A * mask).sum(dim=1) / denom
            else:
                raise ValueError(f"Unknown position spec: {p}")

            feats[p].append(f.detach().float().cpu().numpy())

        ys_all.append(ys)
        catcher.last = None

    handle.remove()

    feats_by_pos = {p: np.concatenate(feats[p], axis=0) for p in positions}
    labels = np.concatenate(ys_all, axis=0)
    return feats_by_pos, labels


# -----------------------------
# 3) Linear probe training/eval
# -----------------------------
def train_and_eval_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    C: float = 1.0,
    max_iter: int = 2000,
) -> Dict[str, float]:
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")),
        ]
    )
    clf.fit(X_train, y_train)
    yhat = clf.predict(X_valid)

    return {
        "acc": float(accuracy_score(y_valid, yhat)),
        "f1": float(f1_score(y_valid, yhat)),
    }


# -----------------------------
# 4) Orchestration
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="google/gemma-2-2b")
    ap.add_argument("--per_label", type=int, default=500, help="train samples per label")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--positions", nargs="+", default=["first", "last"],
                    help="Positions to probe: first last mean (space-separated)")
    ap.add_argument("--save_csv", default="probe_results.csv")

    # Few-shot options
    ap.add_argument("--few_shot_k", type=int, default=0,
                    help="If >0, prepend K examples per label (balanced) as in-context demonstrations.")
    ap.add_argument("--few_shot_seed", type=int, default=123,
                    help="Seed for selecting few-shot demonstration examples.")
    ap.add_argument("--few_shot_template", type=str, default="Label: {label}\nSentence: {sentence}\n\n",
                    help="Format for each demonstration. Must contain {label} and {sentence}.")
    ap.add_argument("--few_shot_target_template", type=str, default="Sentence: {sentence}\nLabel:",
                    help="Format for the target query sentence appended after demonstrations.")
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()

    train_ds, valid_ds = load_cola()

    # Build few-shot prefix (optional)
    few_shot_prefix = ""
    if args.few_shot_k > 0:
        few_shot_prefix = build_few_shot_prefix(
            train_ds=train_ds,
            k_per_label=args.few_shot_k,
            seed=args.few_shot_seed,
            demo_template=args.few_shot_template,
        )
        print(f"Few-shot enabled: {args.few_shot_k} per label ({2*args.few_shot_k} demos total)")
        print("Tip: for few-shot runs, --positions last is usually the most meaningful.\n")

    train_examples = balanced_sample_train(train_ds, per_label=args.per_label, seed=args.seed)
    valid_examples = list(valid_ds)

    train_loader = make_dataloader(
        train_examples, tokenizer, args.batch_size, args.max_length, shuffle=False,
        few_shot_prefix=few_shot_prefix,
        few_shot_target_template=args.few_shot_target_template,
    )
    valid_loader = make_dataloader(
        valid_examples, tokenizer, args.batch_size, args.max_length, shuffle=False,
        few_shot_prefix=few_shot_prefix,
        few_shot_target_template=args.few_shot_target_template,
    )

    layers = find_transformer_layers(model)
    n_layers = len(layers)
    print(f"Model has {n_layers} transformer layers")
    print(f"Probing positions: {args.positions}")

    results = []

    for layer_idx in range(n_layers):
        print(f"\n[Layer {layer_idx}] extracting features...")
        Xtr_by_pos, ytr = extract_features_for_layer(
            model=model, dataloader=train_loader,
            layer_idx=layer_idx, positions=args.positions, device=device
        )
        Xva_by_pos, yva = extract_features_for_layer(
            model=model, dataloader=valid_loader,
            layer_idx=layer_idx, positions=args.positions, device=device
        )

        for pos in args.positions:
            metrics = train_and_eval_probe(
                X_train=Xtr_by_pos[pos], y_train=ytr,
                X_valid=Xva_by_pos[pos], y_valid=yva
            )
            row = {
                "layer": layer_idx,
                "position": pos,
                "acc": metrics["acc"],
                "f1": metrics["f1"],
            }
            results.append(row)
            print(f"  pos={pos:>5s}  acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}")

    print("\n=== Best layer per position (by F1) ===")
    for pos in args.positions:
        pos_rows = [r for r in results if r["position"] == pos]
        best = max(pos_rows, key=lambda r: r["f1"])
        print(f"pos={pos:>5s} -> best_layer={best['layer']}  f1={best['f1']:.4f}  acc={best['acc']:.4f}")

    import csv
    with open(args.save_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "position", "acc", "f1"])
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\nSaved results to: {args.save_csv}")


if __name__ == "__main__":
    main()