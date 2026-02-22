import random
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


TASK_CONFIGS = {
    "sst2": {
        "hf_path": "nyu-mll/glue",
        "subset": "sst2",
        "text_col": "sentence",
        "label_col": "label",
        "train_split": "train",
        "eval_split": "validation",
        "prompt_label": "Sentiment",
        "label_words": {0: "negative", 1: "positive"},
    },
    "cola": {
        "hf_path": "nyu-mll/glue",
        "subset": "cola",
        "text_col": "sentence",
        "label_col": "label",
        "train_split": "train",
        "eval_split": "validation",
        "prompt_label": "Linguistically acceptable",
        "label_words": {0: "no", 1: "yes"},
    },
    "imdb": {
        "hf_path": "stanfordnlp/imdb",
        "subset": "plain_text",
        "text_col": "text",
        "label_col": "label",
        "train_split": "train",
        "eval_split": "test",
        "prompt_label": "Sentiment",
        "label_words": {0: "negative", 1: "positive"},
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_task_config(task: str) -> Dict:
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task '{task}'. Choose from: {', '.join(TASK_CONFIGS)}")
    return TASK_CONFIGS[task]


def load_task_datasets(task: str):
    cfg = get_task_config(task)
    if cfg["subset"] is None:
        train_ds = load_dataset(cfg["hf_path"], split=cfg["train_split"])
        eval_ds = load_dataset(cfg["hf_path"], split=cfg["eval_split"])
    else:
        train_ds = load_dataset(cfg["hf_path"], cfg["subset"], split=cfg["train_split"])
        eval_ds = load_dataset(cfg["hf_path"], cfg["subset"], split=cfg["eval_split"])
    return train_ds, eval_ds, cfg


def sample_balanced_examples(
    dataset,
    per_label: int,
    label_col: str,
    text_col: str,
    seed: int,
) -> List[Dict]:
    if per_label <= 0:
        return [{"text": ex[text_col], "label": int(ex[label_col])} for ex in dataset]

    ds = dataset.shuffle(seed=seed)
    counts = {0: 0, 1: 0}
    out = []
    for ex in ds:
        label = int(ex[label_col])
        if label not in counts:
            continue
        if counts[label] >= per_label:
            continue
        out.append({"text": ex[text_col], "label": label})
        counts[label] += 1
        if counts[0] >= per_label and counts[1] >= per_label:
            break
    return out


def build_support_examples(
    dataset,
    k_shot: int,
    label_col: str,
    text_col: str,
    seed: int,
) -> Dict[int, List[str]]:
    if k_shot == 0:
        return {0: [], 1: []}
    if k_shot % 2 != 0:
        raise ValueError("k_shot must be even for balanced support.")

    per_label = k_shot // 2
    ds = dataset.shuffle(seed=seed)
    support = {0: [], 1: []}
    for ex in ds:
        label = int(ex[label_col])
        if label not in support:
            continue
        if len(support[label]) >= per_label:
            continue
        support[label].append(ex[text_col])
        if len(support[0]) >= per_label and len(support[1]) >= per_label:
            break
    return support


def build_prompt(
    support: Dict[int, List[str]],
    sentence: str,
    prompt_label: str,
    label_words: Dict[int, str],
) -> str:
    prompt = ""
    for label, sents in support.items():
        label_word = label_words[label]
        for sent in sents:
            prompt += f"Sentence: {sent}\n{prompt_label}: {label_word}\n\n"
    prompt += f"Sentence: {sentence}\n{prompt_label}: "
    return prompt


def batchify(items: List[Dict], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def load_model_and_tokenizer(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda",
            torch_dtype="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        )
    model.eval()
    return model, tokenizer


def resolve_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    raise ValueError("Unsupported model architecture. Can't find transformer layers.")


def resolve_mlp_hook(layer):
    mlp = getattr(layer, "mlp", None) or getattr(layer, "ffn", None)
    if mlp is None:
        raise ValueError("Unsupported layer architecture. Can't find MLP/FFN module.")

    proj = None
    for name in ["gate_proj", "up_proj", "dense_h_to_4h", "c_fc"]:
        if hasattr(mlp, name):
            proj = getattr(mlp, name)
            break
    if proj is None:
        raise ValueError("Unsupported MLP architecture. Can't find projection module.")

    act = None
    for name in ["act_fn", "activation", "act", "gelu"]:
        if hasattr(mlp, name):
            act = getattr(mlp, name)
            break
    if act is None:
        act = torch.nn.Identity()

    if not callable(act):
        raise ValueError("Activation function is not callable.")

    return proj, act


def collect_mlp_activations(model, inputs, layers):
    mlp_acts = [None for _ in layers]
    handles = []

    for idx, layer in enumerate(layers):
        proj, act = resolve_mlp_hook(layer)

        def make_hook(layer_idx, act_fn):
            def hook(_mod, _inp, out):
                mlp_acts[layer_idx] = act_fn(out).detach().cpu()
            return hook

        handles.append(proj.register_forward_hook(make_hook(idx, act)))

    with torch.inference_mode():
        model(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)

    for h in handles:
        h.remove()

    return mlp_acts


def parse_positions(positions: str, max_length: int) -> List[int]:
    if positions == "all":
        return list(range(max_length))
    out = []
    for raw in positions.split(","):
        raw = raw.strip()
        if not raw:
            continue
        pos = int(raw)
        if pos < 0:
            pos = max_length + pos
        if pos < 0 or pos >= max_length:
            raise ValueError(f"Position {raw} is out of bounds for max_length={max_length}")
        out.append(pos)
    if not out:
        raise ValueError("No valid positions provided.")
    return sorted(set(out))
