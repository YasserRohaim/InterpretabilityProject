import argparse
import csv
import json
import os
import random
from typing import List

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier

from probing_utils import (
    batchify,
    build_prompt,
    build_support_examples,
    collect_mlp_activations,
    load_model_and_tokenizer,
    load_task_datasets,
    parse_positions,
    resolve_layers,
    sample_balanced_examples,
    set_seed,
)


def make_model_slug(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def concat_layer_features(activations: List[torch.Tensor], position: int) -> torch.Tensor:
    feats = [act[:, position, :].float().cpu() for act in activations]
    return torch.cat(feats, dim=-1)


def update_running_stats(mean, M2, count, batch_np):
    batch_count = batch_np.shape[0]
    if batch_count == 0:
        return mean, M2, count

    batch_mean = batch_np.mean(axis=0)
    batch_var = batch_np.var(axis=0)
    delta = batch_mean - mean
    total = count + batch_count

    mean = mean + delta * batch_count / total
    M2 = M2 + batch_var * batch_count + (delta ** 2) * count * batch_count / total
    count = total
    return mean, M2, count


def main():
    parser = argparse.ArgumentParser(description="Rank most predictive neurons from a global linear probe.")
    parser.add_argument("--task", choices=["sst2", "imdb", "cola"], default="cola")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--k_shot", type=int, default=16)
    parser.add_argument("--train_per_label", type=int, default=500)
    parser.add_argument("--position", type=str, default="-1")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/probing")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, _eval_ds, cfg = load_task_datasets(args.task)
    support = build_support_examples(
        train_ds,
        k_shot=args.k_shot,
        label_col=cfg["label_col"],
        text_col=cfg["text_col"],
        seed=args.seed,
    )
    train_items = sample_balanced_examples(
        train_ds,
        per_label=args.train_per_label,
        label_col=cfg["label_col"],
        text_col=cfg["text_col"],
        seed=args.seed,
    )

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    layers = resolve_layers(model)

    positions = parse_positions(args.position, args.max_length)
    if len(positions) != 1:
        raise ValueError("--position must resolve to a single index")
    position = positions[0]

    # Pass 1: compute mean/std for standardization
    total_dim = None
    layer_sizes = None
    count = 0
    mean = None
    M2 = None

    for batch in batchify(train_items, args.batch_size):
        prompts = [
            build_prompt(support, ex["text"], cfg["prompt_label"], cfg["label_words"])
            for ex in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        ).to(device)
        attention_mask = inputs["attention_mask"].detach().cpu()
        valid = attention_mask[:, position].bool().numpy()
        if valid.sum() == 0:
            continue

        activations = collect_mlp_activations(model, inputs, layers)
        if layer_sizes is None:
            layer_sizes = [act.shape[-1] for act in activations]
            total_dim = int(sum(layer_sizes))
            mean = np.zeros(total_dim, dtype=np.float64)
            M2 = np.zeros(total_dim, dtype=np.float64)

        feats = concat_layer_features(activations, position).numpy()
        feats = feats[valid]
        mean, M2, count = update_running_stats(mean, M2, count, feats)

    if total_dim is None:
        raise RuntimeError("No valid samples found for the requested position.")

    std = np.sqrt(M2 / max(count, 1))
    std[std == 0] = 1.0

    # Pass 2: train global linear probe
    probe = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=args.alpha,
        max_iter=1,
        learning_rate="optimal",
        random_state=args.seed,
    )
    initialized = False
    for epoch in range(args.epochs):
        random.shuffle(train_items)
        for batch in batchify(train_items, args.batch_size):
            prompts = [
                build_prompt(support, ex["text"], cfg["prompt_label"], cfg["label_words"])
                for ex in batch
            ]
            labels = np.array([ex["label"] for ex in batch], dtype=np.int64)

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            attention_mask = inputs["attention_mask"].detach().cpu()
            valid = attention_mask[:, position].bool().numpy()
            if valid.sum() == 0:
                continue

            activations = collect_mlp_activations(model, inputs, layers)
            feats = concat_layer_features(activations, position).numpy()
            feats = feats[valid]
            labels = labels[valid]

            feats = (feats - mean) / std
            if not initialized:
                probe.partial_fit(feats, labels, classes=np.array([0, 1]))
                initialized = True
            else:
                probe.partial_fit(feats, labels)
        print(f"Finished epoch {epoch + 1}/{args.epochs}")

    if not initialized:
        raise RuntimeError("Probe never initialized; no valid samples to train on.")

    weights = probe.coef_.ravel()
    abs_weights = np.abs(weights)
    top_k = min(args.top_k, abs_weights.shape[0])
    top_idx = np.argpartition(-abs_weights, top_k - 1)[:top_k]
    top_idx = top_idx[np.argsort(-abs_weights[top_idx])]

    offsets = np.cumsum([0] + layer_sizes)

    records = []
    layer_counts = {i: 0 for i in range(len(layer_sizes))}
    for rank, idx in enumerate(top_idx, start=1):
        layer = int(np.searchsorted(offsets, idx, side="right") - 1)
        neuron = int(idx - offsets[layer])
        layer_counts[layer] += 1
        records.append({
            "rank": rank,
            "layer": layer,
            "neuron": neuron,
            "weight": float(weights[idx]),
            "abs_weight": float(abs_weights[idx]),
        })

    best_layer = max(layer_counts.items(), key=lambda x: x[1])

    os.makedirs(args.output_dir, exist_ok=True)
    model_slug = make_model_slug(args.model)
    base = os.path.join(args.output_dir, f"top_neurons_{args.task}_{model_slug}")
    csv_path = base + ".csv"
    json_path = base + ".json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "layer", "neuron", "weight", "abs_weight"])
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "task": args.task,
        "model": args.model,
        "position": position,
        "top_k": top_k,
        "total_dim": total_dim,
        "layer_sizes": layer_sizes,
        "layer_counts": layer_counts,
        "layer_with_most_neurons": {"layer": best_layer[0], "count": best_layer[1]},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(
        f"Layer with most predictive neurons: layer {best_layer[0]} "
        f"({best_layer[1]} of top {top_k})"
    )


if __name__ == "__main__":
    main()
