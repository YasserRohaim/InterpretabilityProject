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


def init_probes(n_layers: int, positions: List[int], seed: int, alpha: float):
    probes = {}
    inited = {}
    for layer_idx in range(n_layers):
        for pos in positions:
            key = (layer_idx, pos)
            probes[key] = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha,
                max_iter=1,
                learning_rate="optimal",
                random_state=seed,
            )
            inited[key] = False
    return probes, inited


def train_epoch(
    model,
    tokenizer,
    layers,
    train_items,
    support,
    cfg,
    probes,
    inited,
    positions,
    batch_size,
    max_length,
    device,
):
    for batch in batchify(train_items, batch_size):
        prompts = [
            build_prompt(support, ex["text"], cfg["prompt_label"], cfg["label_words"])
            for ex in batch
        ]
        labels = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(device)
        attention_mask = inputs["attention_mask"].detach().cpu()

        activations = collect_mlp_activations(model, inputs, layers)

        for layer_idx, layer_act in enumerate(activations):
            if layer_act is None:
                continue
            layer_act = layer_act.float()
            for pos in positions:
                valid = attention_mask[:, pos].bool()
                if valid.sum().item() == 0:
                    continue
                X = layer_act[valid, pos, :].numpy()
                y = labels[valid].numpy()
                key = (layer_idx, pos)
                if not inited[key]:
                    probes[key].partial_fit(X, y, classes=np.array([0, 1]))
                    inited[key] = True
                else:
                    probes[key].partial_fit(X, y)


def evaluate(
    model,
    tokenizer,
    layers,
    eval_items,
    support,
    cfg,
    probes,
    positions,
    batch_size,
    max_length,
    device,
):
    correct = {(layer_idx, pos): 0 for layer_idx in range(len(layers)) for pos in positions}
    total = {(layer_idx, pos): 0 for layer_idx in range(len(layers)) for pos in positions}

    for batch in batchify(eval_items, batch_size):
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
            max_length=max_length,
        ).to(device)
        attention_mask = inputs["attention_mask"].detach().cpu()

        activations = collect_mlp_activations(model, inputs, layers)

        for layer_idx, layer_act in enumerate(activations):
            if layer_act is None:
                continue
            layer_act = layer_act.float()
            for pos in positions:
                valid = attention_mask[:, pos].bool().numpy()
                if valid.sum() == 0:
                    continue
                key = (layer_idx, pos)
                X = layer_act[valid, pos, :].numpy()
                y = labels[valid]
                preds = probes[key].predict(X)
                correct[key] += int((preds == y).sum())
                total[key] += int(y.shape[0])

    return correct, total


def summarize_results(correct, total, positions, n_layers):
    rows = []
    for layer_idx in range(n_layers):
        for pos in positions:
            key = (layer_idx, pos)
            tot = total[key]
            acc = (correct[key] / tot) if tot > 0 else 0.0
            rows.append({
                "layer": layer_idx,
                "position": pos,
                "accuracy": acc,
                "n": tot,
            })

    # Best layer per position
    best_by_position = {}
    for pos in positions:
        candidates = [r for r in rows if r["position"] == pos and r["n"] > 0]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: r["accuracy"])
        best_by_position[pos] = {"layer": best["layer"], "accuracy": best["accuracy"]}

    # Best layer by mean accuracy across positions
    layer_scores = {}
    for layer_idx in range(n_layers):
        accs = [r["accuracy"] for r in rows if r["layer"] == layer_idx and r["n"] > 0]
        if accs:
            layer_scores[layer_idx] = float(np.mean(accs))
    best_layer = None
    if layer_scores:
        best_layer = max(layer_scores.items(), key=lambda x: x[1])

    summary = {
        "best_by_position": best_by_position,
        "mean_accuracy_by_layer": layer_scores,
        "best_layer_by_mean_accuracy": {
            "layer": best_layer[0],
            "mean_accuracy": best_layer[1],
        } if best_layer else None,
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description="Train layer/position linear probes on MLP activations.")
    parser.add_argument("--task", choices=["sst2", "imdb", "cola"], default="cola")
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--k_shot", type=int, default=16)
    parser.add_argument("--train_per_label", type=int, default=500)
    parser.add_argument("--eval_per_label", type=int, default=500)
    parser.add_argument("--positions", type=str, default="-1")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/probing")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, eval_ds, cfg = load_task_datasets(args.task)
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
    eval_items = sample_balanced_examples(
        eval_ds,
        per_label=args.eval_per_label,
        label_col=cfg["label_col"],
        text_col=cfg["text_col"],
        seed=args.seed + 1,
    )

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    layers = resolve_layers(model)
    positions = parse_positions(args.positions, args.max_length)

    probes, inited = init_probes(len(layers), positions, args.seed, args.alpha)

    for epoch in range(args.epochs):
        random.shuffle(train_items)
        train_epoch(
            model,
            tokenizer,
            layers,
            train_items,
            support,
            cfg,
            probes,
            inited,
            positions,
            args.batch_size,
            args.max_length,
            device,
        )
        print(f"Finished epoch {epoch + 1}/{args.epochs}")

    correct, total = evaluate(
        model,
        tokenizer,
        layers,
        eval_items,
        support,
        cfg,
        probes,
        positions,
        args.batch_size,
        args.max_length,
        device,
    )

    rows, summary = summarize_results(correct, total, positions, len(layers))

    os.makedirs(args.output_dir, exist_ok=True)
    model_slug = make_model_slug(args.model)
    base = os.path.join(args.output_dir, f"layer_position_probes_{args.task}_{model_slug}")
    csv_path = base + ".csv"
    json_path = base + ".json"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "position", "accuracy", "n"])
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    if summary.get("best_layer_by_mean_accuracy"):
        best = summary["best_layer_by_mean_accuracy"]
        print(f"Best layer by mean accuracy: layer {best['layer']} (acc={best['mean_accuracy']:.4f})")


if __name__ == "__main__":
    main()
