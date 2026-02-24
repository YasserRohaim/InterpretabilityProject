from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import random
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from evaluate import sample_fixed, build_prompt


SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# ---- Model / tokenizer ----
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", device_map="cuda")
model.eval()

# Keep the end of the prompt if truncation happens (where "acceptable:" is)
tokenizer.truncation_side = "left"
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def charpos_to_tokidx(offsets_row, char_pos: int):
    """
    offsets_row: Tensor [seq,2] (start,end) char offsets in original string.
    Returns token index whose span covers char_pos.
    """
    if char_pos < 0:
        return None
    for j in range(offsets_row.shape[0]):
        s = int(offsets_row[j, 0].item())
        e = int(offsets_row[j, 1].item())
        if s <= char_pos < e:
            return j
    return None


def main():
    # ---- SETTINGS (keep simple: edit these) ----
    K = 16
    BATCH_SIZE = 8
    MAX_LEN = 256
    LIMIT = None  # set e.g. 200 for quick debug; None = full validation
    OUT_DIR = "results/task3_cola_pca"

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Data ----
    train = load_dataset("nyu-mll/glue", "cola", split="train")
    val = load_dataset("nyu-mll/glue", "cola", split="validation")
    loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)

    support = sample_fixed(train, k=K)

    num_layers = len(model.model.layers)
    d_mlp = model.model.layers[0].mlp.gate_proj.out_features  # 9216 for Gemma2-2B

    N_total = len(val)
    N = N_total if LIMIT is None else min(LIMIT, N_total)

    # Store only what we need: two positions per layer
    acts_accept = torch.empty((num_layers, N, d_mlp), dtype=torch.float16, device="cpu")
    acts_colon  = torch.empty((num_layers, N, d_mlp), dtype=torch.float16, device="cpu")
    labels = torch.empty((N,), dtype=torch.long, device="cpu")

    # Shared state for hooks (updated per batch)
    state = {"accept_pos": None, "colon_pos": None}
    buf = {"accept": None, "colon": None}

    # Register hooks ONCE
    handles = []

    def make_hook(layer_idx, act_fn):
        def hook(mod, inp, out):
            # out: [B, seq, d_mlp] = gate_proj(...)
            B = out.size(0)
            idx = torch.arange(B, device=out.device)

            ap = state["accept_pos"]  # [B]
            cp = state["colon_pos"]   # [B]

            # Slice only two positions, then apply activation
            pre_a = out[idx, ap, :]   # [B, d_mlp]
            pre_c = out[idx, cp, :]   # [B, d_mlp]

            buf["accept"][layer_idx] = act_fn(pre_a).detach().cpu()
            buf["colon"][layer_idx]  = act_fn(pre_c).detach().cpu()

        return hook

    for li, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        handles.append(mlp.gate_proj.register_forward_hook(make_hook(li, mlp.act_fn)))

    # ---- Extraction loop ----
    cursor = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting (2 positions)"):
            if cursor >= N:
                break

            sentences = batch["sentence"]
            y = batch["label"]

            # respect LIMIT
            if cursor + len(sentences) > N:
                cutoff = N - cursor
                sentences = sentences[:cutoff]
                y = y[:cutoff]

            prompts = [build_prompt(support, s) for s in sentences]

            # character positions: LAST occurrence
            accept_char = [p.lower().rfind("acceptable") for p in prompts]
            colon_char  = [p.rfind(":") for p in prompts]  # last ':' before generation

            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping")  # CPU tensor [B, seq, 2]
            inputs = {k: v.to("cuda") for k, v in enc.items()}

            B = inputs["input_ids"].size(0)

            # map charpos -> token index
            accept_pos = []
            colon_pos = []
            for i in range(B):
                a = charpos_to_tokidx(offsets[i], accept_char[i])
                c = charpos_to_tokidx(offsets[i], colon_char[i])
                if a is None or c is None:
                    raise RuntimeError(
                        f"Could not map positions for sample {cursor+i}. "
                        f"(Try increasing MAX_LEN or ensure truncation_side='left')"
                    )
                accept_pos.append(a)
                colon_pos.append(c)

            # update hook state + buffers
            state["accept_pos"] = torch.tensor(accept_pos, device="cuda", dtype=torch.long)
            state["colon_pos"]  = torch.tensor(colon_pos, device="cuda", dtype=torch.long)
            buf["accept"] = [None] * num_layers
            buf["colon"]  = [None] * num_layers

            # forward pass triggers hooks
            _ = model(**inputs, use_cache=False, return_dict=True)

            # store labels + activations
            labels[cursor:cursor+B] = y.to("cpu").long()
            for li in range(num_layers):
                acts_accept[li, cursor:cursor+B, :] = buf["accept"][li].to(dtype=torch.float16)
                acts_colon[li,  cursor:cursor+B, :] = buf["colon"][li].to(dtype=torch.float16)

            cursor += B

    # cleanup hooks
    for h in handles:
        h.remove()

    labels_np = labels.numpy()

    # ---- PCA + plots ----
    for li in tqdm(range(num_layers), desc="PCA + plotting"):
        for name, Xsrc in [("acceptable", acts_accept), ("colon", acts_colon)]:
            X = Xsrc[li].float().numpy()  # [N, 9216] float32 for PCA
            Z = PCA(n_components=3, random_state=SEED).fit_transform(X)

            plt.figure()
            plt.scatter(Z[:, 0], Z[:, 1], c=labels_np, s=10, alpha=0.7)
            plt.title(f"CoLA PCA - Layer {li} - Position {name}")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"pca_layer{li:02d}_{name}.png"), dpi=200)
            plt.close()

    print(f"Saved plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()