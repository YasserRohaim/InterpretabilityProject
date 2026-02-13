from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.metrics import classification_report
import random
import csv
import json


SEED=0
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

#You need to be logged in to your hf account in the env and agree to the terms in the model's page
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b",device_map="cuda")


def to_label(text: str):
    t = text.strip().lower()
    first = t.split()[0] if t else ""
    # the model was radndomly outputing 100 multiple times so I considered it as a positive lol 
    if "yes" in first or '100' in first or '1' in first:
        return 1
    if 'no' in first or '0' in first:
        return 0
    return None  # could not parse

def sample_fixed(dataset, k, label_col="label"):
    per_label = k // 2
    ds = dataset.shuffle(seed=SEED)

    sample = {0: [], 1: []}
    for  item in ds:
        label = int(item[label_col])
        if label in sample and len(sample[label]) < per_label:
            sample[label].append(item["sentence"])
            if len(sample[0]) == per_label and len(sample[1]) == per_label:
                break
    
    print(sample)
    return sample

def prepare_train_pools(train_ds):
    """
    Pre-split training sentences by label for fast random sampling per example.
    """
    pools = {0: [], 1: []}
    for item in train_ds:
        pools[int(item["label"])].append(item["sentence"])
    return pools

def sample_random_from_pools(pools, k, rng: random.Random):
    """
    Balanced random k-shot sample drawn from pools (k/2 per label).
    """
    per_label = k // 2
    return {
        0: rng.sample(pools[0], per_label),
        1: rng.sample(pools[1], per_label),
    }

def build_prompt(sample, sentence: str) -> str:
    """
    Build the full prompt (support examples + query) for one sentence.
    Avoids .format() so braces in text won't break formatting.
    """
    prompt = ""
    for label, sents in sample.items():
        label_str = "yes" if int(label) == 1 else "no"
        for sent in sents:
            prompt += f"Sentence: {sent}\nLinguistically acceptable: {label_str}\n\n"
    prompt += f"Sentence: {sentence}\nLinguistically acceptable: "
    return prompt
    

train= load_dataset("nyu-mll/glue",'cola',split="train")
val= load_dataset("nyu-mll/glue",'cola',split="validation")
val_loader=DataLoader(val,batch_size=16,num_workers=2)


parser= ArgumentParser()
parser.add_argument('--k', type=int, help='number of examples in few-shot',default=8)
parser.add_argument('--example_strat', choices=['random','fixed'],default='fixed')

def main():
    args = parser.parse_args()
    assert args.k % 2 == 0

    base = f"res_k={args.k}_{args.example_strat}"
    csv_path = base + ".csv"
    json_path = base + ".json"

    gt = val["label"]
    preds = []
    rows = []

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Fixed strategy: choose support once
    fixed_support = None
    if args.example_strat == "fixed":
        fixed_support = sample_fixed(train, args.k)

    # Random strategy: prepare pools once, then sample per validation example
    rng = random.Random(SEED)
    pools = None
    if args.example_strat == "random":
        pools = prepare_train_pools(train)

    with torch.no_grad():
        for batch in tqdm(val_loader):
            sentences = batch["sentence"]
            labels = batch["label"]  # <-- ADD: grab batch ground truth

            if args.example_strat == "fixed":
                prompts = [build_prompt(fixed_support, s) for s in sentences]
            else:
                prompts = []
                for s in sentences:
                    support = sample_random_from_pools(pools, args.k, rng)
                    prompts.append(build_prompt(support, s))

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to("cuda")

            outputs = model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            gen = outputs[:, inputs["input_ids"].shape[1]:]
            gen_text = tokenizer.batch_decode(gen, skip_special_tokens=True)

            batch_preds = [to_label(t) for t in gen_text]
            preds.extend(batch_preds)

            # <-- EDIT: include gt in each CSV row
            for p, y, pr, g in zip(prompts, labels, batch_preds, gen_text):
                rows.append({
                    "prompt": p,
                    "gt": int(y),
                    "pred": pr,
                    "generated_text": g
                })

    # JSON classification report
    report = classification_report(
        gt,
        preds,
        labels=[0, 1],
        target_names=["no", "yes"],
        output_dict=True,
        zero_division=0,
    )

    # Save CSV (prompt, gt, pred, generated text)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "gt", "pred", "generated_text"])
        writer.writeheader()
        writer.writerows(rows)

    # Save JSON report
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print paths + pretty report for convenience
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(classification_report(gt, preds, labels=[0, 1], target_names=["no", "yes"], zero_division=0))

if __name__ == "__main__":
    main()
