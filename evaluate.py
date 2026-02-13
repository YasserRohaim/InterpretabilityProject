from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.metrics import classification_report
import random


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
    """
    Return a balanced dataset with k examples total: k/2 per label.
    Assumes binary labels
    """
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


def build_prompt_fixed(sample):
    prompt=""
    #print(sample)
    for label,sents in sample.items() :
        if label:
            label='yes'
        else:
            label='no'
        for sent in sents:
            prompt+=f"Sentence: {sent}\nLinguistically acceptable: {label}\n\n"
    prompt+="Sentence: {sentence}\nLinguistically acceptable: "
    return prompt
    

train= load_dataset("nyu-mll/glue",'cola',split="train")
val= load_dataset("nyu-mll/glue",'cola',split="validation")
val_loader=DataLoader(val,batch_size=16,num_workers=2)


parser= ArgumentParser()
parser.add_argument('--k', type=int, help='number of examples in few-shot',default=8)
parser.add_argument('--example_strat', choices=['random','fixed'],default='fixed')
def main():
    gt=val["label"]
    preds=[]
    args= parser.parse_args()
    #assert that it is divisible by no of classes (2)
    assert(args.k%2==0)
    if args.example_strat=='fixed':
        sample=sample_fixed(train,args.k)
        prompt=build_prompt_fixed(sample)
        print(prompt)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    acc=[]
    with torch.no_grad():
        for batch in tqdm(val_loader):
            sentences = batch["sentence"]         
            labels = batch["label"]              

            prompts = [prompt.format(sentence=s) for s in sentences]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,          # <-- FIX: pad for batching
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

            acc+=gen_text
            preds += [to_label(t) for t in gen_text]
    for i in range (len(preds)):
        print(i,acc[i],preds[i])
    print(classification_report(gt,preds))

        
main()
