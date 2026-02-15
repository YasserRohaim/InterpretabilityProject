from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
import random
from evaluate import sample_fixed, build_prompt


SEED=0
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

#You need to be logged in to your hf account in the env and agree to the terms in the model's page
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b",device_map="cuda")
model.eval()

def extract_hidden_and_mlp_activations(text, max_length=512, save_path="results/cola_first_sample_acts.pt"):
    inputs=tokenizer(text, return_tensors="pt", max_length=max_length,truncation=True).to("cuda")
    handles=[]
    mlp_acts={}
    for i, layer in enumerate(model.model.layers):
        mlp= layer.mlp
        act_fn=mlp.act_fn
        def make_hook(layer_idx, act):
                def hook(mod, inp, out):
                    # out is gate_proj(x); activation values are act_fn(out)
                    mlp_acts[layer_idx] = act(out).detach().cpu()
                return hook
        handles.append(mlp.gate_proj.register_forward_hook(make_hook(i, act_fn)))

    outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)

    for h in handles:
        h.remove()

    hidden_states=[hs.detach().cpu() for hs in outputs.hidden_states]

    activations = [v for _, v in sorted(mlp_acts.items())]

    payload = {
        "text": text,
        "input_ids": inputs["input_ids"].detach().cpu(),
        "attention_mask": inputs.get("attention_mask", None).detach().cpu() if "attention_mask" in inputs else None,
        "hidden_states": hidden_states,          # list: len n_layers+1, each [seq, hidden]
        "mlp_activations": activations,      # list: len n_layers, each [seq, intermediate]
    }
    torch.save(payload, save_path)

    print("Num hidden_states:", len(hidden_states))
    print("Num mlp_activations:", len(activations))
    print("hidden_states[1] shape:", hidden_states[1].shape)      # [seq, hidden]
    print("mlp_activations[0] shape:", activations[0].shape)  # [seq, intermediate]
    print("Different shapes? ->", hidden_states[1].shape != activations[0].shape)

    return payload



def main():
    train= load_dataset("nyu-mll/glue",'cola',split="train")
    val= load_dataset("nyu-mll/glue",'cola',split="validation")

    support=sample_fixed(train, k=16)
    sent=val[0]["sentence"]
    prompt=build_prompt(support, sent)

    extract_hidden_and_mlp_activations(prompt)

if __name__=="__main__":
    main()

