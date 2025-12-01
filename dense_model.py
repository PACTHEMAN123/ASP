from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import sys,os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utils.eval_ppl import eval_ppl
from utils.data import get_dataset

model_name = "/data2/common/opt-6.7b"
dataset_name = "/data2/common/dataset/wikitext"
subset_name = "wikitext-103-raw-v1"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("================= test1 =================")

prompt = "explain what is CUDA to me."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_length=400,
        temperature=0.7,
        repetition_penalty=1.2,
        do_sample=False
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

print("================= test2 =================")

print("Evaluating dense PPL")

dataset = get_dataset(dataset_name, subset_name, split="train", size=250)
dense_ppl = eval_ppl(model, tokenizer, device="cuda", dataset=dataset, debug=False, context_size=1024, window_size=256)

print(f"PPL: {dense_ppl}")


