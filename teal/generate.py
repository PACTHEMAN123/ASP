from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import sys,os
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from teal_utils import get_sparse_model

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--histogram_path', type=str)
args = parser.parse_args()

model_name = args.model_name
histogram_path = args.histogram_path
sparsity = 0.5

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = get_sparse_model(model_name, device="auto", histogram_path=histogram_path)
model.set_uniform_sparsity(sparsity)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

prompt = "Who is Max Verstappan."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

torch.cuda.synchronize()
start = time.time()

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_length=400,
        temperature=0.3,
        repetition_penalty=1.2,
        do_sample=False
    )

end = time.time()
torch.cuda.synchronize()

# print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

output_ids = output_ids[0]  # batch=1
num_generated_tokens = output_ids.shape[0] - inputs["input_ids"].shape[1]
latency = end - start
tps = num_generated_tokens / latency
print(f"Latency: {latency:.4f}s")
print(f"Speed: {tps:.2f} tokens/s")




