from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import sys,os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utils.eval_ppl import eval_ppl
from utils.data import get_dataset
from teal_utils import get_sparse_model

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--subset', type=str)
parser.add_argument('--histogram_path', type=str)
args = parser.parse_args()

model_name = args.model_name
dataset_name = args.dataset
subset_name = args.subset
histogram_path = args.histogram_path
sparsity = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

sparse_model = get_sparse_model(model_name, device="auto", histogram_path=histogram_path)
dataset = get_dataset(dataset_name, subset_name, split="train", size=250)

for sparse_ratio in sparsity:
    print("="*20)
    print(f"Evaluating sparse PPL at sparsity {sparse_ratio}")

    sparse_model.set_uniform_sparsity(sparse_ratio)
    sparse_ppl = eval_ppl(sparse_model, tokenizer, device="cuda", dataset=dataset, debug=False, context_size=1024, window_size=256)

    print(f"PPL: {sparse_ppl}")


