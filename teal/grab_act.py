import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))

from teal_utils import (
    get_tokenizer,
    get_sparse_model
)
from tqdm import tqdm
import torch
import gc
from utils.data import get_dataset

model_name = "/data2/common/Llama-2-7b-hf"
dataset_name = "/data2/common/dataset/wikitext"
subset_name = "wikitext-103-raw-v1"
histogram_path = "/data2/common/ASP/teal/histogram/llama-2-7B"

tokenizer = get_tokenizer(model_name)

# add a sparse model here
model = get_sparse_model(model_name, device="auto", histogram_path=histogram_path, grab_acts=True)


# pick random words to run inference
dataset = get_dataset(dataset_name, subset=subset_name, split='train', size=300)
text = ""
for sample in tqdm(dataset):
    text += sample["text"] + "\n\n"
print(f"total length of data to generate histogram {len(text)}")


seq_len = 2048
encodings = tokenizer(text, truncation=True, return_tensors="pt", max_length=seq_len, return_overflowing_tokens=True, padding="max_length")
input_ids = encodings.input_ids.to(device="cuda")
bsz = input_ids.shape[0]
input_ids = encodings.input_ids[:bsz,:].to(device="cuda")
print(f"input ids shape {input_ids.shape}")

hidden_states = model.model.embed_tokens(input_ids)
print(f"hidden_states {hidden_states.shape}")

attention_mask = None
position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).repeat(bsz, 1)
print(f"pos id {position_ids.shape}")
past_key_value=None
#output_attentions = False
use_cache = False
cache_position=None
position_embeddings = model.model.rotary_emb(hidden_states, position_ids)


# act_path = os.path.join(args.output_path, "activations")
# os.makedirs(act_path, exist_ok=True)

# pass the activation layer by layer, and save the histograms for later pruning
# copy code from LlamaModel forward
for i in tqdm(range(len(model.model.layers))):
    # for greedyopt
    # torch.save(hidden_states, os.path.join(act_path, f"act_{i}.pt"))

    layer = model.model.layers[i]
    hidden_states = hidden_states.to(layer.self_attn.q_proj.weight.data.device) 
    hidden_states = layer(hidden_states, attention_mask, position_ids, past_key_value, use_cache, cache_position, position_embeddings)

    layer.mlp.activation_module.find_histogram()
    layer.self_attn.activation_module.find_histogram()
    layer.mlp.activation_module.save_histogram()
    layer.self_attn.activation_module.save_histogram()

    del layer.mlp.activation_module.activations
    del layer.self_attn.activation_module.activations
    
    model.model.layers[i] = None

    gc.collect()
    torch.cuda.empty_cache()

print("successfully collect all histogram")
print(f"model: {model_name}, dataset/subset: {dataset_name}/{subset_name}")





