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

# test decode time separately
def generate_with_timing(model, input_ids, max_length=400):
    """
    Custom generation function with separate timing for prefill and decode phases.
    
    Args:
        model: Causal LM model
        input_ids: Input token ids
        max_length: Maximum length to generate
        measure_decode_only: If True, only time the decode phase; if False, time both
    
    Returns:
        output_ids: Generated token ids
        timing_info: Dict with timing information
    """
    input_length = input_ids.shape[1]
    
    timing_info = {
        'prefill_time': 0,
        'decode_time': 0,
        'total_time': 0,
        'num_decode_tokens': 0
    }
    
    total_start = time.time()
    
    # ==================== PREFILL PHASE ====================
    prefill_start = time.time()
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
    
    torch.cuda.synchronize()
    prefill_end = time.time()
    timing_info['prefill_time'] = prefill_end - prefill_start
    
    # Get first generated token
    next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
    output_ids = torch.cat([input_ids, next_tokens], dim=1)
    
    # ==================== DECODE PHASE ====================
    decode_start = time.time()
    
    for step in range(input_length + 1, max_length):
        with torch.no_grad():
            outputs = model(
                next_tokens,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
        
        next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
        output_ids = torch.cat([output_ids, next_tokens], dim=1)
    
    torch.cuda.synchronize()
    decode_end = time.time()
    
    timing_info['decode_time'] = decode_end - decode_start
    timing_info['num_decode_tokens'] = output_ids.shape[1] - input_length - 1
    timing_info['total_time'] = decode_end - total_start
    
    return output_ids, timing_info

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

prompt = "Who is Max Verstappan."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

torch.cuda.synchronize()

with torch.no_grad():
    output_ids, timing_info = generate_with_timing(
        model,
        inputs["input_ids"],
        max_length=400
    )

print(f"\nPrefill Time: {timing_info['prefill_time']:.4f}s")
print(f"Decode Time: {timing_info['decode_time']:.4f}s")
print(f"Total Time: {timing_info['total_time']:.4f}s")
print(f"\nGenerated tokens: {timing_info['num_decode_tokens']}")
print(f"Decode Speed: {timing_info['num_decode_tokens'] / timing_info['decode_time']:.2f} tokens/s")
print(f"Overall Speed (including prefill): {timing_info['num_decode_tokens'] / timing_info['total_time']:.2f} tokens/s")

# print(tokenizer.decode(output_ids[0], skip_special_tokens=True))




