from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "/data2/common/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "解释一下量子纠缠是什么"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_length=400,
        temperature=0.7,
        repetition_penalty=1.2,
        do_sample=False
    )

# 5. 解码输出
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))