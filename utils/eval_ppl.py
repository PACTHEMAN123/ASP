import sys,os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))

import torch
from tqdm import tqdm
import os

def eval_ppl(model, tokenizer, device,dataset=None, debug=False, context_size=2048, window_size=512):
    text = ""
    for sample in dataset:
        text += sample["text"] + "\n\n"

    encodings = tokenizer(text, return_tensors="pt")

    if debug:
        print(tokenizer.decode(encodings.input_ids[0][:100]))

    max_length = context_size + window_size
    stride = window_size
    seq_len = encodings.input_ids.size(1)
    # make seq_len a multiple of stride
    seq_len = seq_len - (seq_len % stride)

    if debug:
        print(f"seq_len: {seq_len}")

    if debug:
        pbar = tqdm(range(0, seq_len, stride))
    else:
        pbar = range(0, seq_len, stride)

    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)

    if model_vocab_size != tokenizer_vocab_size:
        # resize model embeddings to fit tokenizer
        if model_vocab_size != tokenizer_vocab_size:
            print("Resize model embeddings to fit tokenizer")
        model.resize_token_embeddings(tokenizer_vocab_size)

    model.eval()
    nlls = []

    for begin_loc in pbar:
        end_loc = begin_loc + max_length
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # print(outputs.logits.shape)
            neg_log_likelihood = outputs.loss

        if debug:
            pbar.set_description(
                f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
            )

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).double().mean())

    return ppl.item()

# only use for debug
def compare_models(model1, model2, tokenizer, dataset):
    import gc

    text = ""
    for sample in tqdm(dataset):
        text += sample["text"] + "\n\n"
    print(f"total length of data to generate histogram {len(text)}")

    seq_len = 512
    encodings = tokenizer(text, truncation=True, return_tensors="pt", max_length=seq_len, return_overflowing_tokens=True, padding="max_length")
    input_ids = encodings.input_ids.to(device="cuda")
    bsz = input_ids.shape[0]
    input_ids = encodings.input_ids[:bsz,:].to(device="cuda")
    print(f"input ids shape {input_ids.shape}")

    hidden_states1 = model1.model.decoder.embed_tokens(input_ids)
    hidden_states2 = model2.model.decoder.embed_tokens(input_ids)
    print(f"init, diff: {(hidden_states2 - hidden_states1).abs().max().item()}")

    attention_mask = None
    position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states1.device).unsqueeze(0).repeat(bsz, 1)
    print(f"pos id {position_ids.shape}")
    past_key_value=None
    output_attentions = False
    use_cache = False
    cache_position=None

    layer1 = model1.model.decoder.layers[0]
    layer2 = model2.model.decoder.layers[0]
    hidden_states1 = hidden_states1.to(layer1.self_attn.q_proj.weight.data.device) 
    hidden_states2 = hidden_states2.to(layer2.self_attn.q_proj.weight.data.device) 
    hidden_states1 = layer1(hidden_states1, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position)[0]
    hidden_states2 = layer2(hidden_states2, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position)[0]

    print(f"layer 0, diff: {(hidden_states2 - hidden_states1).abs().max().item()}")