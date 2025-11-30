# TEAL

code inherits from original TEAL project, the following steps will guide you through the reproduction.

## Histogram

Before applying the activation sparsity, we need to get the magnitude for each layer. This is done by calculating from activation histogram we collect when model running inference.

To collect histogram

```bash
uv run grab_act.py --model_name $MODEL_NAME --dataset $DATASET --histogram_path $HISTOGRAM_PATH
```

for example:

run llama-2-7b

```bash
uv run grab_act.py --model_name /data2/common/Llama-2-7b-hf --dataset /data2/common/dataset/wikitext --subset wikitext-103-raw-v1 --histogram_path /data2/common/ASP/teal/histogram/llama-2-7B
```

run llama-3-8b

```bash
uv run grab_act.py --model_name /data2/common/Meta-Llama-3-8B --dataset /data2/common/dataset/wikitext --subset wikitext-103-raw-v1 --histogram_path /data2/common/ASP/teal/histogram/llama-3-8B
```

run opt-6.7b

```bash
uv run grab_act.py --model_name /data2/common/opt-6.7b --dataset /data2/common/dataset/wikitext --subset wikitext-103-raw-v1 --histogram_path /data2/common/ASP/teal/histogram/opt-6.7B
```

## Perplexity

Now, we can use the histogram collected before to sparsify activations at inference runtime.

To test the model's accuarcy after activation sparsification, run

```bash
uv run test_ppl.py --model_name /data2/common/Llama-2-7b-hf --dataset /data2/common/dataset/wikitext --subset wikitext-103-raw-v1 --histogram_path /data2/common/ASP/teal/histogram/llama-2-7B
```

```bash
uv run test_ppl.py --model_name /data2/common/Meta-Llama-3-8B --dataset /data2/common/dataset/wikitext --subset wikitext-103-raw-v1 --histogram_path /data2/common/ASP/teal/histogram/llama-3-8B
```

run opt-6.7b

```bash
uv run test_ppl.py --model_name /data2/common/opt-6.7b --dataset /data2/common/dataset/wikitext --subset wikitext-103-raw-v1 --histogram_path /data2/common/ASP/teal/histogram/opt-6.7B
```

