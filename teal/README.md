# TEAL

code inherits from original TEAL project, the following steps will guide you through the reproduction.

## Histogram

Before applying the activation sparsity, we need to get the magnitude for each layer. This is done by calculating from activation histogram we collect when model running inference.

To collect histogram

```bash
uv run grab_act.py --model_name $MODEL_NAME --dataset $DATASET --histogram_path $HISTOGRAM_PATH
```

## Perplexity

Now, we can use the histogram collected before to sparsify activations at inference runtime.

To test the model's accuarcy after activation sparsification, run

```bash
uv run test_ppl.py --model_name $MODEL_NAME --dataset $DATASET --histogram_path $HISTOGRAM_PATH
```

