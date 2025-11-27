# TEAL

code inherits from original TEAL project

## Histogram

before applying the activation sparsity, we need to get the magnitude for each layer. This is done by calculating from activation histogram we collect when model running inference.

To collect histogram

```bash
uv run grab_act.py --model_name $MODEL_NAME --dataset $DATASET --save_path $HISTOGRAM_PATH
```



