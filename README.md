

# ASP

Activation SParsity test framework.

## Installation

### Install uv

`uv` is a fast Python package and environment manager. We use `uv` for Python package and environment management

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Or see detailed instructions: https://docs.astral.sh/uv/


## Environment Setup

### uv

`uv` automatically reads the pyproject.toml file and resolves dependencies and downloads:

```sh
uv sync
```

Run Python inside the environment:

```sh
export CUDA_VISIBLE_DEVICES=4,5,6,7
uv run python main.py
```