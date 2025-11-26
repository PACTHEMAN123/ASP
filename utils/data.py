
from datasets import load_dataset

def get_dataset(dataset_name, subset, split, size=None, start=0):
    if size is None:
        dataset = load_dataset(dataset_name, subset)[split]
    else:
        dataset = load_dataset(dataset_name, subset, streaming=True)[split]
        dataset = dataset.skip(start).take(size)

    return dataset