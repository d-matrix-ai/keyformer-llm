# experiment config
dataset_id = "Open-Orca/OpenOrca"

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import numpy as np
import os
import simplejson as json
import sys

save_dataset_path = os.environ.get("DATASET_ORCA_PATH", "../data")

# Check whether the specified path exists or not
isExist = os.path.exists(save_dataset_path)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(save_dataset_path)

# Load dataset from the hub
print("Loading Dataset!!")
dataset = load_dataset(dataset_id, cache_dir="../data/data_cache")
print("Dataset Loaded!!")


def preprocess_function(sample):
    # create list of samples
    inputs = []

    for i in range(0, len(sample["id"])):
        x = dict()
        x["instruction"] = sample["system_prompt"][i]
        x["input"] = sample["question"][i]
        x["output"] = sample["response"][i]

        inputs.append(x)
    model_inputs = dict()
    model_inputs["text"] = inputs

    return model_inputs


# process dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# save dataset to disk

with open(os.path.join(save_dataset_path, "orca_train.json"), "w") as write_f:
    json.dump(tokenized_dataset["train"]["text"], write_f, indent=4, ensure_ascii=False)


print("Dataset saved in ", save_dataset_path)
