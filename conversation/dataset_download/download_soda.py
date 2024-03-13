# experiment config
dataset_id = "allenai/soda"

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import numpy as np
import os
import simplejson as json
import sys

save_dataset_path = os.environ.get("DATASET_SODA_PATH", "../data")

# Check whether the specified path exists or not
isExist = os.path.exists(save_dataset_path)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(save_dataset_path)

# Load dataset from the hub
dataset = load_dataset(dataset_id, cache_dir="../data/data_cache")


def preprocess_function(sample):
    # create list of samples
    inputs = []

    for i in range(0, len(sample["head"])):
        x = dict()
        x["head"] = sample["head"][i]
        x["relation"] = sample["relation"][i]
        x["tail"] = sample["tail"][i]
        x["literal"] = sample["literal"][i]
        x["narrative"] = sample["narrative"][i]
        x["dialogue"] = sample["dialogue"][i]
        x["speakers"] = sample["speakers"][i]
        x["PersonX"] = sample["PersonX"][i]
        x["PersonY"] = sample["PersonY"][i]
        x["PersonZ"] = sample["PersonZ"][i]
        x["original_index"] = sample["original_index"][i]
        x["split"] = sample["split"][i]
        x["head_answer"] = sample["head_answer"][i]
        x["pmi_head_answer"] = sample["pmi_head_answer"][i]
        x["relation_tail_answer"] = sample["relation_tail_answer"][i]
        x["pmi_relation_tail_answer"] = sample["pmi_relation_tail_answer"][i]

        inputs.append(x)
    model_inputs = dict()
    model_inputs["text"] = inputs

    return model_inputs


# process dataset
tokenized_dataset = dataset.map(
    preprocess_function, batched=True, remove_columns=list(dataset["train"].features)
)

# save dataset to disk

with open(os.path.join(save_dataset_path, "soda_eval.json"), "w") as write_f:
    json.dump(
        tokenized_dataset["validation"]["text"], write_f, indent=4, ensure_ascii=False
    )


print("Dataset saved in ", save_dataset_path)
