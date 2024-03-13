# experiment config
"""
====> Loading Longbanch Data
################################################################################################################
from datasets import load_dataset

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

for dataset in datasets:
    data = load_dataset('THUDM/LongBench', dataset, split='test')
################################################################################################################
    
====> Loading Longbanch-E Data (Uniformly Sampled Data)
################################################################################################################
from datasets import load_dataset

datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", \
            "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

for dataset in datasets:
    data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
################################################################################################################
    
################################################################################################################
{
    "input": "The input/command for the task, usually short, such as questions in QA, queries in Few-shot tasks, etc",
    "context": "The long context required for the task, such as documents, cross-file code, few-shot examples in Few-shot tasks",
    "answers": "A List of all true answers",
    "length": "Total length of the first three items (counted in characters for Chinese and words for English)",
    "dataset": "The name of the dataset to which this piece of data belongs",
    "language": "The language of this piece of data",
    "all_classes": "All categories in classification tasks, null for non-classification tasks",
    "_id": "Random id for each piece of data"
}
################################################################################################################
"""
dataset_id = "THUDM/LongBench"
dataset_name = "gov_report"
text_column = "context"
summary_column = "answers"
instruction_template_gov_reports = "You are given a report by a government agency. Write a one-page summary of the report."
instruction_template_multi_news = (
    "You are given several news passages. Write a one-page summary of all news."
)
instruction_template_passage_count = (
    "Determine the number of unique passages among the given set."
)
instruction_template_lcc = "Write the next line of code."

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import numpy as np
import os
import simplejson as json
import sys

save_dataset_path = os.environ.get("DATASET_CNNDM_PATH", "../data")

# Check whether the specified path exists or not
isExist = os.path.exists(save_dataset_path)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(save_dataset_path)

# Load dataset from the hub
dataset = load_dataset(
    dataset_id, name=dataset_name, cache_dir="../data/data_cache", split="test"
)


def preprocess_function(sample):
    # create list of samples
    inputs = []

    for i in range(0, len(sample[text_column])):
        x = dict()
        # x["instruction"] = sample["input"][i]
        x["instruction"] = instruction_template_gov_reports
        x["input"] = sample[text_column][i]
        x["output"] = sample[summary_column][i][0]
        # x["classes"] = sample["all_classes"][i]
        inputs.append(x)
    model_inputs = dict()
    model_inputs["text"] = inputs

    return model_inputs


# process dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# save dataset to disk

file_name = dataset_name + ".json"
with open(os.path.join(save_dataset_path, file_name), "w") as write_f:
    json.dump(tokenized_dataset["text"], write_f, indent=4, ensure_ascii=False)


print("Dataset saved in ", save_dataset_path)
