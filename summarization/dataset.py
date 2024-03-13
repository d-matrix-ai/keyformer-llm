import os
import time
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from typing import Optional, Dict, Sequence
import io
import utils
import copy

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(
        self,
        dataset_path,
        tokenizer,
        model_name,
        return_tensors,
        truncation,
        padding,
        max_length=None,
        total_count_override=None,
        perf_count_override=None,
    ):
        self.dataset = "cnn_dailymail"
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.return_tensors = return_tensors
        self.truncation = truncation
        self.padding = padding
        self.max_length = max_length

        self.list_data_dict = utils.jload(self.dataset_path)

        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )
        self.sources = [
            prompt_input.format_map(example) for example in self.list_data_dict
        ]
        self.targets = [f"{example['output']}" for example in self.list_data_dict]

        # Getting random samples for evaluation
        if total_count_override > 0:
            self.rand_samples = np.random.randint(
                len(self.sources), size=(total_count_override)
            ).tolist()
            self.sources = [self.sources[i] for i in self.rand_samples]
            self.targets = [self.targets[i] for i in self.rand_samples]
            self.count = total_count_override
        else:
            self.count = len(self.sources)

        (
            self.source_encoded_input_ids,
            self.source_encoded_attn_masks,
        ) = self.encode_samples()

        self.perf_count = perf_count_override or self.count

    def encode_samples(self):
        print("Encoding Samples")

        total_samples = self.count

        source_encoded_input_ids = []
        source_encoded_attn_masks = []

        for i in range(total_samples):
            source_encoded = self.tokenizer(
                self.sources[i],
                return_tensors=self.return_tensors,
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
            )
            source_encoded_input_ids.append(source_encoded.input_ids)
            source_encoded_attn_masks.append(source_encoded.attention_mask)

        return source_encoded_input_ids, source_encoded_attn_masks

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return (
            self.source_encoded_input_ids[index],
            self.source_encoded_attn_masks[index],
        )
