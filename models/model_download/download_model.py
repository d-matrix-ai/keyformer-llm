import argparse
import sys
import os
import torch
import transformers
from transformers import AutoModelForCausalLM

model_path = os.path.join(os.getcwd(), "model")

if not os.path.exists(os.path.dirname(model_path)):
    os.mkdir(model_path)

def download_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torchscript=True, trust_remote_code=True
    )  # torchscript will force `return_dict=False` to avoid jit errors
    print("Loaded model")

    model.save_pretrained(model_path)

    print("Model downloaded and Saved in : ", model_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="like cerebras/Cerebras-GPT-6.7B, etc."
    )
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()

    # This check will be relaxed as more models are supported.
    supported_models = ['cerebras/Cerebras-GPT-6.7B', 
        'mosaicml/mpt-7b', 'EleutherAI/gpt-j-6B']
    if args.model_name not in supported_models:
        raise Exception(f'Unsupported Model Name,. Only models in ' \
        f'{supported_models} are supported.')
    
    download_model(args.model_name)