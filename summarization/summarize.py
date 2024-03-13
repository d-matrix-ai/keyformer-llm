# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Test the optimizations on Huggingface to make sure the changes do not affect the
model accuracy.
"""
import time
import argparse
import os
import csv
import sys
import json
import logging
import copy
import numpy as np
from pathlib import Path
from tqdm import tqdm
from operator import itemgetter
import evaluate
import nltk

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from transformers.generation import (
    BeamSearchEncoderDecoderOutput,
    BeamSearchDecoderOnlyOutput,
    GreedySearchDecoderOnlyOutput,
    GreedySearchEncoderDecoderOutput,
    SampleDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
    BeamSampleDecoderOnlyOutput,
    BeamSampleEncoderDecoderOutput,
)

from dataset import Dataset
from utils import (
    use_task_specific_params,
    trim_batch,
    calculate_rouge,
    calculate_bleu_score,
    postprocess_text,
)


logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GENERATE_FINISHED = "done"
POSTPROCESS_FINISHED = None


def generate_summary(
    dataset_path: str,
    score_path: str,
    out_file: str,
    attentions_path: str,
    model_name: str,
    model_path: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    dtype="float32",
    task="summarization",
    decoder_start_token_id=None,
    no_repeat_ngram_size=None,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
    preprocess_workers=2,
    postprocess_workers=2,
    return_tensors="pt",
    truncation=True,
    save_attentions=True,
    save_prompt_attentions=False,
    padding="max_length",
    max_tokenizer_length=None,
    max_gen_length=None,
    max_new_tokens=None,
    min_gen_length=None,
    use_causal_lm=False,
    early_stopping=True,
    output_summaries_only=False,
    output_sequence_scores=False,
    model_parallelize=False,
    keyformer=False,
    kv_cache=60,
    recent=30,
    tau_init=1.0,
    tau_end=2.0,
    num_beams=None,
    eos_token_id=None,
    temperature=None,
    top_k=None,
    top_p=None,
    do_sample=None,
    repetition_penalty=None,
    num_return_sequences=None,
    padding_side=None,
    use_slow_tokenizer=False,
    n_obs=2,
    **gen_kwargs,
) -> None:
    """Run generation"""
    fout = Path(out_file).open("w", encoding="utf-8")
    model_name = str(model_name)
    model_path = str(model_path)
    dataset_path = str(dataset_path)

    if dtype == "bfloat16":
        amp_enabled = True
        amp_dtype = torch.bfloat16
        print("BF16 autocast")
    elif dtype == "float16":
        amp_enabled = True
        amp_dtype = torch.float16
        print("FP16 autocast")
    else:
        amp_enabled = False
        amp_dtype = torch.float32
        print("FP32 autocast")

    print("=============================")
    print("Getting Model and Tokenizer!!")
    print("=============================")

    if model_name.split("/")[0] == "mosaicml":
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, output_attentions=save_attentions
        )
        config.attn_config["attn_impl"] = "torch"
        config.init_device = "cuda:0"
        config.use_cache = True
        config.torch_dtype = dtype
        config.keyformer_config["keyformer"] = keyformer
        config.keyformer_config["kv_cache"] = kv_cache
        config.keyformer_config["recent"] = recent
        config.keyformer_config["tau_init"] = tau_init
        config.keyformer_config["tau_delta"] = (tau_end - tau_init) / max_new_tokens
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=amp_dtype,
            trust_remote_code=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b", use_fast=not use_slow_tokenizer
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    elif model_name == "EleutherAI/gpt-j-6B":
        config = AutoConfig.from_pretrained(
            model_path, output_attentions=save_attentions, trust_remote_code=True
        )
        config.keyformer_config["keyformer"] = keyformer
        config.keyformer_config["kv_cache"] = kv_cache
        config.keyformer_config["recent"] = recent
        config.keyformer_config["tau_init"] = tau_init
        config.keyformer_config["tau_delta"] = (tau_end - tau_init) / max_new_tokens
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=amp_dtype, trust_remote_code=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=not use_slow_tokenizer
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    else:
        config = AutoConfig.from_pretrained(
            model_path, output_attentions=save_attentions, trust_remote_code=True
        )
        config.keyformer_config["keyformer"] = keyformer
        config.keyformer_config["kv_cache"] = kv_cache
        config.keyformer_config["recent"] = recent
        config.keyformer_config["tau_init"] = tau_init
        config.keyformer_config["tau_delta"] = (tau_end - tau_init) / max_new_tokens
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=amp_dtype, trust_remote_code=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=not use_slow_tokenizer
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    if model_parallelize:
        print("\n======================================")
        print(f"Parallelizing model across {torch.cuda.device_count()} GPUs!!")
        print("========================================")
        device_map = None
        model.parallelize(device_map)

    if decoder_start_token_id is None:
        decoder_start_token_id = gen_kwargs.pop("decoder_start_token_id", None)
    if hasattr(tokenizer, "model_max_length") and max_tokenizer_length is not None:
        tokenizer.model_max_length = max_tokenizer_length
    if padding_side is not None:
        tokenizer.padding_side = padding_side

    # update config with summarization specific params
    use_task_specific_params(model, task)

    model.eval()

    print("=============================")
    print("Getting Dataset!!")
    print("=============================")

    dataset = Dataset(
        dataset_path,
        tokenizer,
        model_name,
        return_tensors,
        truncation,
        padding,
        max_tokenizer_length,
        n_obs,
    )

    training_generator = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, drop_last=False
    )

    print("=============================")
    print("Start Summary!!")
    print("=============================")

    attentions = []

    try:
        for ind, batch in tqdm(enumerate(training_generator)):
            input_ids, attention_mask = batch
            input_ids = input_ids.view(input_ids.size(0), -1).to(device)
            attention_mask = attention_mask.view(input_ids.size(0), -1).to(device)
            input_batch = dict()
            input_batch["input_ids"] = input_ids
            input_batch["attention_mask"] = attention_mask
            try:
                summaries = model.generate(
                    **input_batch,
                    decoder_start_token_id=decoder_start_token_id,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    max_length=max_gen_length,
                    max_new_tokens=max_new_tokens,
                    min_length=min_gen_length,
                    output_scores=output_sequence_scores,
                    return_dict_in_generate=output_sequence_scores,
                    output_attentions=save_attentions,
                    num_beams=num_beams,
                    eos_token_id=eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=num_return_sequences,
                    early_stopping=early_stopping,
                    use_cache=config.use_cache,
                    **gen_kwargs,
                )
            except:
                logger.exception(sys.exc_info()[0])
                sys.exit(1)
            if output_sequence_scores:
                sequences = summaries.sequences
            else:
                sequences = summaries

            scores_cpu = None

            if save_attentions:
                # Attention Weights
                curr_attn = summaries.attentions
                itrs = len(curr_attn)
                layers = len(curr_attn[0])

                itr_attn = []
                if save_prompt_attentions:
                    for i in range(itrs):
                        if i == 0:
                            layer_attn = []
                            for j in range(layers):
                                layer_attn.append(curr_attn[i][j].cpu().numpy())
                            itr_attn.append(layer_attn)
                            break
                        attentions.append(itr_attn)
                else:
                    for i in range(itrs):
                        if i == 0:
                            continue
                        layer_attn = []
                        for j in range(layers):
                            layer = curr_attn[i][j].cpu().numpy()
                            layer_shape = layer.shape
                            zero_size = max_new_tokens - i
                            zero_layer = np.zeros(
                                (
                                    layer_shape[0],
                                    layer_shape[1],
                                    layer_shape[2],
                                    zero_size,
                                )
                            )
                            layer_array = np.append(layer, zero_layer, axis=3)
                            layer_attn.append(layer_array)
                        itr_attn.append(layer_attn)
                    attentions.append(itr_attn)

            if output_summaries_only:
                sequences = sequences[:, input_ids.shape[-1] :]
            sequences_cpu = sequences.cpu()

            dec = tokenizer.batch_decode(
                sequences_cpu,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for i, hypothesis in enumerate(dec):
                hypothesis = hypothesis.replace("\n", " ")
                score = ""
                if scores_cpu is not None:
                    if isinstance(scores_cpu[i], str):
                        score = scores_cpu[i] + "\t"
                    else:
                        score = "%f\t" % scores_cpu[i].item()
                fout.write(score + hypothesis + "\n")
                fout.flush()

    except:
        logger.exception(sys.exc_info()[0])
        sys.exit(1)
    fout.close()

    if model_parallelize:
        print("\n=============================")
        print(f"Deparallelizing models across {torch.cuda.device_count()} GPUs!!")
        print("=============================")
        model.deparallelize()

    print("=============================")
    print("Evaluating Score!!")
    print("=============================")

    metric = evaluate.load("rouge")
    nltk.download("punkt")

    target_required = copy.deepcopy(dataset.targets)
    preds_decoded_text = [x for x in open(out_file).readlines()]

    preds, targets = postprocess_text(preds_decoded_text, target_required)

    result = metric.compute(
        predictions=preds, references=targets, use_stemmer=True, use_aggregator=False
    )
    result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
    prediction_lens = [len(pred) for pred in preds]
    result["gen_len"] = np.sum(prediction_lens)
    result["gen_num"] = len(preds)

    print("\n")
    print("=============================")
    print("ROUGE Score!!")
    print("=============================")
    print(result)

    result = {k: float(v) for k, v in result.items()}

    if score_path is not None:
        json.dump(result, open(score_path, "w+"))

    if save_attentions:
        print("\n")
        print("=============================")
        print("Saving Attention Weights!!")
        print("=============================")

        attentions_save = np.array(attentions, dtype=object)
        np.savez_compressed(attentions_path, attentions_save)

        print("Saving attention weights completed!!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, help="like cerebras/Cerebras-GPT-6.7B, etc."
    )
    parser.add_argument("--dataset_path", type=str, help="like cnn_dm/cnn_eval.json")
    parser.add_argument("--save_path", type=str, help="where to save summaries")

    parser.add_argument(
        "--model_path", type=str, default="EleutherAI/gpt-j-6B", required=False, help=""
    )
    parser.add_argument(
        "--score_path",
        type=str,
        required=False,
        help="where to save the rouge score in json format",
    )
    parser.add_argument(
        "--attentions_path",
        type=str,
        required=False,
        help="where to save the attention weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default=DEFAULT_DEVICE,
        help="cuda, cuda:1, cpu etc.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="summarization",
        help="typically translation or summarization",
    )
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument(
        "--decoder_start_token_id",
        type=int,
        default=None,
        required=False,
        help="decoder_start_token_id (otherwise will look at config)",
    )
    parser.add_argument(
        "--n_obs",
        type=int,
        default=-1,
        required=False,
        help="How many observations. Defaults to all.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="data type of the model, choose from float16, bfloat16 and float32",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=None,
        required=False,
        help="size of no repeat ngram",
    )
    parser.add_argument("--include_special_tokens", action="store_true")
    parser.add_argument("--clean_up_tokenization_spaces", action="store_true")
    parser.add_argument(
        "--preprocess_workers",
        type=int,
        default=2,
        required=False,
        help="pre-processing worker threads",
    )
    parser.add_argument(
        "--postprocess_workers",
        type=int,
        default=1,
        required=False,
        help="post-processing worker threads",
    )
    parser.add_argument("--no_truncation", action="store_true")
    parser.add_argument("--save_attentions", action="store_true")
    parser.add_argument(
        "--save_prompt_attentions",
        action="store_true",
        help="when enabled, only prompt attentions are saved!!",
    )
    parser.add_argument(
        "--return_tensors",
        type=str,
        help="specify return tensors",
        default="pt",
        required=False,
    )
    parser.add_argument(
        "--padding",
        type=str,
        help="specify padding",
        default="max_length",
        required=False,
    )
    parser.add_argument(
        "--max_tokenizer_length",
        type=int,
        help="max length for the tokenized sentence",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--max_gen_length",
        type=int,
        help="max length for generation",
        default=None,
        required=False,
    ),
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="max tokens to generate, ignoring the number of "
        "current tokens. Use either max_gen_length or "
        "max_new_tokens, but not both - they serve the same purpose.",
        default=None,
        required=False,
    ),
    parser.add_argument(
        "--min_gen_length",
        type=int,
        default=-1,
        required=False,
        help="min length for decode",
    )
    parser.add_argument("--causal_lm", action="store_true")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--output_summaries_only", action="store_true")
    parser.add_argument("--output_sequence_scores", action="store_true")
    parser.add_argument(
        "--model_parallelize",
        action="store_true",
        help="Model parallelism enabled across available GPUs",
    )
    ############################################### Keyformer #################################################################
    parser.add_argument(
        "--keyformer", action="store_true", help="Keyformer enabled - reduced KV cache"
    )
    parser.add_argument(
        "--kv_cache",
        type=float,
        default=60,
        required=False,
        help="KV cache percentage for Keyformer",
    )
    parser.add_argument(
        "--recent",
        type=float,
        default=30,
        required=False,
        help="Recent window percentage for Keyformer",
    )
    parser.add_argument(
        "--tau_init",
        type=float,
        default=1.0,
        required=False,
        help="Initial temperature",
    )
    parser.add_argument(
        "--tau_end", type=float, default=2.0, required=False, help="Final temperature"
    )
    ###########################################################################################################################
    parser.add_argument(
        "--beam",
        type=int,
        default=None,
        required=False,
        help="beam size for generation. If None, beam size will be loaded from the model configuration file (the parameter name is num_beams). If the model configuration file does not have this parameter, beam size will be set as 1",
    )
    parser.add_argument(
        "--eos_token_id",
        type=int,
        default=None,
        required=False,
        help="id fo the end-of-sequence token",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        required=False,
        help="The value used to module the next token probabilities.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        required=False,
        help="The number of highest probability vocabulary tokens to "
        "keep for top-k-filtering.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        required=False,
        help="If set to float < 1, only the most probable tokens with "
        "probabilities that add up to `top_p` or higher are kept for generation.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        required=False,
        help="The parameter for repetition penalty. 1.0 means no penalty.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether or not to use sampling ; use greedy decoding otherwise.",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=None,
        required=False,
        help="The number of independently computed returned sequences for each element in the batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        required=False,
        help="Specify a random seed for initialization",
    )
    parser.add_argument(
        "--padding_side",
        type=str,
        default=None,
        required=False,
        help="Specify which side the tokenizer should pad. Options are 'left' or 'right'.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="Try to load regular <model>Tokenizer instead of <model>TokenizerFast (default)",
    )
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    Path(args.save_path).parent.mkdir(exist_ok=True)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    print("=============================")
    print("Starting Summarization Task!!")
    print("=============================")

    generate_summary(
        args.dataset_path,
        args.score_path,
        args.save_path,
        args.attentions_path,
        args.model_name,
        args.model_path,
        batch_size=args.bs,
        device=args.device,
        dtype=args.dtype,
        task=args.task,
        decoder_start_token_id=args.decoder_start_token_id,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        skip_special_tokens=not args.include_special_tokens,
        clean_up_tokenization_spaces=args.clean_up_tokenization_spaces,
        preprocess_workers=args.preprocess_workers,
        postprocess_workers=args.postprocess_workers,
        return_tensors=args.return_tensors,
        truncation=not args.no_truncation,
        save_attentions=args.save_attentions,
        save_prompt_attentions=args.save_prompt_attentions,
        padding=args.padding,
        max_tokenizer_length=args.max_tokenizer_length,
        max_gen_length=args.max_gen_length,
        max_new_tokens=args.max_new_tokens,
        min_gen_length=args.min_gen_length,
        use_causal_lm=args.causal_lm,
        early_stopping=args.early_stopping,
        output_summaries_only=args.output_summaries_only,
        output_sequence_scores=args.output_sequence_scores,
        model_parallelize=args.model_parallelize,
        keyformer=args.keyformer,
        kv_cache=args.kv_cache,
        recent=args.recent,
        tau_init=args.tau_init,
        tau_end=args.tau_end,
        num_beams=args.beam,
        eos_token_id=args.eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        num_return_sequences=args.num_return_sequences,
        padding_side=args.padding_side,
        use_slow_tokenizer=args.use_slow_tokenizer,
        n_obs=args.n_obs,
    )


if __name__ == "__main__":
    main()
