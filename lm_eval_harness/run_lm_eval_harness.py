import argparse
import json, tqdm
import torch
import copy
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-type", type=str, default="opt")
    parser.add_argument("--cache-dir", type=str, default="./")
    parser.add_argument(
        "--dtype",
        default="float32",
        help="data type of the model, choose from float16, bfloat16 and float32",
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
        "--token_discard",
        action="store_true",
        help="When enable token discarded below threshold",
    )
    parser.add_argument(
        "--sparse_threshold",
        type=float,
        default=0.0,
        required=False,
        help="Threshold for sparsification",
    )
    parser.add_argument(
        "--prompt_sparse_threshold",
        type=float,
        default=0.0,
        required=False,
        help="Threshold for prompt sparsification",
    )
    parser.add_argument(
        "--sparse_itr",
        type=int,
        default=1,
        required=False,
        help="Tokene Generation Iteration for updating token discard mask",
    )

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model_name
    model_path = args.model_path
    dtype = args.dtype
    keyformer = args.keyformer
    kv_cache = args.kv_cache
    recent = args.recent
    tau_init = args.tau_init
    tau_end = args.tau_end

    if dtype == "bfloat16":
        amp_enabled = True
        amp_dtype = torch.bfloat16
    elif dtype == "float16":
        amp_enabled = True
        amp_dtype = torch.float16
    else:
        amp_enabled = False
        amp_dtype = torch.float32

    if model_name.split("/")[0] == "mosaicml":
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.attn_config["attn_impl"] = "torch"
        config.init_device = "cuda:0"
        config.use_cache = True
        config.torch_dtype = dtype
        config.keyformer_config["keyformer"] = keyformer
        config.keyformer_config["kv_cache"] = kv_cache
        config.keyformer_config["recent"] = recent
        config.keyformer_config["tau_init"] = tau_init
        config.keyformer_config["tau_delta"] = tau_end - tau_init
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=amp_dtype, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    elif model_name == "EleutherAI/gpt-j-6B":
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.keyformer_config["keyformer"] = keyformer
        config.keyformer_config["kv_cache"] = kv_cache
        config.keyformer_config["recent"] = recent
        config.keyformer_config["tau_init"] = tau_init
        config.keyformer_config["tau_delta"] = tau_end - tau_init
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=amp_dtype, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.keyformer_config["keyformer"] = keyformer
        config.keyformer_config["kv_cache"] = kv_cache
        config.keyformer_config["recent"] = recent
        config.keyformer_config["tau_init"] = tau_init
        config.keyformer_config["tau_delta"] = tau_end - tau_init
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=amp_dtype, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.eval().cuda()

    requests = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip() != "":
                requests.append(json.loads(line))

    results = []
    with torch.no_grad():
        for request in tqdm.tqdm(requests):
            result = {"request": request, "result": {}}
            prompt = request["prompt"]
            input_ids = tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(model.device)

            logits = model(input_ids).logits.log_softmax(dim=-1)
            values, indices = logits.squeeze(0).topk(dim=-1, k=1)
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

            gold_indices = input_ids[:, 1:]  # skip first
            logprobs = [None] + torch.gather(
                logits, -1, gold_indices.unsqueeze(-1)
            ).squeeze(-1).squeeze(0).detach().cpu().tolist()
            top_logprobs = [None] + [
                {tokenizer.convert_ids_to_tokens(i.item()): v.item()}
                for v, i in zip(values.squeeze(-1), indices.squeeze(-1))
            ]

            result["result"] = {
                "choices": [
                    {
                        "text": prompt,
                        "logprobs": {
                            "tokens": tokens,
                            "token_logprobs": logprobs,
                            "top_logprobs": top_logprobs,
                            "text_offset": [],
                        },
                        "finish_reason": "length",
                    }
                ],
                "request_time": {"batch_time": 0, "batch_size": 1},
            }

            results.append(result)

    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
