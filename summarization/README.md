# 📖 Summarization Task

## Download dataset

You can download the respective summarization datasets using below script.
```bash
cd dataset_download
python download_cnndm.py
```

We have provided data download script for below summarization datasets.
- [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail)
- [XSUM](https://huggingface.co/datasets/EdinburghNLP/xsum)
- [GovReports](https://huggingface.co/datasets/ccdv/govreport-summarization?row=0)

## Quick start with the Summarization Task

You can directly run the summarization task using the following - 

```
bash run_summarization_task.sh
```

By default, this runs `mosaicml/mpt-7b`. It assumes that the model copy with 
`keyformer` is stored in `models/mpt-7b-keyformer` and the dataset is stored in
`data/cnn_eval.json`. For custom execution, read below.


## Summarization Task

To get started with summarization task, setup the model parameters and use below script

```
python summarize.py --model_name <name of model used for summarization> \
                    --dataset_path <path to cnn_eval.json> \
                    --save_path <path to output.summary file the contains summaries> \
                    --score_path <path to output.score file with ROUGE scores> \
                    --model_path <path to model in case of local copy of model> \
                    --attentions_path <path for storing attention weights>
                    --device cuda \ # Device
                    --task summarization \ # Task - Currently only summarization supported
                    --bs 1 \ # Batch Size
                    --dtype float16 \ # Data type of model parameters
                    --causal_lm \ # Causal language model for summarization
                    --early_stopping \ # Enable early stopping while generation
                    --output_summaries_only \ # Output only summary - No prompt
                    --output_sequence_scores \ # Output sequence scores and enable for output attentions
                    --save_attentions \ # Save attention weights - Token Generation only
                    --save_prompt_attentions \ # If enabled only prompt attention weights are stored
                    --padding_side left \ # Padding side
                    --beam 4 \ # Beam Size
                    --model_parallelize \ # Parallelize Model across all  available GPUs
                    --keyformer \ # Enable Keyformer
                    --kv_cache 60 \ # KV cache percentage of prompt length
                    --recent 30 \ # Recent window percentage
                    --tau_init 1 \ # Initial temperature parameter for Gumbel
                    --tau_end 2 \ # End temperature parameter for Gumbel
                    --no_repeat_ngram_size 0 \ 
                    --repetition_penalty 1 \
                    --max_tokenizer_length 1920 \ # Maximum prompt size
                    --max_new_tokens 128 \ # Maximum newly generated tokens
                    --min_gen_length 30 \ # Minimum newly generated tokens
                    --num_return_sequences 1 \ # Number os return summaries per input
                    --seed 12345 \ # Random seed value for radom samples
                    --n_obs 1000 \ # Number of input samples
```

Note: For data type of FP16, do not use model.half() instead utilize dtype in model creation [Link](https://stackoverflow.com/questions/69994731/what-is-the-difference-between-cuda-amp-and-model-half)
