# 💬 Conversation Task

## Download dataset
```bash
cd dataset_download
python download_soda.py
```

## Getting started with Conversation Task

To get started with conversation task, follow use below inputs

```
python summarize.py --model_name <name of model used for conversation> \
                    --dataset_path <path to data.json> \
                    --save_path <path to out_model_dialogue> \
                    --score_path <path to out_model_dialogue.score> \
                    --model_path <path to model in case of local copy of model> \
                    --attentions_path <path for storing attention weights>
                    --device cuda \ # Device
                    --task conversation \ # Task
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
