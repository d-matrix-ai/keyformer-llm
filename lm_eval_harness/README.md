# 📊 LM Eval Harness Tasks

Evaluation of **Keyformer** on [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework tasks.

## Generate Task data
```python
python -u generate_task_data.py \
    --output-file ./task_data/${task}-${shots}.jsonl \
    --task-name ${task} \
    --num-fewshot ${shots}
```

## Generate Output data with model
#### Full Attention
```python
python -u run_lm_eval_harness.py \
    --input-path ./task_data/${task}-${shots}.jsonl \
    --output-path ./model_eval_data/${task}-${shots}-${model_type}-${keyformer}-${kv_cache}-${recent}.jsonl \
    --model-name ${model_name} \
    --model-path ${model_path} \
    --dtype ${dtype} \
    --kv_cache ${kv_cache} \
    --recent ${recent}
```
#### Keyformer
```python
python -u run_lm_eval_harness.py \
    --input-path ./task_data/${task}-${shots}.jsonl \
    --output-path ./model_eval_data/${task}-${shots}-${model_type}-${keyformer}-${kv_cache}-${recent}.jsonl \
    --model-name ${model_name} \
    --model-path ${model_path} \
    --dtype ${dtype} \
    --keyformer \
    --kv_cache ${kv_cache} \
    --recent ${recent}
```

## Evaluate the performance
```python
python -u evaluate_task_result.py \
    --result-file ./model_eval_data/${task}-${shots}-${model_type}-${keyformer}-${kv_cache}-${recent}.jsonl \
    --output-file ./output/${task}-${shots}-${model_type}-${keyformer}-${kv_cache}-${recent}.jsonl \
    --task-name ${task} \
    --num-fewshot ${shots} \
    --model-name ${model_name}
```