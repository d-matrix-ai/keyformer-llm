# ## Obtain inference data
task=$1
shots=$2
model_name=$3
model_path=$4
model_type=$5
dtype=$6
keyformer=$7
kv_cache=$8
recent=$9

echo " "
echo "--------------------------------------------"
echo "Task ${task} data with ${shots} shots generation"
echo "--------------------------------------------"

python -u generate_task_data.py --output-file ./task_data/${task}-${shots}.jsonl --task-name ${task} --num-fewshot ${shots}

## Inference, and generate output json file
echo " "
echo "--------------------------------------------"
echo "LM Eval Harness for model ${model_name}"
echo "--------------------------------------------"

if [ $keyformer -eq 0 ]
then
    echo "====> Full Attention!!"
    python -u run_lm_eval_harness.py --input-path ./task_data/${task}-${shots}.jsonl --output-path ./model_eval_data/${task}-${shots}-${model_type}-${keyformer}-${kv_cache}-${recent}.jsonl --model-name ${model_name} --model-path ${model_path} --dtype ${dtype} --kv_cache ${kv_cache} --recent ${recent}
else
    echo "====> Keyformer enabled with ${kv_cache}% KV Cache with ${recent}% recent tokens!!"
    python -u run_lm_eval_harness.py --input-path ./task_data/${task}-${shots}.jsonl --output-path ./model_eval_data/${task}-${shots}-${model_type}-${keyformer}-${kv_cache}-${recent}.jsonl --model-name ${model_name} --model-path ${model_path} --dtype ${dtype} --keyformer --kv_cache ${kv_cache} --recent ${recent}
fi

## Evaluate results
echo " "
echo "--------------------------------------------"
echo "Results Generation"
echo "--------------------------------------------"
python -u evaluate_task_result.py --result-file ./model_eval_data/${task}-${shots}-${model_type}-${keyformer}-${kv_cache}-${recent}.jsonl --output-file ./output/${task}-${shots}-${model_type}-${keyformer}-${kv_cache}-${recent}.jsonl --task-name ${task} --num-fewshot ${shots} --model-name ${model_name}