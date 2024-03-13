# 🏁 Models

## LLM Model Checkpoint
You can download the model parameters using `download_model.py` file by providing the model_name.
```bash
cd
cd models/model_download
python3 download_model.py --model_name mosaicml/mpt-7b
```

Available options for `model_name` are restricted to `['cerebras/Cerebras-GPT-6.7B', 
        'mosaicml/mpt-7b', 'EleutherAI/gpt-j-6B']`

This step will download the model parameters by creating a `model` folder inside `model_download` directory.
Downloaded model parameters are required to be moved to the same directory where model files are available.

## Preparing mosaicml/mpt-7b for Keyformer

```
cp -r models/mpt-7b models/mpt-7b-keyformer
```

Move the parameters from the model download folder for `mosaicml/mpt-7b` to
`models/mpt-7b-keyformer` as follows - 

```bash
cd
mv models/model_download/model/* models/mpt-7b-keyformer/.
```

## LLM Model Cards

We have provided model cards for below models.
- [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6b)
- [MPT](https://huggingface.co/mosaicml/mpt-7b)
- [Cerebras-GPT](https://huggingface.co/cerebras/Cerebras-GPT-6.7B)
