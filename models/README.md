# 🏁 Models

## LLM Model Checkpoint
Note - **Keyformer** has been evaluated on the following models and we restrict this tutorial to the use of these -
`['cerebras/Cerebras-GPT-6.7B', 
        'mosaicml/mpt-7b', 'EleutherAI/gpt-j-6B']`

Clone the relevant model from huggingface into the `models` directory. For instance, to clone `mosaicml/mpt-7b` into `models/mpt-7b-keyformer`, do the following - 

```bash
cd models
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/mosaicml/mpt-7b mpt-7b-keyformer
cd ..
```

Then, download the model weights (`*.bin`) using `download_model.py` file and use the model_name as the argument.

```bash
cd models/model_download
python3 download_model.py --model_name mosaicml/mpt-7b
cd ../../
```

Please keep in mind that supported arguments for `model_name` are restricted to `['cerebras/Cerebras-GPT-6.7B', 
        'mosaicml/mpt-7b', 'EleutherAI/gpt-j-6B']`

This step will download the model parameters by creating a `model` folder inside `model_download` directory.
Downloaded model parameters are required to be moved to the same directory where model files are available.

To move the model parameters from `models/model_download/model/` to `models/mpt-7b-keyformer`

```bash
mv models/model_download/model/* models/mpt-7b-keyformer/
```

### Integrating Keyformer
Copy the **Keyformer** source files from `models/mpt-keyformer-lib` to `models/mpt-7b-keyformer`

```bash
cp -r models/mpt-keyformer-lib models/mpt-7b-keyformer
```

This sets up `models/mpt-7b-keyformer` to be used with the various tasks described below.

Similarly, find the keyformer-lib files for other supported models in their respective directories in `models/`.
<br></br>
## LLM Model Cards

We have provided model cards for below models.
- [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6b)
- [MPT](https://huggingface.co/mosaicml/mpt-7b)
- [Cerebras-GPT](https://huggingface.co/cerebras/Cerebras-GPT-6.7B)
