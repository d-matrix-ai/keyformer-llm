# Keyformer: KV Cache reduction through key tokens selection for Efficient Generative Inference

**Keyformer** proposes KV Cache reduction through key tokens identification and without the need
for fine-tuning.

This repository contains the source code implementation of **Keyformer**.

This source code is available under the [Apache 2.0 License](LICENSE).

For the readers, to get a quick overview of how **keyformer** reduces the KV Cache size and speeds up LLM inference, we invite you to browse through our blog - 

- [🎫 Blog](blog/README.md) - Quick summary of Keyformer 

But if you'd rather see **keyformer** in action first, please continue reading.
<br></br>
## ⚙️ Environment Setup

#### conda Environment
You can create a conda environment using the `conda-env.yml` file.
```bash
conda env create --file=conda-env.yml
conda activate keyformer-env
```
<br></br>
## 💻 System Requirements

Right now, **keyformer** has been tested on a 1xA100 node for `mosaicml/mpt-7b` and
`EleutherAI/gpt-j-6B` with the models in `FP32` and evaluation done using `FP16` data formats.

We welcome contributions to port and evaluate **keyformer** with different
data formats, targeting lower model memory requirements and overall KVCache size
<br></br>
## 🏁 Model Download and Integration with Keyformer

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
mv models/model_download/model/* models/mpt-7b-keyformer/.
```

### Integrating Keyformer
Copy the **Keyformer** source files from `models/mpt-keyformer-lib` to `models/mpt-7b-keyformer`

```bash
cp -r models/mpt-keyformer-lib models/mpt-7b-keyformer
```

This sets up `models/mpt-7b-keyformer` to be used with the various tasks described below.

Similarly, find the keyformer-lib files for other supported models in their respective directories in `models/`.
<br></br>
## Using Keyformer for KV Cache reduction

After setting up a model with `keyformer`, let's run it with a downstream task of interest. In the case of `keyformer`, we provide examples on how to run downstream tasks of **summarization** and **conversation**. Further, for the purposes of evaluation, we provide a step-by-step tutorial on how to use **lm_eval_harness**.

Depending on your task of interest, please refer to the following links for more details

- [📖 Summarization](summarization/README.md) - Running the Summarization task with Keyformer
- [💬 Conversation](conversation/README.md) - Running the Conversation task with Keyformer
- [📊 LM Eval Harness](lm_eval_harness/README.md) - Using LM Eval harness for evaluation with Keyformer
<br></br>
## TODO

[ ] Instructions to integrate keyformer with any model from huggingface

[ ] Using keyformer with quantized models
<br></br>

## Thank You

Keyformer uses open source components available on [huggingface](https://huggingface.co/), [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b), [cerebras/Cerebras-GPT-6.7B](https://huggingface.co/cerebras/Cerebras-GPT-6.7B), and
[EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6b).
<br></br>
<a name="citation"></a>
## 📝 Citation
```
@article{2023keyformer,
  title={Keyformer: KV Cache reduction through key tokens selection for Efficient Generative Inference},
  author={Adnan, Muhammad and Arunkumar, Akhil and Jain, Gaurav and Nair, Prashant and Soloveychik, Ilya and Kamath, Purushotham},
  journal={Proceedings of Machine Learning and Systems},
  volume={7},
  year={2024}
}
```
