# Keyformer: KV Cache reduction through key tokens selection for Efficient Generative Inference

**Keyformer** proposes KV Cache reduction through key tokens identification and without the need
for fine-tuning.

This repository contains the source code implementation of **Keyformer**.

This source code is available under the [MIT License](LICENSE.txt).

## ⚙️ Environment Setup

#### conda Environment
You can create a conda environment using the `conda-env.yml` file.
```bash
conda env create --file=conda-env.yml
conda activate keyformer-env
```


## 💻 System Requirements

Right now, **keyformer** has been tested on a 1xA100 node for `mosaicml/mpt-7b` and
`EleutherAI/gpt-j-6B` with the models in `FP32` and evaluation done using `FP16` data formats.

We welcome contributions to port and evaluate **keyformer** with different
data formats, targeting lower model memory requirements and overall KVCache size


## 🏁 Model Download

#### LLM Model Checkpoint
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

For example, to move the parameters for `mosaicml/mpt-7b` to `models/mpt-7b-keyformer` do
the following - 

```bash
cd
mv models/model_download/model/* models/mpt-7 b-keyformer/
```

## Using Keyformer for KV Cache reduction

The current approach walks through using **keyformer** with a pre-determined set of models
and for the tasks that have been covered in the paper.

### 🔍 Table of Contents
- [🎫 Blog](blog/README.md)
- [🧮 Models](models/README.md)
- [📖 Summarization](summarization/README.md)
- [💬 Conversation](conversation/README.md)
- [📊 LM Eval Harness](lm_eval_harness/README.md)
- [📝 Citation](#citation)
## 


### Todo

[ ] Instructions to integrate keyformer with any model from huggingface

[ ] Using keyformer with quantized models


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