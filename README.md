
# Encoder/Decoder Tutorial

Learn the fundamentals of the Encoder/Decoder Transformer architecture (the building block of LLMs like ChatGPT) with a working pytorch example that translates Dutch (NL) to English (EN). This example closely follows the transformer architecture of the [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper.
 
- Get started with the interactive Google Colab notebook [**here**](https://colab.research.google.com/drive/1I-IFQvNW-PLEcDT3WeIX0OSgbExYcE6k#scrollTo=6fb3ba33-3de6-468c-a070-04886cb4329a).

![architecture](architecture.svg)
*^ model architecture [here](https://docs.google.com/presentation/d/1sRWV0hxIgL8ZNyrqV_jz7vX5l815RJI5l_KlXoVJHAQ/edit#slide=id.g29d8fe39d9a_0_0)*

## Training Data
- Credit goes to [Tatoeba.org](https://tatoeba.org/en/downloads) for the Dutch <-> English sentence pairs.

## Model Architecture
- C = 512 *(aka `d_model`)*
- T = 30  *(max context length; informed by sentence length)*
- number of layers = 6
- number of heads = 8
- head size = 64

## Training Params
- Trained via Google Colab (V100 Machine)
- Epochs: 20 
- batch size: 8
- One cycle learning schedule (init=1e-7; max=1e-5, final=1e-6)
- Warmup steps: 5000
- dropout: 10%

## Limitations
The primary purpose of this repo is educational and as such has the following limitations:
- The model itself it trained on a very small dataset (~140K sentence pairs) whereas modern LLMs are trained on +trillion tokens.  The performance of the model reflects this.
- Training data sentence pairs are fairly short in length (mean of ~30 characters each with a long right-skewed tail) which likely limits the model's ability to translate long sentences.
- Training epochs were limited to 20 but additional training could be performed (see `model_object` to continue training on your own).
    - `main.py` can be used for remote host training (via services such as Lambda Labs, Paperspace) to avoid common timeouts errors with Google Colab.  [Here](https://docs.google.com/document/d/1CVrw9Hn5Qwk4iHWWcI8Gh2pwAa1OY8sH6YVAanX4agc/edit) are the steps I followed to train the model on a Windows 10 remote host (via Paperspace).

## Python Dependencies
- Conda used to build environment (see `environment.yml` for dependencies)
- note: to enable GPU usage, pytorch+cu118 was installed using `pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118` and not through the typical `conda install` process.