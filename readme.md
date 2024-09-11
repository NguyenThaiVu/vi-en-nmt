
# English-Vietnamese Neural Machine Translation (NMT) using Transformer

This project implements an English-to-Vietnamese translation model using the Transformer architecture, as described in the seminal paper **"Attention is All You Need"**. The model is trained on the **PhoMT** dataset, containing 3 million sentence pairs, and achieves a BLEU score of **0.26**.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture and Training](#model-architecture-and-training)
- [Evaluation](#evaluation)
- [Future Improvements](#future-improvements)
- [References](#references)

## Introduction

This project aims to build an English-to-Vietnamese translation model using the Transformer architecture, which leverages the self-attention mechanism to capture dependencies in input sequences without relying on recurrent networks.

The model is trained on a large-scale bilingual corpus and fine-tuned to maximize translation accuracy, achieving a BLEU score of 0.26.

## Dataset

We use the **PhoMT dataset** for training and evaluation, which contains **3 million sentence pairs** for English-Vietnamese translation. 
The dataset was preprocessed and tokenized for both source (English) and target (Vietnamese) languages using **Byte-Pair Encoding (BPE)** to handle out-of-vocabulary words.

- **Number of sentence pairs**: 3 million
- **Languages**: English (source) and Vietnamese (target)
- **Tokenization**: Byte-Pair Encoding (BPE)

## Model Architecture and Training

This project implements the **Transformer architecture**, as described in the paper [*Attention is All You Need*](https://arxiv.org/abs/1706.03762). 

**The key configuration are:**

- Number of Encoder Layers: 4
- Number of Decoder Layers: 4
- Model Dimension: 128
- Number of Attention Heads: 8
- Feedforward Layer Dimension: 512
- Dropout: 0.1

### Training

The model was trained on the PhoMT dataset with 3 million sentence pairs. The following techniques were used for efficient training:

- Loss function: Cross-entropy
- Optimizer: Adam 
- Batch size: 128
- Number of epochs: 5
- GPU usage: NVIDIA RTX 4090


## Evaluation

The model's performance is evaluated using the **BLEU (Bilingual Evaluation Understudy)** score, a widely used metric for machine translation.

- **BLEU Score**: 0.26


### Translation Example

You can also use the trained model for translating sentences by running the notebook: `running_translation.ipynb`


### Sample Translations

Here are a few examples of translations produced by the model:

- **Input**: "The weather is nice today."
  - **Output**: "thời tiết hôm nay đẹp lắm."
  
- **Input**: "I am learning machine translation."
  - **Output**: "tôi đang học bản dịch máy."


## Future Improvements

- **Data Augmentation**: Explore synthetic data generation to increase the size of the training dataset.
- **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and number of layers.
- **Transfer Learning**: Use pretrained models for fine-tuning on the English-Vietnamese task.

## References

1. **Vaswani et al.**, *Attention is All You Need*, [arXiv](https://arxiv.org/abs/1706.03762), 2017.
2. **PhoMT Dataset**: [Link to the dataset](https://github.com/VinAIResearch/PhoMT)
3. **BLEU Metric**: [BLEU](https://en.wikipedia.org/wiki/BLEU)
