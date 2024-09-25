
# English-Vietnamese Neural Machine Translation (NMT) using Transformer

This project implements an English-to-Vietnamese translation model using the Transformer architecture, as described in the seminal paper **"Attention is All You Need"**. The model is trained on the **PhoMT** dataset, containing 3 million sentence pairs, and achieves a BLEU score of **0.26**.

In addition, we utilize post-training quantization to reduce 20 MB the model size , making it more suitable for deployment on limited devices.

## Table of Contents

- [Dataset](#dataset)
- [Model Architecture and Training](#model-architecture-and-training)
- [Evaluation](#evaluation)
- [Future Improvements](#future-improvements)
- [References](#references)

## Dataset

We use the **PhoMT dataset** for training and evaluation, which contains **3 million sentence pairs** for English-Vietnamese translation. 
The dataset was preprocessed and tokenized for both source (English) and target (Vietnamese) languages using **Byte-Pair Encoding (BPE)** to handle out-of-vocabulary words.

- **Number of sentence pairs**: 3 million
- **Languages**: English (source) and Vietnamese (target)
- **Tokenization**: Byte-Pair Encoding (BPE)

## Model Architecture and Training

This project implements the **Transformer architecture**, as described in the paper [*Attention is All You Need*](https://arxiv.org/abs/1706.03762). 

**The key architecture configuration:**

- Number of Encoder/Decoder Layers: 4
- Model Dimension: 128
- Number of Attention Heads: 8
- Feedforward Layer Dimension: 512

**The key training configuration:**

- Loss function: Cross-entropy
- Optimizer: Adam 
- Batch size: 128
- Number of epochs: 10
- GPU usage: NVIDIA RTX 4090


## Post-Training Quantization (PTQ)
This section focuses on post-training quantization to reduce the size of the trained model. We quantized the multi-head attention, cross-attention, and the dense layer.

Post-training quantization includes:
- Quantized data types: int8, uint8, float16.
- Quantization techniques: symmetric and asymmetric.

### Result of PTQ:
- Reduce model size from 110MB to 90MB. 


## Evaluation

The model's performance is evaluated using the **BLEU (Bilingual Evaluation Understudy)** score:

- **BLEU Score**: 0.26

### Sample Translations

Here are a few examples of translations produced by the model:

- **Input**: "The weather is nice today."
  - **Output**: "thời tiết hôm nay đẹp lắm."
  
- **Input**: "I am learning machine translation."
  - **Output**: "tôi đang học bản dịch máy."


You can also use the trained model for translating sentences by running the notebook: `running_translation.ipynb`


## Future Improvements

- **Data Augmentation**: Explore synthetic data generation to increase the size of the training dataset.
- **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and number of layers.
- **Transfer Learning**: Use pretrained models for fine-tuning on the English-Vietnamese task.

## References

1. **Vaswani et al.**, *Attention is All You Need*, [arXiv](https://arxiv.org/abs/1706.03762), 2017.
2. **PhoMT Dataset**: [Link to the dataset](https://github.com/VinAIResearch/PhoMT)
3. **BLEU Metric**: [BLEU](https://en.wikipedia.org/wiki/BLEU)
4. **Quantization**: [link](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html)
