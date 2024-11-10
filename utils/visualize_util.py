import os 
import numpy as np
import matplotlib.pyplot as plt

def plot_attention_head(in_tokens, translated_tokens, attention):
    translated_tokens = translated_tokens[1:]

    ax = plt.gca()
    cax = ax.matshow(attention)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    labels = [label.decode('utf-8') for label in in_tokens.numpy()]
    ax.set_xticklabels(labels, rotation=90)

    labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
    ax.set_yticklabels(labels)
    plt.colorbar(cax)


def plot_attention_weights(in_tokens, translated_tokens, attention_heads):
    """
    This function visualize the attention scores between in_tokens and translated_tokens

    * Parameter
    in_tokens -- array of tokens
    translated_tokens -- array of tokens
    attention_heads -- shape (# heads, len_translated_tokens, len_in_tokens)
    """

    fig = plt.figure(figsize=(16, 8))

    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(2, 4, h+1)

        plot_attention_head(in_tokens, translated_tokens, head)

        ax.set_xlabel(f'Head {h+1}')

    plt.tight_layout()
    plt.show()


def calculate_average_attention(attention_weights):
    """
    This function calculate the average attention weights across all heads.

    * Parameters:
        - attention_weights: tf.Tensor, shape=(num_heads, out_seq_len, input_seq_len)

    * Returns:
        - average_attention_weights: np.array, shape=(out_seq_len, input_seq_len)
    """
    average_attention_weights = attention_weights.numpy()
    average_attention_weights = average_attention_weights.mean(axis=0)
    return average_attention_weights



def calculate_entropy(prob_distribution, epsilon=1e-9):
    """
    Calculate the entropy of a probability distribution.
    
    Parameters:
    - prob_distribution (list or numpy array): A list or array containing probabilities for each possible word in the vocabulary.
    
    Returns:
    - entropy (float): The entropy of the distribution, indicating the uncertainty/confidence of the prediction.
    """
    if type(prob_distribution) != np.ndarray:
        prob_distribution = np.array(prob_distribution)
    entropy = -np.sum(prob_distribution * np.log(prob_distribution + epsilon))
    return entropy


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 


def visualize_confidence_estimation(translated_tokens, list_predicted_prob, threshold_entropy=0.5):
    """
    Visualize confidence estimation on the translated tokens.
    
    Parameters:
    - translated_tokens (list): A list of tokens in the translated sentence.
    - list_predicted_prob (list): A list of probability distributions for each token in the translated sentence.
    
    Returns:
    - None
    """
    # threshold_entropy = np.log(len(list_predicted_prob[0])) / 2 # Set threshold to be half of maximum entropy
    threshold_entropy = np.log(len(list_predicted_prob[0])) / 4
    print(f"   [INFO] Threshold entropy: {threshold_entropy:.4f}")

    for (translated_token, predicted_prod) in zip(translated_tokens, list_predicted_prob):
        if type(predicted_prod) != np.ndarray:
            predicted_prod = predicted_prod.numpy()
        predicted_prod = predicted_prod.flatten()
        predicted_prod = softmax(predicted_prod)

        entropy = calculate_entropy(predicted_prod)

        if entropy < threshold_entropy:
            print(f"Token: {translated_token.numpy().decode('utf-8')}, Entropy: {entropy:.4f}, High confidence")
        else:
            print(f"Token: {translated_token.numpy().decode('utf-8')}, Entropy: {entropy:.4f}, Low confidence")