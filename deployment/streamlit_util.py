import os 
import matplotlib.pyplot as plt
import numpy as np


def plot_attention_matrix(in_tokens, translated_tokens, attention):

    fig, ax = plt.subplots()  # Create a new figure and axis
    cax = ax.matshow(attention)  # Plot attention matrix
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    # Set x-axis labels with rotation
    labels = [label for label in in_tokens]
    ax.set_xticklabels(labels, rotation=90)

    # Set y-axis labels
    labels = [label for label in translated_tokens]
    ax.set_yticklabels(labels)

    # Add color bar to the right of the plot
    plt.colorbar(cax)

    return fig


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