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


def visualize_sentence_mapping(in_tokens, translated_tokens, heat_map, threshold=0.4, figsize=(8, 4)):
    """
    Visualizes the mapping from English to Vietnamese sentences based on a heat map.

    Parameters:
    heat_map (numpy.ndarray): 2D array representing relationship strengths between sentences.
    in_tokens (list): List of English sentences.
    translated_tokens (list): List of Vietnamese sentences.
    """
    translated_tokens = translated_tokens[1:]

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create a table of sentences
    table_data = []
    for i in range(len(translated_tokens)):
        for j in range(len(in_tokens)):
            if heat_map[i, j] > threshold:
                table_data.append((translated_tokens[i].decode('utf-8'), in_tokens[j].decode('utf-8')))

    # Create table and add it to the plot
    table = ax.table(cellText=table_data, colLabels=["English", "Vietnamese"], cellLoc='center', loc='center')

    table.set_fontsize(12)
    ax.set_title("Mapping from English to Vietnamese Sentences")
    plt.show()



def visualize_sentence_mapping(in_tokens, translated_tokens, heat_map, threshold=0.4, figsize=(8, 4)):
    """
    Visualizes the mapping from English to Vietnamese sentences based on a heat map.

    Parameters:
    heat_map (numpy.ndarray): 2D array representing relationship strengths between sentences.
    in_tokens (list): List of English sentences.
    translated_tokens (list): List of Vietnamese sentences.
    """
    translated_tokens = translated_tokens[1:]

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create a table of sentences
    table_data = []
    for i in range(len(translated_tokens)):
        for j in range(len(in_tokens)):
            if heat_map[i, j] > threshold:
                table_data.append((translated_tokens[i].decode('utf-8'), in_tokens[j].decode('utf-8')))

    # Create table and add it to the plot
    table = ax.table(cellText=table_data, colLabels=["--- ENGLISH ---", "--- VIETNAMESE ---"], cellLoc='center', loc='center')

    table.set_fontsize(12)
    plt.axis('off')
    plt.show();