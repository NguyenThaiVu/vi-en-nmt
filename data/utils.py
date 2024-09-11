import os 
import regex as re 

def read_text_file(filename):
    """
    Function to read raw text file line-by-line and return list of sentence.
    """
    with open(filename, encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    
    return lines


def get_len_sentence(sentence):
    """
    This function return the length of sentence.
    """
    return len(sentence.split())


def get_vocab_from_list_sentence(list_sentence):
    """
    This function take a list of sentence (corpus) and return the list of vocab
    """

    list_vocab = set()

    for sentence in list_sentence:
        words = sentence.split()
        list_vocab.update(words)

    list_vocab = list(list_vocab)
    return list_vocab


def save_sentences_to_file(list_sentences, filename):
    """
    Function to save list of sentences to a text file
    """

    with open(filename, 'w') as file:
        for sentence in list_sentences:
            file.write(sentence + '\n')


def format_sentence(sentence):
    """
    This function format sentence with the following criteria:
    - Convert to lowercase
    - Add spaces around punctuation
    """
    sentence = sentence.lower()
    
    # Define a regex pattern to find punctuation characters
    # pattern = re.compile(f'([{re.escape(string.punctuation)}])')
    pattern = re.compile(f'([{re.escape(",.;?!")}])')
    
    # Add spaces around punctuation
    formatted_sentence = pattern.sub(r' \1 ', sentence)
    
    # Remove any extra spaces (in case there were multiple punctuation marks together)
    formatted_sentence = re.sub(r'\s+', ' ', formatted_sentence).strip()
    
    return formatted_sentence
