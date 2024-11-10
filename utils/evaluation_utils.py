import os 
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu_score(list_true_sentence, pred_sentence):
    """
    Calculate BLEU score for a predicted sentence.
    * Params:
        - list_true_sentence: list of true sentences
        - pred_sentence: predicted sentence
    * Return:
        - BLEU score
    """

    if isinstance(pred_sentence, str):
        pred_sentence = pred_sentence.split()

    for (idx, true_sentence) in enumerate(list_true_sentence):
        if isinstance(true_sentence, str):
            true_sentence = true_sentence.split()
            list_true_sentence[idx] = true_sentence

    smoothie = SmoothingFunction().method1  # Smoothing function for cases with no n-gram overlaps
    score = sentence_bleu(list_true_sentence, pred_sentence, smoothing_function=smoothie)

    return score


def calculate_edit_distance(reference, candidate):
    """
    This function calculate the edit distance between two sequences, including insertion, deletion, and substitution.
    
    * Parameters:
        - reference: list of str, the reference sequence
        - candidate: list of str, the candidate sequence
        
    * Returns:
        - int, the edit distance between the reference and candidate
    """
    
    # Initialize the matrix
    ref_len = len(reference) + 1
    cand_len = len(candidate) + 1
    matrix = [[0] * cand_len for _ in range(ref_len)]
    
    # Fill the first row and column
    for i in range(ref_len):
        matrix[i][0] = i
    for j in range(cand_len):
        matrix[0][j] = j
    
    # Compute the edit distance
    for i in range(1, ref_len):
        for j in range(1, cand_len):
            if reference[i - 1] == candidate[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,       # Deletion
                               matrix[i][j - 1] + 1,       # Insertion
                               matrix[i - 1][j - 1] + cost)  # Substitution
    
    return matrix[-1][-1]

def calculate_ter(reference, candidate):
    """
    This function calculate the Translation Error Rate (TER) between the reference and candidate translations.
    
    * Parameters:
        - reference: str, the reference translation
        - candidate: str, the candidate translation
    
    * Returns:
        - float, the TER score between the reference and candidate translations
    """
    
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    edit_distance = calculate_edit_distance(ref_tokens, cand_tokens)
    
    # Normalizing edit distance with the length of true sentence
    ter_score = edit_distance / len(ref_tokens)
    return ter_score