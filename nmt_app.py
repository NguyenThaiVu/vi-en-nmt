import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:   print(e)

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import regex as re
import string
import nltk
from sklearn.model_selection import train_test_split

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset
import tensorflow_text as tf_text

from utils.read_file_utils import *
from utils.model_utils import *
from utils.visualize_util import *

import streamlit as st



# ========================================================
PATH_MODEL_TRANSLATOR = "translator"
PATH_TOKENIZER = r"data/tokeninzer_en_vi_converter"
# ========================================================


# Load the model (You can replace this with your custom model)
@st.cache_resource
def load_model(path_tokenizer, path_model_translator):
    tokenizers = tf.saved_model.load(path_tokenizer)
    translator = tf.saved_model.load(path_model_translator)
    return tokenizers, translator

tokenizers, translator = load_model(PATH_TOKENIZER, PATH_MODEL_TRANSLATOR)


# Streamlit UI
st.title("Neural Machine Translation (NMT) Demo")

# Text input for translation
input_text = st.text_area("Enter text in English", value="", height=150)

if st.button("Translate"):
    if input_text:
        with st.spinner("Translating..."):
            # Translation
            input_text = input_text.lower()

            translated_text, translated_tokens, attention_weights = translator(tf.constant(input_text))
            translated_text = translated_text.numpy().decode('utf-8')

            st.success("Translation completed!")
            st.write(f"**Translated Text (Vietnamese):**")
            st.write(translated_text)
    else:
        st.error("Please enter text to translate.")

# Footer with some helpful information
st.write("---")
