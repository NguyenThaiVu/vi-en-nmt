# backend.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:   print(e)

import tensorflow_text as tf_text

import sys 
sys.path.append("../")

from flask import Flask, request, jsonify


PATH_MODEL_TRANSLATOR = "translator"
PATH_TOKENIZER = r"tokeninzer_en_vi_converter"

app = Flask(__name__)

def load_model(path_tokenizer, path_model_translator):
    """
    This function loads the trained model and tokenizers from the saved model files.
    """
    tokenizers = tf.saved_model.load(path_tokenizer)
    translator = tf.saved_model.load(path_model_translator)
    return tokenizers, translator

tokenizers, translator = load_model(PATH_TOKENIZER, PATH_MODEL_TRANSLATOR)


def process_translated_tokens(translated_tokens):
    """
    This function adjust the translated tokens and decode them to utf-8.
    """
    translated_tokens = translated_tokens[1:]  # Adjust translated tokens
    translated_tokens = [label.decode('utf-8') for label in translated_tokens.numpy()]
    return translated_tokens
    

def calculate_average_attention_weight(attention_weights):
    """
    This function calculates the average attention weights.
    """
    # Get average attention map
    average_attention_weights = attention_weights[0].numpy()
    average_attention_weights = average_attention_weights.mean(axis=0)
    return average_attention_weights

def extract_input_to_intokens(input_text):
    """
    Function extract input sentence into tokens.
    """
    in_tokens = tf.convert_to_tensor([input_text])
    in_tokens = tokenizers.en.tokenize(in_tokens).to_tensor()
    in_tokens = tokenizers.en.lookup(in_tokens)[0]

    in_tokens = [label.decode('utf-8') for label in in_tokens.numpy()]
    return in_tokens


# Define an endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    input_text = request.json.get("input_text")

    if not input_text:
        return jsonify({"error": "Missing data"})
    
    # Translation
    input_text = input_text.lower()
    translated_text, translated_tokens, attention_weights, list_predicted_prob = translator(tf.constant(input_text))
    translated_text = translated_text.numpy().decode('utf-8')
    average_attention_weights = calculate_average_attention_weight(attention_weights).tolist()
    translated_tokens = process_translated_tokens(translated_tokens)
    in_tokens = extract_input_to_intokens(input_text)
    list_predicted_prob = list_predicted_prob.numpy().tolist()

    result = {"translated_text": translated_text, "in_tokens": in_tokens,\
              "translated_tokens": translated_tokens, "average_attention_weights": average_attention_weights,\
              "list_predicted_prob": list_predicted_prob}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
