import os 
import streamlit as st
import requests
from streamlit_util import *

ADDRESS = "0.0.0.0"
# ADDRESS = "172.17.0.1"
PORT = 5000
URL = f"http://{ADDRESS}:{PORT}"
# URL = f"http://backend:5000"




# Streamlit UI
st.title("Neural Machine Translation (NMT)")

# Text input for translation
input_text = st.text_area("Enter text in English", value="", height=150)

if st.button("Translate"):
    if input_text:
        with st.spinner("Translating..."):
            try:
                response = requests.post(f"{URL}/predict", json={"input_text": str(input_text)})

                if response.status_code == 200:
                    prediction = response.json()
                    translated_text = prediction["translated_text"]
                    translated_tokens = prediction["translated_tokens"]
                    average_attention_weights = prediction["average_attention_weights"]
                    list_predicted_prob = prediction["list_predicted_prob"]
                    in_tokens = prediction["in_tokens"]

                    st.success("Translation completed!")
                    st.write(f"**Translated Text (Vietnamese):**")
                    st.write(translated_text)

                    # Display the attention matrix
                    st.header("Attention Matrix Visualization")
                    fig = plot_attention_matrix(in_tokens, translated_tokens, average_attention_weights)
                    st.pyplot(fig)
                    
                    # Display the predicted probabilities
                    st.header("Translation Confidence Estimation")
                                        
                    threshold_entropy = np.log(len(list_predicted_prob[0])) / 4
                    output_translation_confidence = ""
                    st.write(f"[INFO] Threshold entropy: {threshold_entropy:.4f}")
                    st.write(f"[INFO] Green color indicates high confidence, while red color indicates low confidence.")

                    for (translated_token, predicted_prod) in zip(translated_tokens, list_predicted_prob):
                        if type(predicted_prod) != np.ndarray:
                            try:
                                predicted_prod = predicted_prod.numpy()
                            except:
                                predicted_prod = np.array(predicted_prod)
                        predicted_prod = predicted_prod.flatten()
                        predicted_prod = softmax(predicted_prod)

                        entropy = calculate_entropy(predicted_prod)

                        if entropy < threshold_entropy: # High confidence
                            color = "green"
                            output_translation_confidence += f"<span style='color: {color}; font-weight: bold;'>{translated_token}</span> &nbsp; "
                        else: # Low confidence
                            color = "red"
                            output_translation_confidence += f"<span style='color: {color}; font-weight: bold;'>{translated_token}</span> &nbsp; "
                    st.markdown(output_translation_confidence, unsafe_allow_html=True)
                            
            except requests.exceptions.RequestException as e:
                st.error("Error connecting to the server. Please try again later.")

    else:
        st.error("Please enter text to translate.")


# Footer with some helpful information
st.write("---")
