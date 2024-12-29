import streamlit as st
import pickle

# Load the saved model and vectorizer
with open('spam_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.sav', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit UI
st.title("Spam Mail Detection")
st.write("Enter a message to classify it as spam or not spam.")

# Input message
message = st.text_area("Message")

if st.button("Classify"):
    if message.strip():
        # Preprocess and predict
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)[0]
        result = "Spam" if prediction == 1 else "Not Spam"
        st.write(f"Prediction: {result}")
    else:
        st.write("Please enter a valid message.")