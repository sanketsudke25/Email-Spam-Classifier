import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("📧 Email Spam Detector")

text = st.text_area("Enter Email Message")

if st.button("Predict"):
    input_data = vectorizer.transform([text])
    prediction = model.predict(input_data)

    if prediction[0] == "spam":
        st.error("This is Spam Email")
    else:
        st.success("This is Not Spam")
