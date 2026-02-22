import streamlit as st
import pickle

# ✅ Change Browser Tab Name
st.set_page_config(page_title="📧 Email Spam Detector")

# ✅ Background Gradient
st.markdown("""
    <style>
    .stApp {
          background: linear-gradient(135deg, #74b9ff, #6c5ce7);
    }
    </style>
""", unsafe_allow_html=True)

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