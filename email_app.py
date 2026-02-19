import streamlit as st
import pickle
import spacy

# Load saved files
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words

st.title("📧 Email Spam Classifier")

text = st.text_area("Enter Email Message")

if st.button("Predict"):
    words = []
    doc = nlp(text)
    
    for token in doc:
        if token.lemma_ not in stop_words:
            words.append(token.lemma_)
    
    new_text = " ".join(words)
    input_data = vectorizer.transform([new_text])
    prediction = model.predict(input_data)
    
    if prediction[0] == "spam":
        st.error("🚨 This is Spam Email")
    else:
        st.success("✅ This is Not Spam")
