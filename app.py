import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Downloading required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#Loadding saved model and vectorizer
model = joblib.load("best_svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

#Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

#Streamlit interface
st.title("Fake News Detector")
st.write("Paste a news article to check if it's likely fake.")

# Users input
user_input = st.text_area("️ Enter article text below:")

# Predicting button
if st.button("Check"):
    if user_input.strip():
        # Preprocessing and transforming
        cleaned_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]

        # Showing result
        if prediction == 0:
            st.error(" This article appears to be FAKE.")
        else:
            st.success(" This article does NOT appear to be fake.")
    else:
        st.warning("Please enter some text.")
