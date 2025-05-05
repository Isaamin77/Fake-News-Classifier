import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Downloading necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#Loadding existing combined dataset
original_df = pd.read_csv("combined_news.csv")

#Loadding the new real news file
extra_real = pd.read_csv("extra_real.csv")

#Merging both datasets
df = pd.concat([original_df, extra_real], ignore_index=True)
print(f"Merged dataset size: {df.shape}")

#Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

#Droping rows with missing text
df = df.dropna(subset=["text"])

#Applying preprocessing to 'text' column
df["processed_text"] = df["text"].apply(preprocess_text)

#Saving preprocessed data
df.to_csv("preprocessed_data.csv", index=False)
print(" Preprocessing complete. Saved as 'preprocessed_data.csv'")
