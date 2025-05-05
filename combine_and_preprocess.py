import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Downloading required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#Preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

print("Loading datasets")

#Loading fake news
fake = pd.read_csv("fake.csv")
fake = fake[["text"]].dropna()
fake["label"] = 0

#Loading original true news
true = pd.read_csv("true.csv")
true = true[["text"]].dropna()
true["label"] = 1

#Loading new true dataset 
new_true = pd.read_csv("DataSet_Misinfo_TRUE.csv")  
new_true = new_true[["text"]].dropna()
new_true["label"] = 1

#Combining everything
df = pd.concat([fake, true, new_true], ignore_index=True).sample(frac=1, random_state=42)

#Applying preprocessing
print("Preprocessing text")
df["processed_text"] = df["text"].apply(preprocess_text)

#Saving to CSV
df.to_csv("combined_preprocessed_data.csv", index=False)
print("Preprocessed data saved as 'combined_preprocessed_data.csv'")
