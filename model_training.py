import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Loading preprocessed data")
df = pd.read_csv("combined_preprocessed_data.csv")
df = df.dropna(subset=['processed_text'])

#Sampling the dataset
print("Balancing dataset")
fake = df[df['label'] == 0]
real = df[df['label'] == 1]

fake_sample = fake.sample(n=10000, random_state=42)
real_sample = real.sample(n=20000, random_state=42)

df_sampled = pd.concat([fake_sample, real_sample]).sample(frac=1, random_state=42)

#Splitting the data
X = df_sampled['processed_text']
y = df_sampled['label']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#TF-IDF Vectorizer
print("Applying TF-IDF vectorizer")
tfidf = TfidfVectorizer(
    max_features=30000,
    stop_words='english',
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.9
)
X_train = tfidf.fit_transform(X_train_raw)
X_test = tfidf.transform(X_test_raw)

#Training SVM model
print("Training SVM model")
svm_model = SVC(kernel='linear', C=1, probability=True, random_state=42)
svm_model.fit(X_train, y_train)

#Training Logistic Regression model
print("Training Logistic Regression model")
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train, y_train)

#Evaluating both models
def evaluate_model(name, model):
    y_pred = model.predict(X_test)
    print(f"\n {name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

evaluate_model("SVM", svm_model)
evaluate_model("Logistic Regression", logreg_model)

#Saving the models and vectorizer
print("Saving models and vectorizer")
joblib.dump(svm_model, "best_svm_model.pkl")
joblib.dump(logreg_model, "best_logreg_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("Both models trained and saved successfully.")

