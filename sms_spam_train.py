"""
train.py — Run this once to train the model and save model.pkl + vectorizer.pkl
Usage: python train.py
Requires: spam.csv in the same directory
"""

import numpy as np
import pandas as pd
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ── Download NLTK data ────────────────────────────────────────────────────────
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

# ── Text preprocessing ────────────────────────────────────────────────────────
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# ── Load & clean data ─────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv('spam.csv', encoding='latin1')
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, errors='ignore')
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode labels (ham=0, spam=1)
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Remove duplicates
df = df.drop_duplicates(keep='first')
print(f"Dataset shape after cleaning: {df.shape}")
print(f"Spam: {df['target'].sum()} | Ham: {(df['target']==0).sum()}")

# ── Feature engineering ───────────────────────────────────────────────────────
print("Preprocessing text (this may take a minute)...")
df['transformed_text'] = df['text'].apply(transform_text)

# ── Vectorise ─────────────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# ── Train Multinomial Naive Bayes ─────────────────────────────────────────────
print("Training Multinomial Naive Bayes...")
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = mnb.predict(X_test)
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print("\n── Model Performance ──────────────────────────────────")
print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
print(f"  Precision : {prec:.4f} ({prec*100:.2f}%)")
print(f"  Confusion Matrix:\n{cm}")
print("───────────────────────────────────────────────────────")

# ── Save artefacts ────────────────────────────────────────────────────────────
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb,   open('model.pkl', 'wb'))
print("\n✅ Saved vectorizer.pkl and model.pkl — ready to launch the app!")
print("   Run: streamlit run app.py")
