# 🛡️ SpamShield — SMS Spam Detector

An AI/ML-powered SMS spam detection web app built with **Streamlit**, **scikit-learn**, and **NLTK**.

---

## 🚀 Quick Start

### 1. Clone & install dependencies
```bash
git clone https://github.com/YOUR_USERNAME/spam-shield.git
cd spam-shield
pip install -r requirements.txt
```

### 2. Add the dataset
Download the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place `spam.csv` in the project root.

### 3. Train the model
```bash
python sms_spam_train.py
```
This generates `model.pkl` and `vectorizer.pkl`.

### 4. Launch the app
```bash
streamlit run sms_spam_detector.py
```

---

## 🧠 ML Pipeline

| Step | Technique |
|------|-----------|
| Text Cleaning | Lowercasing, tokenization, punctuation removal |
| Stop Word Removal | NLTK English stopwords |
| Stemming | Porter Stemmer |
| Vectorization | TF-IDF (3000 features) |
| Classification | **Multinomial Naive Bayes** |

### Why Multinomial Naive Bayes?
Among 11 algorithms benchmarked (SVM, KNN, Random Forest, AdaBoost, XGBoost, etc.), **MNB + TF-IDF** delivered:
- ✅ ~97% Accuracy
- ✅ ~97% Precision  
- ✅ Fastest inference (ideal for real-time web app)

---

## 📁 Project Structure

```
spam-shield/
├── sms_spam_detector.py   # Streamlit web application
├── sms_spam_train.py      # Model training script
├── requirements.txt   # Python dependencies
├── spam.csv           # Dataset (add manually)
├── model.pkl          # Trained model (generated)
├── vectorizer.pkl     # TF-IDF vectorizer (generated)
└── README.md
```

---

## 📊 Dataset
**SMS Spam Collection** — 5,572 labeled SMS messages (ham/spam).  
Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

---

## 🛠️ Tech Stack
- **Streamlit** — Web UI
- **scikit-learn** — ML models & TF-IDF
- **NLTK** — NLP preprocessing
- **pandas / numpy** — Data handling

---

## 📸 Features
- Real-time spam classification
- Message statistics (chars, words, tokens)
- Preprocessed token preview
- One-click sample messages
- Dark-themed, responsive UI
