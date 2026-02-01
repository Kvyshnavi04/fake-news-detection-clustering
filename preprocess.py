
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

def load_and_vectorize(path):
    df = pd.read_csv(path)
    df['cleaned'] = df['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned'])

    return X, df
