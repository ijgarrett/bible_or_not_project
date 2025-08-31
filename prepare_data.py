import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

with open("combined_dataset.json", "r", encoding = "utf-8") as f:
    data = json.load(f)

texts = [item["text"] for item in data]
labels = [item["label"] for item in data]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size = 0.2, random_state = 42, stratify = labels)

# will create sparse matrix where rows are the texts, columns are the unique words/ngrams, and values are the weight (how rare a word is)
# ngram_range (1, 2) can capture unigrams (1, 1) or single words and bigrams (1, 2) like "the king"
vectorizer = TfidfVectorizer(
    max_features = 10000,
    ngram_range = (1, 2),
    stop_words = "english"
)

X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

scaler = StandardScaler(with_mean=True)
X_train_norm = scaler.fit_transform(X_train_vec)
X_test_norm = scaler.transform(X_test_vec)

joblib.dump(vectorizer, "vectorizer.joblib")

np.save("X_train.npy", X_train_norm)
np.save("X_test.npy", X_test_norm)

np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Created and saved X and y test/train and vectorizer")