import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import os

# ✅ Load dataset
file_path = r"C:\Users\deepi\spam-email-detection\spam.csv"  # Update path if needed
df = pd.read_csv(file_path, encoding="latin-1")

# ✅ Ensure correct columns
df = df[['text', 'spam']]
df['spam'] = df['spam'].astype(int)

# ✅ Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X = vectorizer.fit_transform(df['text'])
y = df['spam']

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# ✅ Save Model & Vectorizer
joblib.dump(model, "spam_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model trained and saved successfully!")
