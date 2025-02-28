from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    email_text = data.get("email_text", "")

    if not email_text:
        return jsonify({"error": "No email text provided"}), 400

    # Transform input text
    email_vectorized = vectorizer.transform([email_text])
    prediction = model.predict(email_vectorized)[0]

    return jsonify({"prediction": "Spam" if prediction == 1 else "Not Spam"})

if __name__ == "__main__":
    app.run(debug=True)
