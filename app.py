from flask import Flask, request, jsonify
import joblib
import traceback

app = Flask(__name__)

# ✅ Load Model & Vectorizer with Error Handling
try:
    model = joblib.load("spam_classifier.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("✅ Model and Vectorizer Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")
    model, vectorizer = None, None

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "email_text" not in data:
            return jsonify({"error": "Invalid input"}), 400

        text = [data["email_text"]]
        text_vectorized = vectorizer.transform(text)
        prediction = model.predict(text_vectorized)[0]
        
        return jsonify({"prediction": "Spam" if prediction == 1 else "Not Spam"})
    
    except Exception as e:
        print("❌ Prediction Error:", traceback.format_exc())  # ✅ Debugging Info
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # ✅ Debug mode enabled
