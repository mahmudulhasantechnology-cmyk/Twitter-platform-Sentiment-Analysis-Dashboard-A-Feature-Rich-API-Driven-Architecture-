import os
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# 1. Global Asset Ingestion: Load structural ML parameters into memory on startup
try:
    # Resolves exact file designations parsed from repository metrics
    MODEL_PATH = "sentiment_svm_model.pkl"
    VECTORIZER_PATH = "tfidf_vectorizer.pkl"
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Missing local serialization assets (.pkl) in backend directory.")
        
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("🚀 Custom SVM Pipeline loaded successfully into active process memory.")
except Exception as e:
    print(f"❌ Critical Failure during ML asset ingestion: {e}")
    model, vectorizer = None, None


@app.route('/')
def home():
    """Fallback fallback route to ensure status checking across testing loops."""
    return jsonify({
        "status": "online",
        "message": "Twitter Platform Sentiment Analysis Dashboard API Engine active.",
        "assets_loaded": model is not None and vectorizer is not None
    }), 200


@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    """
    Core Classification Processing Endpoint.
    Expects JSON body structural context: {"text": "Your tweet sequence here"}
    """
    # Defensive programming validations
    if not model or not vectorizer:
        return jsonify({"error": "Internal Server Error: Machine learning pipeline uninitialized configuration."}), 500

    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({"error": "Bad Request: JSON body containing a 'text' key string parameters sequence is required."}), 400

    user_text = str(data['text']).strip()
    if not user_text:
        return jsonify({"error": "Unprocessable Entity: Tracking text content cannot be blank space elements."}), 422

    try:
        # Process vector transformations natively using structural TF-IDF parameters
        transformed_vector = vectorizer.transform([user_text])
        
        # Pull index value classification prediction output arrays natively
        predicted_class = model.predict(transformed_vector)[0]
        
        # Return scannable standard JSON structure
        return jsonify({
            "success": True,
            "input_text": user_text,
            "prediction": str(predicted_class)
        }), 200

    except Exception as inner_err:
        return jsonify({
            "success": False,
            "error": f"Mathematical mapping parsing execution error sequence: {str(inner_err)}"
        }), 500


if __name__ == '__main__':
    # Initialize deployment routing loops over standard local loop frameworks
    app.run(debug=True, host='127.0.0.1', port=5000)


