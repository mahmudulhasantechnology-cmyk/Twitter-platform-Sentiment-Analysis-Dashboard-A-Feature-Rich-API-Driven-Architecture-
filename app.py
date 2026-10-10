import os
import re
import pickle
import pandas as pd
import nltk
from flask import Flask, request, jsonify, render_template
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

# ==============================================================================
# 1. INITIALIZE DEPENDENCIES (Runs once on startup)
# ==============================================================================
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')

download_nltk_resources()
stop_words = set(stopwords.words('english'))

def load_pipeline_assets():
    """Loads the serialized TF-IDF vectorizer and the champion SVM model."""
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as vf:
            vectorizer = pickle.load(vf)
        with open('sentiment_svm_model.pkl', 'rb') as mf:
            model = pickle.load(mf)
        return vectorizer, model
    except FileNotFoundError:
        return None, None

tfidf, svm_model = load_pipeline_assets()

# ==============================================================================
# 2. DEFINE EXACT WORKSPACE CLEANING PIPELINE
# ==============================================================================
def clean_text(text):
    text = str(text).lower()
    text = text.replace('<unk>', '')
    text = re.sub(r'https?://\S+|www\.\S+|pic\.twitter\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = word_tokenize(text)
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# ==============================================================================
# 3. FLASK ENDPOINTS (API ROUTES & DASHBOARD)
# ==============================================================================

@app.route('/', methods=['GET'])
def home():
    """Renders the dashboard UI or provides API confirmation status to prevent 404 errors."""
    try:
        # Tries to render the dashboard UI if 'templates/index.html' exists
        return render_template('index.html')
    except Exception:
        # Fallback response for browser confirmation if no index.html UI exists yet
        return jsonify({
            "status": "online",
            "message": "Twitter Sentiment Analysis API is running successfully. Send POST requests to /predict or /predict_batch.",
            "pipeline_loaded": tfidf is not None and svm_model is not None
        }), 200


@app.route('/predict', methods=['POST'])
def predict_single():
    """Endpoint for Single Text Analyzer"""
    if not tfidf or not svm_model:
        return jsonify({"error": "Pipeline asset files missing on server"}), 500
        
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' key in JSON request payload"}), 400
        
    user_input = data['text']
    if not user_input.strip():
        return jsonify({"error": "The provided text is empty"}), 400
        
    cleaned_phrase = clean_text(user_input)
    if not cleaned_phrase.strip():
        return jsonify({"error": "The text provided contains only noise/stopwords"}), 400
        
    # Transform and Predict
    vector_input = tfidf.transform([cleaned_phrase])
    prediction = svm_model.predict(vector_input)[0]
    
    return jsonify({
        "original_text": user_input,
        "cleaned_text": cleaned_phrase,
        "predicted_sentiment": prediction
    })


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Endpoint for Batch File Processor"""
    if not tfidf or not svm_model:
        return jsonify({"error": "Pipeline asset files missing on server"}), 500
        
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if not file.filename.endswith('.xlsx'):
        return jsonify({"error": "Invalid file type. Please upload an Excel sheet (.xlsx)"}), 400

    try:
        df = pd.read_excel(file, header=None)
        
        # Match schema structure safely
        if len(df.columns) >= 4:
            df.columns = ['ID', 'Brand', 'Sentiment_Actual', 'Text'] + list(df.columns[4:])
        else:
            df.rename(columns={df.columns[-1]: 'Text'}, inplace=True)
            
        df = df.dropna(subset=['Text'])
        df['Cleaned_Text'] = df['Text'].apply(clean_text)
        
        # Filter empty rows
        valid_mask = df['Cleaned_Text'].str.strip() != ''
        df_valid = df[valid_mask].copy()
        
        if len(df_valid) == 0:
            return jsonify({"error": "All entries in the uploaded file became empty after preprocessing"}), 400
            
        # Predict
        batch_vectors = tfidf.transform(df_valid['Cleaned_Text'])
        df_valid['Predicted_Sentiment'] = svm_model.predict(batch_vectors)
        
        # Prepare metrics for JSON return
        metrics_matrix = df_valid['Predicted_Sentiment'].value_counts().to_dict()
        samples = df_valid[['Text', 'Cleaned_Text', 'Predicted_Sentiment']].head(20).to_dict(orient='records')
        
        return jsonify({
            "status": "success",
            "total_evaluated_records": len(df_valid),
            "sentiment_distribution": metrics_matrix,
            "sample_results": samples
        })
        
    except Exception as e:
        return jsonify({"error": f"Error parsing data parameters: {str(e)}"}), 500

if __name__ == '__main__':
    # Start the Flask app locally on http://127.0.0.1:5000
    app.run(debug=True, port=5000)
