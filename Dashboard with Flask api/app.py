import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ==============================================================================
# 1. INITIALIZE & CACHE DEPENDENCIES (Prevents reloading components on rerun)
# ==============================================================================
@st.cache_resource
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

@st.cache_resource
def load_pipeline_assets():
    """Loads the serialized TF-IDF vectorizer and the champion SVM model."""
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as vf:
            vectorizer = pickle.load(vf)
        with open('sentiment_svm_model.pkl', 'rb') as mf:
            model = pickle.load(mf)
        return vectorizer, model
    except FileNotFoundError:
        st.error("⚠️ Pipeline asset files missing! Ensure 'tfidf_vectorizer.pkl' and 'sentiment_svm_model.pkl' are placed in the same directory as app.py.")
        return None, None

tfidf, svm_model = load_pipeline_assets()

# ==============================================================================
# 2. DEFINE EXACT WORKSPACE CLEANING PIPELINE
# ==============================================================================
def clean_text(text):
    """Exact text cleaning configuration extracted from the training pipeline."""
    text = str(text).lower()
    text = text.replace('<unk>', '')
    text = re.sub(r'https?://\S+|www\.\S+|pic\.twitter\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = word_tokenize(text)
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)

# ==============================================================================
# 3. STREAMLIT FRONT-END LAYOUT & BRANDING
# ==============================================================================
st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="📊", layout="wide")

st.title("📊 Twitter Platform Sentiment Analysis Dashboard")
st.markdown("A feature-rich, interactive workspace powered by a custom-trained **Linear Support Vector Machine** engine.")
st.markdown("---")

# Sidebar for workspace choices
st.sidebar.header("📁 Operational Modes")
app_mode = st.sidebar.radio("Choose Analysis Target:", ["Single Text Analyzer", "Batch File Processor"])

# ==============================================================================
# MODE A: SINGLE TEXT ANALYZER
# ==============================================================================
if app_mode == "Single Text Analyzer":
    st.subheader("📝 Real-Time Tweet Sentiment Analysis")
    user_input = st.text_area("Type or paste a tweet below:", placeholder="Type something like: 'Wow, the new gameplay leak looks incredible! <unk>'")
    
    if st.button("Run Prediction Pipeline", type="primary"):
        if user_input.strip() == "":
            st.warning("Please type a valid phrase before analyzing.")
        elif tfidf and svm_model:
            # 1. Clean input
            cleaned_phrase = clean_text(user_input)
            
            if cleaned_phrase.strip() == "":
                st.error("The text provided contains only noise/stopwords. Please try a different phrase.")
            else:
                # 2. Transform and Predict
                vector_input = tfidf.transform([cleaned_phrase])
                prediction = svm_model.predict(vector_input)[0]  # Safely extract the string classification label
                
                # 3. UI Display Map
                color_map = {
                    "Positive": "green",
                    "Negative": "red",
                    "Neutral": "blue",
                    "Irrelevant": "grey"
                }
                text_color = color_map.get(prediction, "black")
                
                st.markdown(f"### Predicted Sentiment: :{text_color}[{prediction}]")
                
                with st.expander("🔍 Review Structural Preprocessing Metrics"):
                    st.write(f"**Original Input Text:** {user_input}")
                    st.write(f"**Cleaned Processing String:** `{cleaned_phrase}`")

# ==============================================================================
# MODE B: BATCH FILE PROCESSOR
# ==============================================================================
else:
    st.subheader("📂 Batch File Pipeline Processing Layout")
    st.markdown("Upload an Excel dataset containing a text column to view platform distribution metrics.")
    
    uploaded_file = st.file_uploader("Upload Excel Matrix Sheet (.xlsx)", type=["xlsx"])
    
    if uploaded_file is not None and tfidf and svm_model:
        # Load File
        try:
            df = pd.read_excel(uploaded_file, header=None)
            
            # Match schema structure safely
            if len(df.columns) >= 4:
                df.columns = ['ID', 'Brand', 'Sentiment_Actual', 'Text'] + list(df.columns[4:])
            else:
                st.warning("The system is mapping the last column as your text source since the dataset doesn't have 4 standard structural columns.")
                df.rename(columns={df.columns[-1]: 'Text'}, inplace=True)
                
            # Drop empty data rows
            df = df.dropna(subset=['Text'])
            
            with st.spinner("Executing pipeline predictions across all matrix samples..."):
                # Clean, Transform, and Predict
                df['Cleaned_Text'] = df['Text'].apply(clean_text)
                
                # Filter records that became empty strings to keep execution matrices safe
                valid_mask = df['Cleaned_Text'].str.strip() != ''
                df_valid = df[valid_mask].copy()
                
                if len(df_valid) == 0:
                    st.error("All entries in the uploaded file became empty strings after text preprocessing.")
                else:
                    batch_vectors = tfidf.transform(df_valid['Cleaned_Text'])
                    df_valid['Predicted_Sentiment'] = svm_model.predict(batch_vectors)
                    
                    st.success(f"Processing Complete! Successfully evaluated {len(df_valid)} valid records.")
                    st.markdown("---")
                    
                    # Layout grid components - Explicitly assigned count parameter
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📊 Classification Sentiment Spread")
                        metrics_matrix = df_valid['Predicted_Sentiment'].value_counts()
                        st.dataframe(metrics_matrix.rename("Total Mentions"))
                        
                    with col2:
                        st.markdown("### 📈 Metric Distributions")
                        # Bar Chart Visualization natively in Streamlit
                        st.bar_chart(metrics_matrix)
                    
                    st.markdown("### 🔍 Sample Pipeline Matrix View")
                    st.dataframe(df_valid[['Text', 'Cleaned_Text', 'Predicted_Sentiment']].head(20))
                    
        except Exception as error:
            st.error(f"Error parsing data parameters: {error}")

