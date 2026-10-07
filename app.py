import os
import joblib
import pandas as pd
import streamlit as st

# Set page layout to wide
st.set_page_config(page_title="Twitter Sentiment Dashboard", layout="wide")

# 1. Load the saved model and vectorizer
@st.cache_resource
def load_assets():
    # Make sure your file name matches exactly (remove spaces/brackets from the filename)
    model = joblib.load("sentiment_svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl") 
    return model, vectorizer

try:
    model, vectorizer = load_assets()
except Exception as e:
    st.error(f"Error loading model assets. Please check your filenames. Details: {e}")

# 2. Sidebar Layout (Displays your champion model performance image)
st.sidebar.title("📊 Model Analytics")
image_path = "CHAMPION MODEL PERFORMANCE BY SENTIMENT CLASS.png"

if os.path.exists(image_path):
    st.sidebar.image(
        image_path, 
        caption="Champion Model Performance Metrics by Sentiment Class",
        use_container_width=True
    )
else:
    st.sidebar.warning(f"Performance image '{image_path}' not found in repository.")

st.sidebar.markdown("""
### Pipeline Details:
* **Algorithm:** Linear SVM
* **Text Processing:** TF-IDF Vectorizer
* **Task:** Multi-class Sentiment Classification
""")

# 3. Main Dashboard UI
st.title("🐦 Twitter Platform Sentiment Analysis Dashboard")
st.markdown("---")

st.subheader("🔮 Live Sentiment Inference")
st.write("Type a tweet or statement below to analyze its underlying sentiment class in real-time.")

# 4. User Input Area
user_input = st.text_area(
    "Tweet Content", 
    placeholder="e.g., I love the new updates to the platform! It runs so smoothly.",
    height=100
)

# 5. Prediction Logic
if st.button("Predict Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text before analyzing!")
    else:
        with st.spinner("Analyzing text patterns..."):
            # Transform text using your TF-IDF vectorizer
            transformed_text = vectorizer.transform([user_input])
            
            # Predict the sentiment class
            prediction = model.predict(transformed_text)[0]
            
            # Display metrics output style
            st.markdown("### Result:")
            st.info(f"The predicted sentiment class is: **{prediction}**")

