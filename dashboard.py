import streamlit as st
import requests
import pandas as pd

# Configure the page layout
st.set_page_config(page_title="Twitter Sentiment Dashboard", layout="wide")
st.title("📊 Twitter Platform Sentiment Analysis Dashboard")

FLASK_API_URL = "http://127.0.0.1:5000"

# Sidebar navigation
option = st.sidebar.selectbox("Choose Mode", ["Single Text Analyzer", "Batch File Processor"])

# ==========================================
# MODE 1: SINGLE TEXT ANALYZER
# ==========================================
if option == "Single Text Analyzer":
    st.subheader("🔍 Analyze a Single Tweet")
    user_input = st.text_area("Enter your tweet text below:", placeholder="Type something here...")
    
    if st.button("Analyze Sentiment", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text before analyzing.")
        else:
            with st.spinner("Processing..."):
                try:
                    response = requests.post(f"{FLASK_API_URL}/predict", json={"text": user_input})
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Layout display metrics
                        col1, col2 = st.columns(2)
                        col1.info(f"**Cleaned Text:** {result.get('cleaned_text')}")
                        
                        sentiment = result.get('predicted_sentiment')
                        # Wrap prediction arrays safely if returned as list
                        if isinstance(sentiment, list):
                            sentiment = sentiment[0]
                            
                        col2.success(f"**Predicted Sentiment:** {sentiment}")
                    else:
                        st.error(f"Error {response.status_code}: {response.json().get('error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to Flask backend. Is your Flask app running?")

# ==========================================
# MODE 2: BATCH FILE PROCESSOR
# ==========================================
elif option == "Batch File Processor":
    st.subheader("📁 Upload Excel Dataset")
    st.write("Upload an `.xlsx` file. Ensure the last column contains the text you want to evaluate.")
    
    uploaded_file = st.file_uploader("Choose an Excel file...", type=["xlsx"])
    
    if uploaded_file is not None:
        if st.button("Process Batch File", type="primary"):
            with st.spinner("Analyzing dataset rows..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
                    response = requests.post(f"{FLASK_API_URL}/predict_batch", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Successfully evaluated {result.get('total_evaluated_records')} records!")
                        
                        # Display summary counts
                        st.markdown("### 📈 Sentiment Distribution Breakdown")
                        dist_df = pd.DataFrame(list(result.get('sentiment_distribution').items()), columns=['Sentiment', 'Count'])
                        st.dataframe(dist_df, use_container_width=True)
                        
                        # Display sample preview table
                        st.markdown("### 📋 Sample Preview Data (Top 20)")
                        samples_df = pd.DataFrame(result.get('sample_results'))
                        st.dataframe(samples_df, use_container_width=True)
                        
                    else:
                        st.error(f"Error {response.status_code}: {response.json().get('error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to Flask backend. Is your Flask app running?")
