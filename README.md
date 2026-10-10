# Twitter-platform-Sentiment-Analysis-Dashboard-A-Feature-Rich-API-Driven-Architecture-

# Twitter Platform Sentiment Analysis Dashboard

## Overview

The **Twitter Platform Sentiment Analysis Dashboard** is a machine learning project designed to classify text-based social media content into four sentiment categories: Positive, Negative, Neutral, and Irrelevant.

The project implements a natural language processing (NLP) pipeline that transforms raw text into numerical features and applies machine learning classification algorithms to identify sentiment. It also includes model evaluation and serialization components to support future integration into an interactive dashboard or API-driven application.

## Project Objectives

- Analyze text data from social media platforms.
- Clean and preprocess unstructured text for machine learning.
- Convert textual data into numerical representations using TF-IDF.
- Train and compare multiple machine learning classification models.
- Evaluate model performance using standard classification metrics.
- Export the trained model and vectorizer for reuse in prediction applications.

## Key Features

- **Text preprocessing:** Lowercase conversion, URL removal, special-character removal, tokenization, and stop-word filtering.
- **Feature extraction:** TF-IDF vectorization with a vocabulary limit of 5,000 features and unigram/bigram support.
- **Multiclass classification:** Classification into Positive, Negative, Neutral, and Irrelevant categories.
- **Model comparison:** Evaluation of Logistic Regression, Multinomial Naive Bayes, and Linear Support Vector Classification.
- **Performance evaluation:** Accuracy, precision, recall, F1-score, and confusion matrix analysis.
- **Model serialization:** Export of the trained classifier and TF-IDF vectorizer as pickle files.
- **Deployment readiness:** Saved model artifacts can be used as the foundation for a prediction API or interactive dashboard.

## Machine Learning Workflow

The project follows this workflow:

1. Load the Twitter sentiment dataset.
2. Inspect the dataset and identify missing values.
3. Clean and normalize the text.
4. Tokenize the cleaned text and remove English stop words.
5. Transform text into numerical features using TF-IDF.
6. Prepare the four target sentiment classes.
7. Train and compare machine learning classifiers.
8. Evaluate the selected model on test data.
9. Save the trained classifier and vectorizer for future inference.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| NLTK | Tokenization and stop-word processing |
| Regular Expressions | Text cleaning and normalization |
| Scikit-learn | Feature extraction, model training, and evaluation |
| Matplotlib | Visualization of evaluation results |
| Jupyter Notebook | Interactive development and experimentation |
| Pickle | Serialization of the trained model and vectorizer |

## Machine Learning Models

The notebook includes the following classification algorithms:

- **Logistic Regression:** A linear classification model for multiclass sentiment prediction.
- **Multinomial Naive Bayes:** A probabilistic classifier commonly used for text classification.
- **Linear Support Vector Classification:** A linear margin-based classifier for high-dimensional text features.

The selected model is determined by the model comparison process in the notebook. Actual performance should be reported using the results produced during evaluation.

## Dataset

The notebook expects an Excel dataset named `twitter_training.xlsx` with the following columns:

| Column | Description |
|---|---|
| `ID` | Record identifier |
| `Brand` | Associated brand or entity |
| `Sentiment` | Target sentiment label |
| `Text` | Original text content |

The notebook loads the dataset using Pandas and assigns these column names. The dataset should be available in the notebook's working directory before execution.

The classification pipeline uses four sentiment labels:

- `Positive`
- `Negative`
- `Neutral`
- `Irrelevant`

Ensure that your dataset follows the expected structure and that the sentiment labels match these categories.

## Text Preprocessing

The preprocessing pipeline prepares raw text for feature extraction by performing the following operations:

1. Converts text to lowercase.
2. Removes `<unk>` tokens.
3. Removes URLs and selected social media links.
4. Removes punctuation and special characters while retaining letters, numbers, and whitespace.
5. Tokenizes the text using NLTK.
6. Removes English stop words.
7. Combines the remaining tokens into cleaned text.

Rows with missing text are removed during preprocessing.

## Feature Extraction

The project uses `TfidfVectorizer` from Scikit-learn to convert cleaned text into numerical features.

Configuration:

- Maximum features: `5000`
- N-gram range: `(1, 2)`
- Feature types: Unigrams and bigrams

This representation captures individual words and two-word combinations, enabling the classifiers to learn patterns associated with different sentiment categories.

## Model Evaluation

The notebook evaluates the trained classifier using standard classification metrics:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted positive instances for a class that are correct.
- **Recall:** Proportion of actual instances of a class that are correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Macro-average metrics:** Unweighted averages across the sentiment classes.
- **Confusion matrix:** Comparison of actual and predicted class labels.

The evaluation section reports overall accuracy and macro F1-score, along with class-level metrics. Insert the actual results from your completed notebook if you want to publish numerical performance claims.

## Project Structure

A suggested repository structure is:

```text
twitter-sentiment-analysis/
├── README.md
├── Twitter_platform_Sentiment_Analysis_Dashboard_A_Feature_Rich_API_Driven_Architecture.ipynb
├── twitter_training.xlsx
├── tfidf_vectorizer.pkl
├── sentiment_svm_model.pkl
├── app.py
├── requirements.txt
└── .gitignore
```

**Note:** This is a suggested structure, not a claim that every file is already present in your repository. The notebook exports `tfidf_vectorizer.pkl` and `sentiment_svm_model.pkl`. The application and dependency files should be included if you implement the corresponding deployment components.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd twitter-sentiment-analysis
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

Create a `requirements.txt` file containing the packages required by your implementation. For the notebook's core workflow, the main dependencies are:

```text
pandas
numpy
nltk
scikit-learn
matplotlib
openpyxl
jupyter
```

Install them with:

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place `twitter_training.xlsx` in the working directory used by the notebook.

### 5. Run the notebook

Start Jupyter:

```bash
jupyter notebook
```

Open the project notebook and execute the cells in order, from data loading and preprocessing through training, evaluation, and model export.

The notebook currently includes Google Colab-specific download code in its final cell. If you run it locally, replace the Colab download commands with standard local file-saving operations.

## Model Export

After training and evaluation, the notebook saves two artifacts:

```text
tfidf_vectorizer.pkl
sentiment_svm_model.pkl
```

- `tfidf_vectorizer.pkl` stores the fitted TF-IDF vectorizer.
- `sentiment_svm_model.pkl` stores the selected trained classifier, using the notebook's `best_model_name` selection.

Both files are needed to reproduce the trained pipeline for new text inputs. New text must be processed using the same preprocessing logic before being passed through the fitted vectorizer and classifier.

**Important:** Only load pickle files from trusted sources. Pickle files can execute malicious code when deserialized.

## Future Improvements

- Build an interactive dashboard using Streamlit.
- Develop a REST API using FastAPI or Flask.
- Add real-time sentiment prediction for user-provided text.
- Visualize sentiment distributions and model performance.
- Add error handling and input validation.
- Track experiments and compare model performance systematically.
- Evaluate the model on unseen data and monitor potential data leakage.
- Add automated tests for preprocessing and prediction.
- Deploy the application to a suitable hosting platform.

These are potential extensions and should not be interpreted as features already implemented in the notebook.

## Limitations

- Prediction quality depends on the dataset's quality, label consistency, and representativeness.
- The preprocessing pipeline is designed primarily for English text.
- Removing punctuation and special characters may discard information useful for interpreting sentiment.
- TF-IDF represents text through word and phrase statistics rather than contextual meaning.
- Performance results should be validated on held-out data before practical deployment.
- The notebook's model-selection logic and evaluation results should be reviewed before claiming a particular classifier is the best-performing model.

## Applications

Potential applications include:

- Social media sentiment monitoring.
- Brand and product feedback analysis.
- Customer opinion classification.
- Exploratory analysis of public social media discussions.
- Sentiment classification as a component of a larger NLP application.

## Author

**Mahmudul Hasan**
