# SENTIMENT-ANALYSIS-WITH-NLP

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SAYERA YOUSUFA

*INTERN ID*: CT04DK237

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

# Customer Review Sentiment Analysis Using TF-IDF VECTORIZATION And LOGISTIC REGRESSION

This project performs sentiment analysis on customer reviews using Natural Language Processing (NLP) techniques, adapted from a Twitter sentiment analysis pipeline. It classifies reviews as positive or negative using the Amazon Fine Food Reviews dataset from Kaggle. The project preprocesses text data, extracts features with TF-IDF, trains a Logistic Regression model, and visualizes sentiment distribution.

## Project Overview

- **Objective**: Classify customer reviews as positive (4–5 stars) or negative (1–2 stars) based on their text content.
- **Dataset**: Amazon Fine Food Reviews (~568,454 reviews, sampled to 20,000 for efficiency).
- **Techniques**: Text preprocessing (stemming, stopword removal), TF-IDF vectorization, Logistic Regression.
- **Visualization**: Bar chart of sentiment distribution (positive vs. negative reviews).
- **Tools**: Python, Pandas, NLTK, Scikit-learn, Matplotlib.

## Dataset

The [Amazon Fine Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) contains 568,454 reviews with text and star ratings (1–5). For this project:
- A subset of 100,000 reviews is used to manage computational resources.
- Star ratings are converted to binary labels:
  - 1–2 stars: Negative (0)
  - 4–5 stars: Positive (1)
  - 3-star reviews are excluded to focus on clear sentiments.

## Usage

1. **Run the Jupyter Notebook**:
   - Open `customer_review_sentiment_analysis.ipynb` in Jupyter Notebook.
   - Execute cells sequentially to:
     - Download and extract the dataset.
     - Preprocess review text (stemming, stopword removal).
     - Train the Logistic Regression model.
     - Evaluate model accuracy.
     - Visualize sentiment distribution.
     - Test the model on sample reviews.

2. **Sample Prediction**:
   - The notebook includes a sample review ("This product is amazing and works perfectly!") to demonstrate prediction.
   - Output: Predicted sentiment (Positive/Negative).

## Project Structure

- `customer_review_sentiment_analysis.ipynb`: Main Jupyter Notebook with all code.
- `requirements.txt`: List of required Python packages.
- `kaggle.json`: Kaggle API token (not included; user must provide).
- `Reviews.csv`: Dataset file (downloaded via Kaggle API).

## Methodology

1. **Data Preprocessing**:
   - Remove non-alphabetic characters, convert to lowercase.
   - Apply Porter Stemming to reduce words to their root form.
   - Remove English stopwords using NLTK.

2. **Feature Extraction**:
   - Convert text to numerical features using TF-IDF Vectorizer (limited to 5,000 features).

3. **Model Training**:
   - Train a Logistic Regression model with `max_iter=1000`.
   - Split data: 80% training, 20% testing (stratified to maintain sentiment distribution).

4. **Evaluation**:
   - Compute accuracy on training and test sets (~85–90% training, ~80–85% testing).

5. **Visualization**:
   - Generate a bar chart of sentiment distribution using Matplotlib:
   
## Results

- **Accuracy**: 
  - Training: 91.9% 
  - Testing: 89.5%.
- **Sentiment Distribution**: Typically ~80% positive, ~20% negative 
- **Sample Prediction**: Correctly classifies reviews (e.g., "This product is amazing..." as Positive).

## Output Screenshots

<img width="379" alt="Image" src="https://github.com/user-attachments/assets/4ecd3534-2dc2-4918-bd31-768216536cf7" />

## Acknowledgments

- Adapted from a Twitter sentiment analysis project using the Sentiment140 dataset.
- Dataset provided by [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).
- Libraries: NLTK, Scikit-learn, Pandas, Matplotlib.


