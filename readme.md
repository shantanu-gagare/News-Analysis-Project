# News Article Project

## Overview
This Python script automates the process of sentiment analysis, summarization, and thematic categorization of news articles. Utilizing Natural Language Processing (NLP) techniques, the pipeline provides insights into the sentiment and themes prevalent in news articles.

## Objective
The objective of the News Analysis Project is to develop a comprehensive system capable of enhancing the comprehension and summarization of news articles through advanced natural language processing (NLP). The system aims to provide valuable insights into the emotional tone and thematic connections across various articles, thereby aiding stakeholders in making informed decisions based on current news trends.


## Setup
To set up your environment to run this script, follow these steps:

1. **Install Required Libraries**  
   Use the provided `setup.py` to install necessary libraries. Run:


2. **Download NLTK Resources**  
Run the following commands in your Python environment to download necessary NLTK resources:


## Input Data Preparation
Prepare your input data in an Excel file named `news_articles.xlsx`. Each article should be stored in a separate row under the column named 'Article'. Save this file in the project directory.

## Key Files
- **svm_sentiment_model.joblib**: Trained SVM model for sentiment analysis.
- **tfidf_vectorizer.joblib**: TF-IDF vectorizer for text data transformation.

Ensure both files are in the project directory or specified path for successful execution.

## Running the Script
1. **Modify the Script**  
Open `final_code.py` and update the `process_articles()` function call at the end of the script to point to your input file, e.g., `process_articles('news_articles.xlsx')`.

2. **Execute the Script**  
Run the following command in your terminal:


## Output
After execution, the script will generate an Excel file named `final_processed_articles.xlsx`. This file contains the original articles, their summaries, sentiment predictions, and categorized themes.

## Troubleshooting
If you encounter issues, ensure all libraries are installed correctly and that both model and vectorizer files are accessible by the script. Also, verify the format of the input Excel file as described in the Input Data Preparation section.

