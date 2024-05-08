import pandas as pd
from joblib import load
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import heapq
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer


#================================================Dont change here================================================
# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved models for sentiment analysis
model = load('svm_sentiment_model.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

# Define categories and their keywords with weights for categorization
categories = {
    'General Business': {
        'finance': 3, 'economic': 2, 'investment': 3, 'trade': 2, 'corporate': 2, 'commercial': 2
    },
    'Legal and Politics': {
        'lawsuit': 3, 'legal': 2, 'law': 2, 'regulation': 2, 'politics': 3, 'election': 3, 'policy': 2, 'government': 3
    },
    'Health and Medicine': {
        'health': 3, 'medical': 3, 'therapy': 2, 'surgery': 2, 'clinical': 3, 'pharmaceutical': 3, 'disease': 3,
        'cancer': 5, 'medicine': 4, 'fda': 4, 'treatment': 3, 'healthcare': 3, 'nursing': 2, 'diagnosis': 3,
        'patient': 2, 'hospital': 3, 'doctor': 3, 'nurse': 2, 'drug': 2, 'vaccine': 4, 'mental health': 3
    },
    'Science and Technology': {
        'technology': 4, 'innovation': 3, 'science': 3, 'research': 3, 'digital': 2, 'data': 2, 'ai': 4,
        'machine learning': 4, 'biotechnology': 3, 'engineering': 2, 'robotics': 3, 'quantum': 2, 'tech': 2
    },
    'Arts and Entertainment': {
        'movie': 2, 'music': 2, 'theater': 2, 'festival': 2, 'entertainment': 3, 'concert': 2, 'celebrity': 2,
        'art': 2, 'culture': 2
    },
    'Environment and Energy': {
        'environment': 4, 'climate': 4, 'energy': 3, 'conservation': 3, 'sustainable': 3, 'pollution': 3,
        'wildlife': 2, 'ecology': 3, 'nature': 2
    },
    'Education': {
        'education': 4, 'university': 3, 'school': 3, 'student': 2, 'academic': 2, 'learning': 2,
        'scholarship': 2, 'campus': 2, 'teacher': 2, 'curriculum': 2
    },
    'Travel and Lifestyle': {
        'travel': 3, 'tourism': 2, 'accommodation': 2, 'lifestyle': 2, 'leisure': 2, 'destination': 3,
        'adventure': 2, 'holiday': 2, 'tourist': 2
    },
    'Sports': {
        'sports': 4, 'athlete': 2, 'tournament': 3, 'game': 2, 'competition': 2, 'team': 2, 'match': 2,
        'sporting': 2, 'coach': 2
    },
    'Real Estate and Infrastructure': {
        'real estate': 4, 'property': 3, 'construction': 3, 'infrastructure': 3, 'housing': 2,
        'development': 2, 'building': 2, 'realty': 2
    },
    'Retail and Consumer Goods': {
        'retail': 3, 'shopping': 2, 'consumer': 2, 'brand': 2, 'store': 2, 'product': 2, 'market': 2,
        'sales': 2, 'customer': 2
    },
    'Food and Dining': {
        'food': 3, 'restaurant': 3, 'cuisine': 2, 'meal': 2, 'diner': 2, 'eatery': 2, 'dining': 2,
        'cooking': 2, 'gourmet': 2, 'chef': 2, 'culinary': 2, 'menu': 2, 'bistro': 2, 'cafe': 2,
        'nutrition': 3, 'organic': 2, 'eat': 2, 'drink': 2, 'taste': 2
    }
}


def preprocess_text(text):
    """Preprocess text by removing stopwords, punctuation, and applying stemming."""
    stop_words = set(stopwords.words('english')) | set(string.punctuation)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in word_tokenize(text.lower()) if word not in stop_words and word.isalnum()]
    return " ".join(tokens)

def predict_sentiment(text):
    """Predict sentiment of the preprocessed text."""
    cleaned_text = preprocess_text(text)
    features = vectorizer.transform([cleaned_text])
    prediction = model.predict(features)[0]
    return {1: 'Positive', 0: 'Neutral', -1: 'Negative'}.get(prediction, 'Neutral')

def nltk_summarize(text, n_sentences=2):
    """Generate a summary of the given text."""
    stop_words = set(stopwords.words('english')) | set(string.punctuation)
    word_frequencies = FreqDist(word.lower() for word in word_tokenize(text) if word.lower() not in stop_words and word.isalnum())
    sentence_scores = {}
    for sentence in sent_tokenize(text):
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]
    summary_sentences = heapq.nlargest(n_sentences, sentence_scores, key=sentence_scores.get)
    return ' '.join(summary_sentences)

def get_relevant_category(text, categories):
    words = word_tokenize(text.lower())
    word_freq = Counter(words)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words and word.isalnum()]

    highest_score = 0
    relevant_category = 'None'

    for category, keywords in categories.items():
        category_score = sum(word_freq[word] * weight for word, weight in keywords.items() if word in filtered_words)
        if category_score > highest_score:
            highest_score = category_score
            relevant_category = category

    return relevant_category

def process_articles(file_path):
    """Process each article for summary, sentiment, and categorization."""
    data = pd.read_excel(file_path)
    data['Summary'] = data['Article'].apply(lambda x: nltk_summarize(x, n_sentences=2))
    data['Sentiment'] = data['Summary'].apply(predict_sentiment)
    data['Theme'] = data['Article'].apply(lambda x: get_relevant_category(x, categories))
    output_path = 'final_processed_articles.xlsx'
    data.to_excel(output_path, index=False)


# ================================================= Change the name of excel file below ===================

# Give the path of excel file 
process_articles('Assignment.xlsx')
