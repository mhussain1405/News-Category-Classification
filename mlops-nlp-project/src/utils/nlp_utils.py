# src/utils/nlp_utils.py
import re
import nltk
import spacy
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are downloaded if not already
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    word_tokenize("test")
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    WordNetLemmatizer().lemmatize("tests")
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True) # for wordnet

# Load spaCy model (can be slow, consider loading once if used heavily in a script)
# nlp_spacy = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def basic_cleaning(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = text.strip()
    return text

def advanced_processing(text, lemmatizer_type='nltk'):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    if lemmatizer_type == 'nltk':
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    elif lemmatizer_type == 'spacy':
        # For spaCy, it's better to process the whole doc once
        # This is a simplified example for token-wise lemmatization if text is already tokenized
        # A more efficient spaCy approach would be:
        # doc = nlp_spacy(" ".join(filtered_tokens))
        # lemmatized_tokens = [token.lemma_ for token in doc]
        # However, to keep it simple for direct replacement:
        nlp_spacy_lazy = spacy.load('en_core_web_sm', disable=['parser', 'ner']) # Load on demand
        lemmatized_tokens = [token.lemma_ for token in nlp_spacy_lazy(" ".join(filtered_tokens))]
    else:
        lemmatized_tokens = filtered_tokens # No lemmatization

    return " ".join(lemmatized_tokens)

def feature_engineering(text_series):
    """
    Generates features from a series of text.
    Args:
        text_series (pd.Series): A pandas Series containing text data.
    Returns:
        pd.DataFrame: A DataFrame with new features.
    """
    features = pd.DataFrame()
    features['text_length'] = text_series.apply(len)
    # Basic sentiment score (example, can be replaced with VADER or TextBlob)
    # This is a placeholder for simplicity.
    # A real sentiment analyzer would be more complex.
    # For TextBlob: from textblob import TextBlob
    # features['sentiment'] = text_series.apply(lambda x: TextBlob(x).sentiment.polarity)
    def basic_sentiment(text):
        positive_words = ['good', 'great', 'awesome', 'happy', 'love']
        negative_words = ['bad', 'terrible', 'sad', 'hate']
        score = 0
        for word in text.lower().split():
            if word in positive_words:
                score +=1
            if word in negative_words:
                score -=1
        return score
    features['sentiment_score'] = text_series.apply(basic_sentiment)
    return features

if __name__ == '__main__':
    sample_text = "This is a sample Text with Punctuation! And numbers 123. It's great for testing."
    print(f"Original: {sample_text}")

    cleaned = basic_cleaning(sample_text)
    print(f"Basic Cleaned: {cleaned}")

    advanced_nltk = advanced_processing(cleaned, lemmatizer_type='nltk')
    print(f"Advanced NLTK: {advanced_nltk}")

    # advanced_spacy = advanced_processing(cleaned, lemmatizer_type='spacy')
    # print(f"Advanced spaCy: {advanced_spacy}")

    # Example for feature engineering
    import pandas as pd
    data = {'text': [cleaned, advanced_nltk]}
    df_sample = pd.DataFrame(data)
    engineered_features = feature_engineering(df_sample['text'])
    print("Engineered Features:")
    print(engineered_features)