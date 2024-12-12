import spacy
import pandas as pd
from nltk.corpus import words
from nltk.stem.snowball import SnowballStemmer

# Initialize SpaCy, Snowball Stemmer, and load English words corpus
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
stemmer = SnowballStemmer("english")

# Download words corpus if not available
try:
    english_words = set(words.words())
except LookupError:
    import nltk
    nltk.download('words')
    english_words = set(words.words())

# Define a function for filtered stemming
def filtered_stem(word):
    stemmed = stemmer.stem(word)
    if stemmed in english_words or stemmed == word:
        return stemmed
    return word

# Function to preprocess each text entry with filtered stemming and lemmatization
def preprocess_text(text):
    # Initial stemming
    doc = nlp(text.lower())
    stemmed_words = [filtered_stem(token.text) for token in doc 
                     if token.is_alpha and not token.is_stop]
    
    # Apply lemmatization on stemmed tokens and keep as list
    lemmatized_words = [token.lemma_ for token in nlp(" ".join(stemmed_words)) 
                        if len(token.lemma_) > 1]
    
    return lemmatized_words  

# Preprocess test data file
def preprocess_test_data(test_data_path, output_path):
    # Load test data
    test_data = pd.read_csv(test_data_path)
    
    # Preprocess the 'text' column and keep tokenized lists
    test_data['processed_text'] = test_data['text'].apply(preprocess_text)
    
    # Save only the processed text column to the output file
    test_data[['processed_text']].to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

# Define paths
test_data_path = 'test_data.csv'
output_path = 'processed_test_data.csv'

# Preprocess the test data
preprocess_test_data(test_data_path, output_path)
