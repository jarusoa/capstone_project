import pandas as pd
import spacy
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

# Load dataset
file_path = "BBC_train_full.csv"  
data = pd.read_csv(file_path)

# Define a function for filtered stemming
def filtered_stem(word):
    stemmed = stemmer.stem(word)
    if stemmed in english_words or stemmed == word:
        return stemmed
    return word

# Define function for batch processing with filtered stemming and lemmatization
def preprocess_texts(texts):
    processed_texts = []
    
    # Process texts in batches using spaCy's pipe
    for doc in nlp.pipe(texts, batch_size=50):
        # Apply filtered stemming and keep tokenized words as a list
        stemmed_words = [filtered_stem(token.text) for token in doc 
                         if token.is_alpha and not token.is_stop]
        
        # Apply lemmatization on stemmed tokens and retain as a list
        lemmatized_words = [token.lemma_ for token in nlp(" ".join(stemmed_words)) 
                            if len(token.lemma_) > 1]
        
        processed_texts.append(lemmatized_words)  # Append list of words instead of joining
        
    return processed_texts

# Apply preprocessing to the 'text' column, keeping 'category'
data['processed_text'] = preprocess_texts(data['text'])

# Save only the 'category' and 'processed_text' columns to a new CSV file
data[['category', 'processed_text']].to_csv('processed_training_data.csv', index=False)


print(data[['category', 'processed_text']].head())
