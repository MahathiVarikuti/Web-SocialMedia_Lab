import re
import emoji
import spacy
import nltk
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

text = "Heyyyy!! https://www.apple.com/in/ this is a crazy website 😀 #apple"

def clean_text(text):
    text = re.sub(r'(https?://\S+|www\.\S+)', '', text)
    text = emoji.replace_emoji(text, '')
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# spaCy tokenizer
def spacy_tokenize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_.strip() != '']

# NLTK tokenizer
def nltk_tokenize(text):
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in stop_words and t.strip() != '']

# Process text
cleaned = clean_text(text)
spacy_tokens = spacy_tokenize(cleaned)
nltk_tokens = nltk_tokenize(cleaned)

print("spaCy Tokens:", spacy_tokens)
print("NLTK Tokens:", nltk_tokens)