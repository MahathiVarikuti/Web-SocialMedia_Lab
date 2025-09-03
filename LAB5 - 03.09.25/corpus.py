#explore lexical diversity and extract top keywords usinf TF-IDF
#1.calculate lexical diversity using formula -> no  of unique words/total words
#2.generate unigrams,bigrams and trigrams 
#3.build a TD-IF model using TfidVectorizer
#4.take out the top keywords per article
#TF-IDF term feq, 
#info diff funcs
#matrices in the form of graph 
import nltk, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import string

nltk.download('punkt')

text = """
Artificial intelligence is transforming industries across the globe.
From healthcare to finance, AI systems are improving efficiency and decision-making.
Machine learning algorithms analyze vast datasets to uncover patterns and insights.
Natural language processing allows computers to understand and generate human language.
Robotics powered by AI are revolutionizing manufacturing and logistics.
Ethical concerns around bias and transparency continue to shape AI development.
Governments and organizations are investing heavily in AI research and innovation.
Education systems are adapting to prepare students for an AI-driven future.
The integration of AI into daily life raises questions about privacy and control.
Despite challenges, the potential of AI to solve complex problems remains immense.
"""

docs = text.strip().split("\n")

def clean(text): return [t for t in word_tokenize(text.lower()) if t not in string.punctuation]

def lexical(docs):
    print("Lexical Diversity:")
    for i, d in enumerate(docs):
        t = clean(d)
        print(f"Doc {i+1}: {len(set(t))/len(t):.2f} (Words: {len(t)})")
lexical(docs)

def ngram_show(docs):
    print("\nN-grams:")
    for i, d in enumerate(docs):
        t = clean(d)
        print(f"\nDoc {i+1}:")
        print("Unigrams:", list(ngrams(t, 1)))
        print("Bigrams:", list(ngrams(t, 2)))
        print("Trigrams:", list(ngrams(t, 3)))
ngram_show(docs)

def tfidf_model(docs):
    v = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
    m = v.fit_transform(docs)
    return m, v.get_feature_names_out()
matrix, terms = tfidf_model(docs)

def top_keywords(m, terms, n=5):
    print("\nTop Keywords:")
    for i in range(m.shape[0]):
        row = m[i].toarray()[0]
        top = row.argsort()[::-1][:n]
        print(f"Doc {i+1}: {[terms[j] for j in top if row[j] > 0]}")
top_keywords(matrix, terms)

def show_tfidf_matrix(m, terms):
    print("\nTF-IDF Matrix:")
    df = pd.DataFrame(m.toarray(), columns=terms)
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', None)        # Prevent line wrapping
    print(df.round(2))                          # Round values for readability
show_tfidf_matrix(matrix, terms)