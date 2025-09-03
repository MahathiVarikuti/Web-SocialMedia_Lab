#explore lexical diversity and extract top keywords usinf TF-IDF
#1.calculate lexical diversity using formula -> no  of unique words/total words
#2.generate unigrams,bigrams and trigrams 
#3.build a TD-IF model using TfidVectorizer
#4.take out the top keywords per article
#TF-IDF term feq, 
#info diff funcs
#matrices in the form of graph 
# Import necessary libraries

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('punkt')

docs = [
    "Artificial intelligence is transforming industries by automating tasks and improving efficiency.",
    "Machine learning models can identify patterns in data and make accurate predictions.",
    "Natural language processing allows computers to understand and generate human language.",
    "Big data analytics helps organizations make data-driven decisions and uncover hidden insights.",
    "Cloud computing provides scalable resources and enables remote collaboration.",
    "Cybersecurity is critical for protecting sensitive information and maintaining trust.",
    "Internet of Things connects devices and enables smart environments.",
    "Blockchain technology ensures transparency and security in digital transactions.",
    "Augmented reality blends digital content with the physical world for immersive experiences.",
    "Quantum computing promises to solve complex problems beyond the reach of classical computers."
]

# calculate lexical diversity
def lexical_diversity(text):
    tokens = nltk.word_tokenize(text.lower())
    return len(set(tokens)) / len(tokens)

# 1. Lexical Diversity
print("=== Lexical Diversity ===")
for i, doc in enumerate(docs):
    score = lexical_diversity(doc)
    print(f"Doc {i+1}: {score:.2f}")

# 2. N-grams
print("\n=== N-grams ===")
for i, doc in enumerate(docs):
    tokens = nltk.word_tokenize(doc.lower())
    print(f"\nDoc {i+1}:")
    for n in [1, 2, 3]:
        ng_list = list(ngrams(tokens, n))
        print(f"{n}-grams: {ng_list}")

# 3. TF-IDF Matrix
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print("\n=== TF-IDF Matrix ===")
print(df.round(2))  # Rounded for readability

# 4. Top TF-IDF Terms per Document
print("\n=== Top TF-IDF Terms per Document ===")
for idx, row in df.iterrows():
    top_terms = row.sort_values(ascending=False).head(3)
    print(f"Doc {idx+1}:")
    for term, score in top_terms.items():
        print(f"  {term}: {score:.3f}")

# 5. TF-IDF graph
term_scores = df.sum(axis=0).sort_values(ascending=False).head(10)

# Plot top 10 TF-IDF terms across all documents
plt.figure(figsize=(10, 5))
term_scores.plot(kind='bar', color='coral')
plt.title("Top 10 TF-IDF Terms Across All Documents")
plt.ylabel("Total TF-IDF Score")
plt.xlabel("Terms")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
