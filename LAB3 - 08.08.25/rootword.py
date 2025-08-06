#find word roots in tweets
#apply stemming and lemmatization to real-world text
#scrape 100 tweets using snscrape(read about this clearly- works with 3.14 version)
#use porter/snowball stemmers
# lemmatization via spaCy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         #lemmatization via spaCy
#compare word frequencies
#rootwords.py
import re
import spacy
import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords
from collections import Counter


nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

porter = PorterStemmer()
snowball = SnowballStemmer("english")
nlp = spacy.load("en_core_web_sm")

with open("C:/Users/maheit/Desktop/dsd10/dataset.txt", "r") as f:    
    lines = f.readlines()

def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\w+|#\w+|[^\w\s]", "", text)
    return text.strip()

porter_words, snowball_words, lemma_words = [], [], []

for line in lines:
    line = clean(line)
    tokens = line.split()
    tokens = [t for t in tokens if t not in stop_words]

    porter_words += [porter.stem(t) for t in tokens]
    snowball_words += [snowball.stem(t) for t in tokens]

    doc = nlp(line)
    lemma_words += [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]
porter_freq = Counter(porter_words)
snowball_freq = Counter(snowball_words)
lemma_freq = Counter(lemma_words)

print("\n Porter    :", porter_freq.most_common(10))
print(" Snowball  :", snowball_freq.most_common(10))
print(" Lemmatized:", lemma_freq.most_common(10))

with open("C:/Users/maheit/Desktop/dsd10/word_comparison.txt", "w") as out:
    out.write("Porter:\n" + "\n".join(f"{w}: {c}" for w, c in porter_freq.most_common(20)) + "\n\n")
    out.write("Snowball:\n" + "\n".join(f"{w}: {c}" for w, c in snowball_freq.most_common(20)) + "\n\n")
    out.write("Lemmatized:\n" + "\n".join(f"{w}: {c}" for w, c in lemma_freq.most_common(20)))

print("\nDone! Results saved to 'word_comparison.txt'")