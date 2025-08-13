#EXTRACT WHAT MATTERS

#use pos tagging and names entry recognition NER 
#extract nouns,verbs,named entities
#count ORG,PERSON,GPE
#vizualize with bar PythonFinalizationError

#POS tagging 
#NER 
#spacy POS tagger
#spacy NER 
#ORG ; organization names
#PERSON : peoples name 
#GPE: geopolitical

#Noun and verb extraction 
#bar plot visualation , what other was can we do, bar , line graph,pie chart
#matplotlib
#count ORG,PERSON,GPE 
import spacy
import matplotlib.pyplot as plt
from collections import Counter
nlp = spacy.load("en_core_web_sm")

text = """Apple Inc. is expanding in India. Tim Cook met Narendra Modi in New Delhi.
Microsoft and Google are investing in Bengaluru and Hyderabad."""
doc = nlp(text)

print("POS Tag Groups:")
for token in doc:
    print(f"{token.text}: {token.pos_}")

print("\nNamed Entities:")
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")

entity_labels = ["ORG", "PERSON", "GPE"]
entity_counts = Counter(ent.label_ for ent in doc.ents if ent.label_ in entity_labels)
print("\nEntity Counts:", entity_counts)

pos_counts = Counter(token.pos_ for token in doc)
print("\nPOS Tag Counts:", pos_counts)

def plot_bar(data, title, xlabel, ylabel, color, rotation):
    plt.bar(data.keys(), data.values(), color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation)
    plt.show()

def plot_pie(data, title):
    plt.pie(data.values(), labels=data.keys(), autopct='%1.1f%%')
    plt.title(title)
    plt.show()

plot_bar(entity_counts, "ORG, PERSON, GPE Counts", "Entity Type", "Count", color=['skyblue', 'pink', 'lightgreen'], rotation=0)
plot_pie(entity_counts, "Entity Type Proportions")

plot_bar(pos_counts, "POS Tag Distribution", "POS Tag", "Frequency", color=['teal'], rotation=45)
plot_pie(pos_counts, "POS Tag Proportions")