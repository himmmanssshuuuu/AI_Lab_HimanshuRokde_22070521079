import re
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def split_into_sentences(text):
    # Split the document into individual sentences
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 0]

def summarize_text(text, num_sentences=3):
    sentences = split_into_sentences(text)
    if len(sentences) <= num_sentences:
        return text

    # Convert sentences into TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Calculate similarity matrix
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Build graph and apply PageRank
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    # Rank sentences and return top ones
    ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = [s for _, s in ranked[:num_sentences]]
    return '\n'.join(summary)

import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Preprocessing function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove non-alphabets
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 2. Example dataset (you can load from CSV)
data = {
    'text': [
        'Artificial Intelligence is transforming education.',
        'Doctors are using AI for diagnosis.',
        'Banks use AI for fraud detection.',
        'Online courses personalize learning for students.',
        'Health monitoring using wearables is on the rise.'
    ],
    'label': ['Education', 'Health', 'Finance', 'Education', 'Health']
}
df = pd.DataFrame(data)

# 3. Clean the text
df['cleaned_text'] = df['text'].apply(clean_text)

# prompt: how to print the cleaned data

# ... (previous code)

# 3. Clean the text
df['cleaned_text'] = df['text'].apply(clean_text)

# 4. Print the cleaned data
print(df['cleaned_text'])
