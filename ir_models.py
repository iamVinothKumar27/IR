# ir_models.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# -----------------------------
# Product Data (with image filenames)
# -----------------------------
data = [
    {"id": 1, "name": "Beige Men Coat", "description": "Stylish beige coat for men", "image": "beige men coat.jpg"},
    {"id": 2, "name": "Black Jacket", "description": "Trendy black leather jacket", "image": "black jacket.jpg"},
    {"id": 3, "name": "Blue Maxi", "description": "Elegant blue maxi dress", "image": "blue maxi.webp"},
    {"id": 4, "name": "Gown", "description": "Flowy pink gown", "image": "gown.webp"},
    {"id": 5, "name": "Mom Daughter Twinning Gown", "description": "Matching gowns for mom and daughter", "image": "mom daughter twinning gown.jpg"},
    {"id": 6, "name": "Navy Blue Body Fit", "description": "Navy blue body fit dress", "image": "navy blue body fit.jpg"},
    {"id": 7, "name": "Navy Blue Men Suite", "description": "Formal navy blue men's suit", "image": "navy blue men suite.jpg"},
    {"id": 8, "name": "Pastel Blue Shirt", "description": "Casual pastel blue shirt", "image": "pastel blue shirt.jpg"},
    {"id": 9, "name": "Purple Princess Ball Gown", "description": "Royal purple princess gown", "image": "purple princess ball gown.jpg"},
    {"id": 10, "name": "Rangola Shirt", "description": "Multi-color designer shirt", "image": "rangola shirt.jpg"},
    {"id": 11, "name": "Red Ball Gown", "description": "Gorgeous red ball gown", "image": "red ball gown.jpg"},
    {"id": 12, "name": "Sarvani Men Ethnic Wear", "description": "Traditional ethnic outfit", "image": "sarvani men ethinic wear.jpg"},
    {"id": 13, "name": "Velvet Hoodie", "description": "Comfortable green velvet hoodie", "image": "velvet hoodie.jpg"},
    {"id": 14, "name": "Violet 3 Piece", "description": "Stylish violet 3 piece set", "image": "voilet 3 piece.webp"},
]

df = pd.DataFrame(data)

# -----------------------------
# Inverted Index Build
# -----------------------------
def build_inverted_index():
    """
    Build an inverted index mapping each token to a set of document IDs.
    """
    inverted_index = {}
    for idx, row in df.iterrows():
        tokens = row["description"].lower().split()
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = set()
            inverted_index[token].add(row["id"])
    return inverted_index

# Build the inverted index once at module load
inverted_index = build_inverted_index()

# -----------------------------
# TF-IDF Setup (for Vector Space Model)
# -----------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df["description"])

# -----------------------------
# BM25 Setup
# -----------------------------
# Tokenize product descriptions for BM25
tokenized_corpus = [desc.lower().split() for desc in df["description"]]
bm25 = BM25Okapi(tokenized_corpus)

# -----------------------------
# IR Models Implementation
# -----------------------------

# 1. Boolean Model (Simple keyword matching)
def boolean_model(query):
    results = df[df["description"].str.contains(query, case=False, na=False)]
    return results.to_dict(orient="records")

# 2. Vector Space Model (TF-IDF + Cosine Similarity)
def vector_model(query):
    query_vector = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    sorted_indices = similarities.argsort()[::-1]
    return df.iloc[sorted_indices].to_dict(orient="records")

# 3. BM25 Model
def bm25_model(query):
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    sorted_indices = np.argsort(scores)[::-1]
    return df.iloc[sorted_indices].to_dict(orient="records")

# 4. Probabilistic Model (using inverted index)
def probabilistic_model(query):
    """
    A simple probabilistic retrieval model:
    For each query token, look up the documents via the inverted index.
    Then, count the number of query tokens that appear in each document.
    """
    query_tokens = query.lower().split()
    # Initialize scores for each document (by id)
    scores = {doc_id: 0 for doc_id in df['id']}
    for token in query_tokens:
        if token in inverted_index:
            for doc_id in inverted_index[token]:
                scores[doc_id] += 1  # Increase score if token is found
    # Only keep documents with a score > 0
    ranked_doc_ids = [doc_id for doc_id, score in scores.items() if score > 0]
    # Sort document IDs based on their score in descending order
    ranked_doc_ids = sorted(ranked_doc_ids, key=lambda d: scores[d], reverse=True)
    # Retrieve and return the corresponding records
    results = df[df["id"].isin(ranked_doc_ids)]
    # Order the DataFrame to match the ranked order
    results = results.set_index("id").loc[ranked_doc_ids].reset_index()
    return results.to_dict(orient="records")
