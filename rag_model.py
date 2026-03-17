# ===============================
# IMPORTS
# ===============================

import pandas as pd
import re
import numpy as np
import nltk
import faiss

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from sentence_transformers import SentenceTransformer, CrossEncoder


# ===============================
# NLTK SETUP
# ===============================

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()
nltk_stop_words = set(nltk.corpus.stopwords.words('english'))


# ===============================
# TEXT CLEANING
# ===============================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 \-/]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize_lemma(text):
    words = text.split()
    return [
        lemmatizer.lemmatize(w)
        for w in words
        if w not in nltk_stop_words and len(w) > 1
    ]


def clean_answers(text):
    text = text.replace('\u2192', '\n')
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


# ===============================
# LOAD DATASET
# ===============================

print("Loading dataset...")

data = pd.read_excel("Dataset.xlsx")
data.columns = ["Question", "Answers"]

data = data.drop_duplicates().reset_index(drop=True)

questions = data["Question"].astype(str)
answers = data["Answers"].astype(str)

answers = answers.apply(clean_answers)

raw_questions = questions.copy()
cleaned_questions = questions.apply(clean_text)

print("Dataset loaded:", len(data))


# ===============================
# TF-IDF MODEL
# ===============================

vectorizer = TfidfVectorizer(tokenizer=tokenize_lemma)
X_vectors = vectorizer.fit_transform(cleaned_questions)


# ===============================
# EMBEDDING MODEL
# ===============================

print("Loading embedding model...")

embedder = SentenceTransformer("intfloat/e5-small-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")


# ===============================
# FAISS INDEX
# ===============================

print("Creating FAISS index...")

doc_embeddings = embedder.encode(
    ["passage: " + clean_text(q) for q in raw_questions],
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

dim = doc_embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(doc_embeddings)

print("FAISS ready:", index.ntotal)


# ===============================
# RETRIEVAL
# ===============================

def retrieve_answer(user_question):

    q_emb = embedder.encode(
        ["query: " + clean_text(user_question)],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, idxs = index.search(q_emb, 5)

    chunks = []

    for s, i in zip(scores[0], idxs[0]):
        chunks.append({
            "question": raw_questions[int(i)],
            "answer": answers[int(i)],
            "score": float(s)
        })

    # rerank
    pairs = [(user_question, c["question"]) for c in chunks]
    r_scores = reranker.predict(pairs)

    for c, rs in zip(chunks, r_scores):
        c["score"] += float(rs)

    chunks.sort(key=lambda x: x["score"], reverse=True)

    if chunks[0]["score"] < 1.5:
        return "Answer not found in dataset."

    return chunks[0]["answer"]


# ===============================
# MAIN FUNCTION
# ===============================

def get_answer(question):
    return retrieve_answer(question)