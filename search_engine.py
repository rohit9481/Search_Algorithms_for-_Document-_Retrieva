import math
import re
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

class DocumentSearchEngine:
    def __init__(self):
        self.doc_raw = []
        self.documents = []
        self.metadata = []  # title, file name, upload time, length
        self.vectorizer = None
        self.tfidf_matrix = None
        self.bm25_index = []
        self.doc_lengths = []
        self.avg_doc_len = 0
        self.k1 = 1.5
        self.b = 0.75

    def preprocess(self, text):
        text = text.lower()
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    def build_index(self, file_data_list):
        self.doc_raw = []
        self.documents = []
        self.metadata = []

        for file_data in file_data_list:
            raw_text = file_data['content']
            title = file_data.get("title", file_data["filename"])
            filename = file_data["filename"]
            uploaded_at = file_data["uploaded_at"]

            self.doc_raw.append(raw_text)
            self.documents.append(self.preprocess(raw_text))
            self.metadata.append({
                "title": title,
                "filename": filename,
                "uploaded_at": uploaded_at,
                "length": len(raw_text.split())
            })

        # TF-IDF
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

        # BM25
        tokenized_docs = [doc.split() for doc in self.documents]
        self.doc_lengths = [len(doc) for doc in tokenized_docs]
        self.avg_doc_len = np.mean(self.doc_lengths)

        df = defaultdict(int)
        for doc in tokenized_docs:
            for term in set(doc):
                df[term] += 1

        self.bm25_index = []
        for doc in tokenized_docs:
            tf = defaultdict(int)
            for word in doc:
                tf[word] += 1
            scores = {}
            for word in tf:
                idf = math.log((len(tokenized_docs) - df[word] + 0.5) / (df[word] + 0.5) + 1)
                denom = tf[word] + self.k1 * (1 - self.b + self.b * len(doc) / self.avg_doc_len)
                scores[word] = idf * ((tf[word] * (self.k1 + 1)) / denom)
            self.bm25_index.append(scores)

    def hybrid_search(self, query, top_n=5, alpha=0.5):
        query_preprocessed = self.preprocess(query)
        query_vec = self.vectorizer.transform([query_preprocessed])

        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        bm25_scores = []

        query_terms = query_preprocessed.split()
        for doc_scores in self.bm25_index:
            bm25_score = sum(doc_scores.get(term, 0) for term in query_terms)
            bm25_scores.append(bm25_score)

        hybrid_scores = []
        for idx in range(len(self.doc_raw)):
            score = alpha * tfidf_scores[idx] + (1 - alpha) * bm25_scores[idx]
            # Normalize score using length (penalize very long/short docs)
            norm_length = self.metadata[idx]['length'] / self.avg_doc_len
            score = score / (1 + abs(norm_length - 1))
            hybrid_scores.append((idx, score))

        return sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]

    def get_metadata(self):
        return self.metadata

    def get_vocabulary(self):
        return self.vectorizer.get_feature_names_out() if self.vectorizer else []
