import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tokens = [self._tokenize(x["text"]) for x in chunks]
        self.index = BM25Okapi(self.tokens)

    def _tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def search(self, query, k=5):
        scores = self.index.get_scores(self._tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i]), "retriever": "BM25"} for i in order]

class DenseRetriever:
    def __init__(self, chunks, model_name="all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [x["text"] for x in chunks],
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def search(self, query, k=5):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q
        order = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i]), "retriever": "Dense"} for i in order]

def reciprocal_rank_fusion(result_lists, k=60):
    scores = {}
    items = {}
    for results in result_lists:
        for rank, item in enumerate(results, 1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            items[item_id] = item
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [{**items[i], "score": float(scores[i]), "retriever": "Hybrid"} for i in ranked]

class HybridRetriever:
    def __init__(self, bm25, dense):
        self.bm25 = bm25
        self.dense = dense

    def search(self, query, k=5):
        a = self.bm25.search(query, max(k, 10))
        b = self.dense.search(query, max(k, 10))
        return reciprocal_rank_fusion([a, b])[:k]
