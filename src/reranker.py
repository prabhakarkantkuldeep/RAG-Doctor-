from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, results, k=5):
        if not results:
            return []
        pairs = [(query, item["text"]) for item in results]
        scores = self.model.predict(pairs)
        ranked = sorted(
            [{**item, "rerank_score": float(score)} for item, score in zip(results, scores)],
            key=lambda x: x["rerank_score"],
            reverse=True,
        )
        return ranked[:k]
