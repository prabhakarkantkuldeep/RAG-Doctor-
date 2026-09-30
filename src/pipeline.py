from .ingestion import load_documents, build_chunks
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from .reranker import Reranker
from .generation import generate_answer
from .evaluation import recall_at_k, mrr, ndcg, lexical_grounding, first_relevant_rank
from .diagnosis import diagnose

class RAGDoctor:
    def __init__(self, documents, chunks):
        self.documents = documents
        self.chunks = chunks
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseRetriever(chunks)
        self.hybrid = HybridRetriever(self.bm25, self.dense)
        self.reranker = None

    @classmethod
    def from_directory(cls, directory):
        documents = load_documents(directory)
        chunks = build_chunks(documents)
        return cls(documents, chunks)

    def retrieve(self, query, method="BM25", top_k=5, rerank=False):
        retriever = {"BM25": self.bm25, "Dense": self.dense, "Hybrid": self.hybrid}[method]
        results = retriever.search(query, top_k)
        if rerank:
            if self.reranker is None:
                self.reranker = Reranker()
            results = self.reranker.rerank(query, results, top_k)
        return results

    def run(self, query, method="BM25", top_k=5, rerank=False):
        retrieved = self.retrieve(query, method, top_k, rerank)
        answer, generator = generate_answer(query, retrieved)
        relevant_ids = []
        diagnosis = diagnose(retrieved, relevant_ids, answer, retrieved)
        return {
            "query": query,
            "retrieved": retrieved,
            "answer": answer,
            "generator": generator,
            "diagnosis": diagnosis,
        }

    def evaluate(self, eval_set, top_k=5, rerank=False):
        rows = []
        for item in eval_set:
            for method in ["BM25", "Dense", "Hybrid"]:
                results = self.retrieve(item["question"], method, top_k, rerank)
                ids = [x["id"] for x in results]
                relevant = item["relevant_chunk_ids"]
                answer, generator = generate_answer(item["question"], results)
                rows.append({
                    "question_id": item["id"],
                    "method": method,
                    "recall@k": recall_at_k(ids, relevant, top_k),
                    "mrr": mrr(ids, relevant),
                    "nDCG": ndcg(ids, relevant, top_k),
                    "grounding": lexical_grounding(answer, results),
                    "first_relevant_rank": first_relevant_rank(ids, relevant),
                    "diagnosis": diagnose(results, relevant, answer, results)["label"],
                    "generator": generator,
                })
        return rows

    def compare(self, eval_set, top_k=5, rerank=False):
        return self.evaluate(eval_set, top_k, rerank)
