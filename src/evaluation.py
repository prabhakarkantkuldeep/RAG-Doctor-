import math
import re

def recall_at_k(retrieved_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)

def mrr(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    for rank, item_id in enumerate(retrieved_ids, 1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0

def ndcg(retrieved_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    dcg = 0.0
    for i, item_id in enumerate(retrieved_ids[:k]):
        if item_id in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0

def first_relevant_rank(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    for rank, item_id in enumerate(retrieved_ids, 1):
        if item_id in relevant:
            return rank
    return None

def lexical_grounding(answer, contexts):
    if not answer or not contexts:
        return 0.0
    answer_words = set(re.findall(r"\b\w{4,}\b", answer.lower()))
    context_words = set(re.findall(r"\b\w{4,}\b", " ".join(x["text"] for x in contexts).lower()))
    if not answer_words:
        return 0.0
    return len(answer_words & context_words) / len(answer_words)
