def diagnose(retrieved, relevant_ids, answer, contexts):
    retrieved_ids = [x["id"] for x in retrieved]
    relevant = set(relevant_ids)

    if relevant and not (set(retrieved_ids) & relevant):
        return {
            "label": "Retrieval failure",
            "explanation": "No annotated relevant chunk was retrieved in the candidate set. The generator cannot recover evidence that the first-stage retriever missed."
        }

    first_relevant_rank = next(
        (i for i, item_id in enumerate(retrieved_ids, 1) if item_id in relevant),
        None
    )

    if first_relevant_rank and first_relevant_rank > 1:
        return {
            "label": "Ranking failure",
            "explanation": f"Relevant evidence was retrieved, but its first occurrence was at rank {first_relevant_rank}. This indicates a ranking problem worth investigating."
        }

    answer_words = set(answer.lower().split())
    context_words = set(" ".join(x["text"] for x in contexts).lower().split())
    overlap = len(answer_words & context_words) / max(len(answer_words), 1)

    if contexts and overlap < 0.25:
        return {
            "label": "Generation failure",
            "explanation": "Relevant evidence was available, but the generated answer has low lexical overlap with the supplied evidence. Inspect grounding or generation behavior."
        }

    return {
        "label": "Pass",
        "explanation": "Relevant evidence was retrieved at or near the top and the answer shows evidence overlap."
    }
