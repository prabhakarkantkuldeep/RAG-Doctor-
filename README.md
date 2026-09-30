# 🩺 RAG Doctor

### A diagnostic and evaluation system for Retrieval-Augmented Generation (RAG)

RAG Doctor is a Streamlit-based tool that helps analyze how well a RAG pipeline retrieves information, ranks relevant evidence, generates answers, and handles failures.

Instead of building just another chatbot, RAG Doctor treats a RAG system like something that needs to be **tested and diagnosed**.

---

## 🎯 What Problem Does It Solve?

A RAG system can produce a wrong answer for different reasons.

For example:

- The relevant document was never retrieved.
- The correct document was retrieved but ranked poorly.
- The retrieved context was correct, but the LLM generated a poor answer.

RAG Doctor attempts to identify **where the problem occurred**.

```text
User Question
      ↓
Document Retrieval
      ↓
BM25 / Dense / Hybrid
      ↓
Optional Reranking
      ↓
Retrieved Evidence
      ↓
Gemini LLM
      ↓
Generated Answer
      ↓
RAG Doctor Diagnosis
