import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipeline import RAGDoctor

st.set_page_config(page_title="RAG Doctor", page_icon="🩺", layout="wide")

BASE = Path(__file__).parent
DATA = BASE / "data"
DOCS = DATA / "documents"
EVAL = DATA / "evaluation.json"

st.title("🩺 RAG Doctor")
st.caption("An experimental framework for locating retrieval, ranking, and generation failures in RAG pipelines.")

@st.cache_resource
def load_doctor():
    return RAGDoctor.from_directory(DOCS)

doctor = load_doctor()

with st.expander("What is RAG Doctor measuring?"):
    st.markdown("""
    **Retrieval:** Did the relevant evidence enter the candidate set?  
    **Ranking:** Did the relevant evidence appear near the top?  
    **Generation:** Did the answer remain grounded in the retrieved evidence?

    The benchmark intentionally mixes lexical, semantic, mixed, and stress queries so
    different retrieval strategies have opportunities to behave differently.
    """)

with st.sidebar:
    st.header("Experiment")
    method = st.selectbox("Retrieval method", ["BM25", "Dense", "Hybrid"])
    top_k = st.slider("Top-K", 1, 10, 5)
    rerank = st.checkbox("Use reranker", value=True)
    st.divider()
    st.markdown("**Demo dataset**")
    st.write(f"{len(doctor.documents)} documents · {len(doctor.chunks)} indexed chunks")
    st.caption("Set OPENAI_API_KEY to enable live LLM generation. Without it, the app uses a transparent extractive fallback.")

tab1, tab2, tab3 = st.tabs(["🔎 Diagnose a query", "📊 Evaluate dataset", "🧪 Compare retrievers"])

with tab1:
    query = st.text_input("Question", "What is the difference between BM25 and dense retrieval?")
    if st.button("Run diagnosis", type="primary"):
        result = doctor.run(query, method=method, top_k=top_k, rerank=rerank)
        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieved", len(result["retrieved"]))
        c2.metric("Best score", f'{result["retrieved"][0]["score"]:.3f}' if result["retrieved"] else "—")
        c3.metric("Diagnosis", result["diagnosis"]["label"])

        st.subheader("Generated answer")
        st.caption(f"Generator: {result['generator']}")
        st.write(result["answer"])

        st.subheader("Diagnosis")
        st.info(result["diagnosis"]["explanation"])

        st.subheader("Retrieved evidence")
        for i, item in enumerate(result["retrieved"], 1):
            with st.expander(f"{i}. {item['title']} · score {item['score']:.3f}"):
                st.write(item["text"])

with tab2:
    if st.button("Run evaluation"):
        with open(EVAL, "r", encoding="utf-8") as f:
            eval_set = json.load(f)
        with st.spinner("Running retrieval experiments..."):
            # Use a smaller benchmark K to make ranking differences visible.
            rows = doctor.evaluate(eval_set, top_k=min(top_k, 3), rerank=rerank)
        df = pd.DataFrame(rows)
        st.subheader("Aggregate results")
        summary = df.groupby("method")[["recall@k", "mrr", "nDCG", "grounding"]].mean().round(3)
        c1, c2, c3 = st.columns(3)
        c1.metric("Queries", len(eval_set))
        c2.metric("Methods", df["method"].nunique())
        c3.metric("Top-K", min(top_k, 3))
        st.dataframe(summary, use_container_width=True)
        st.subheader("Failure analysis")
        failures = df.groupby(["method", "diagnosis"]).size().reset_index(name="count")
        st.dataframe(failures, use_container_width=True)
        st.subheader("Per-query results")
        st.dataframe(df, use_container_width=True)

with tab3:
    if st.button("Compare BM25 / Dense / Hybrid"):
        with open(EVAL, "r", encoding="utf-8") as f:
            eval_set = json.load(f)
        with st.spinner("Comparing retrieval methods..."):
            rows = doctor.compare(eval_set, top_k=min(top_k, 3), rerank=rerank)
        df = pd.DataFrame(rows)
        pivot = df.groupby("method")[["recall@k", "mrr", "nDCG"]].mean().round(3)
        st.bar_chart(pivot)
        st.dataframe(pivot, use_container_width=True)

st.divider()
st.caption("RAG Doctor — an experimental framework for diagnosing RAG pipelines.")
