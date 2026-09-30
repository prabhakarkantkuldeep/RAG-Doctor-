from pathlib import Path
import re

def load_documents(directory):
    documents = []
    for path in sorted(Path(directory).glob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append({
                "id": path.stem,
                "title": path.name,
                "text": text,
            })
    return documents

def chunk_text(text, chunk_size=180, overlap=40):
    words = re.findall(r"\S+", text)
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks

def build_chunks(documents, chunk_size=180, overlap=40):
    chunks = []
    for doc in documents:
        for idx, text in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            chunks.append({
                "id": f'{doc["id"]}_chunk_{idx}',
                "doc_id": doc["id"],
                "title": doc["title"],
                "text": text,
            })
    return chunks
