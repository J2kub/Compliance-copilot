import os
import json
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

DATA_DIR = Path("data")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True, parents=True)

def load_documents():
    docs = []
    for fname in ["nis2.txt", "gdpr.txt"]:
        path = DATA_DIR / fname
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        source = fname.split(".")[0].upper()  # NIS2 / GDPR

        # jednoduchý chunking: delíme po dvojitom odstavci
        raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        for i, chunk in enumerate(raw_chunks):
            metadata = {
                "source": source,
                "chunk_id": f"{source}_{i}",
            }
            docs.append(Document(page_content=chunk, metadata=metadata))
    return docs

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Nastav prosím OPENAI_API_KEY v prostredí.")

    documents = load_documents()
    print(f"Načítaných chunkov: {len(documents)}")

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_documents(documents, embeddings)

    # uložiť FAISS index
    faiss_path = INDEX_DIR / "faiss_index"
    vector_store.save_local(str(faiss_path))

    # uložiť metadá + texty
    meta = [
        {"content": d.page_content, "metadata": d.metadata}
        for d in documents
    ]
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Index vytvorený v priečinku 'index/'.")

if __name__ == "__main__":
    main()
