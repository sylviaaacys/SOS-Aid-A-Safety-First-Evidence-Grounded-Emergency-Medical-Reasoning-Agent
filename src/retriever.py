'''
Usage

from retrievers import make_retriever

retriever = load_retriever()
docs = retriever.invoke("someone swallowed bleach")
'''
import os
import json
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_db_first_aid")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "first-aid-corpus")
CORPUS_PATH = "/home/dev/zys/safety-medqa/safety-medqa/180_first_aid_corpus.json"

_retriever_cache = None  # singleton cache

def load_retriever():
    """
    Returns a LangChain retriever with .invoke(query) -> List[Document]
    Uses persistent Chroma. Safe to call multiple times.
    """
    global _retriever_cache
    if _retriever_cache is not None:
        return _retriever_cache

    os.makedirs(PERSIST_DIR, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # ---------- load corpus ----------
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        data = raw
    elif isinstance(raw, dict):
        for key in ("data", "items", "documents", "records", "rows"):
            if key in raw and isinstance(raw[key], list):
                data = raw[key]
                break
        else:
            raise ValueError("No list found in JSON root")
    else:
        raise ValueError("Unsupported JSON structure")

    print(f"[retriever] Loaded {len(data)} corpus items")

    # ---------- build documents ----------
    documents: List[Document] = []
    skipped = 0

    for item in data:
        if not isinstance(item, dict):
            continue

        title = (item.get("title") or "").strip()
        scenario = (item.get("scenario") or "").strip()
        content = (item.get("content") or "").strip()
        keywords = item.get("keywords") or []

        parts = []
        if title:
            parts.append(f"Title: {title}")
        if scenario:
            parts.append(f"Scenario: {scenario}")
        if keywords:
            parts.append("Keywords: " + ", ".join(map(str, keywords)))
        if content:
            parts.append(f"Content: {content}")

        text = "\n".join(parts).strip()
        if not text:
            skipped += 1
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "id": item.get("id"),
                    "title": title,
                    "source_hint": item.get("source_hint") or item.get("source"),
                },
            )
        )

    if not documents:
        raise ValueError("No documents built from corpus")

    print(f"[retriever] Built {len(documents)} documents (skipped {skipped})")

    # ---------- chunk ----------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    doc_splits = splitter.split_documents(documents)
    doc_splits = [d for d in doc_splits if d.page_content.strip()]
    print(f"[retriever] Chunks: {len(doc_splits)}")

    # ---------- stable ids ----------
    ids = []
    for i, d in enumerate(doc_splits):
        base = d.metadata.get("id") or "row"
        ids.append(f"{base}::chunk{i}")

    # ---------- load or create chroma ----------
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    if vectorstore._collection.count() == 0:
        print("[retriever] Creating Chroma index")
        vectorstore.add_documents(doc_splits, ids=ids)
    else:
        print("[retriever] Using existing Chroma index")

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    _retriever_cache = retriever
    return retriever
