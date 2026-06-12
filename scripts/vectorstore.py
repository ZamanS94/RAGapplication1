import chromadb
import uuid
import os
from typing import List, Any
import numpy as np


class VectorStore:
    def __init__(self, persist_dir="./vector_store", collection_name="pdf_docs"):
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        print("Vector store ready")

    def add(self, docs: List[Any], embeddings: np.ndarray):

        ids, texts, metas, embs = [], [], [], []

        for i, (doc, emb) in enumerate(zip(docs, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}")

            texts.append(doc.page_content)

            meta = dict(doc.metadata)
            meta["length"] = len(doc.page_content)
            metas.append(meta)

            embs.append(emb.tolist())

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embs,
            metadatas=metas,
        )

        print(f"Stored {len(docs)} chunks")