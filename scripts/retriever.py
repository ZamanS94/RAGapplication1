from typing import List, Dict, Any


class RAGRetriever:
    def __init__(self, vectorstore, embedder):
        self.vectorstore = vectorstore
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:

        q_emb = self.embedder.embed([query])[0]

        results = self.vectorstore.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=top_k,
        )

        docs = []

        if results["documents"]:
            for i in range(len(results["documents"][0])):
                docs.append(
                    {
                        "content": results["documents"][0][i],
                        "score": 1 - results["distances"][0][i],
                    }
                )

        return docs