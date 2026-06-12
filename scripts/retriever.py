from typing import List, Dict, Any


class RAGRetriever:
    def __init__(self, vectorstore, embedder):
        self.vectorstore = vectorstore
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:

        # embed query
        q_emb = self.embedder.embed([query])[0]

        # vector search
        results = self.vectorstore.collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=top_k,
        )

        docs = []

        if results.get("documents"):
            for i in range(len(results["documents"][0])):

                score = 1 - results["distances"][0][i]  # convert distance → similarity

                # apply threshold filter
                if score >= score_threshold:
                    docs.append({
                        "content": results["documents"][0][i],
                        "score": score,
                    })

        docs.sort(key=lambda x: x["score"], reverse=True)

        return docs