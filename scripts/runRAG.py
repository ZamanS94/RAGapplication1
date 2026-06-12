
def rag_simple(retriever, llm, query: str, top_k: int = 5):

    docs = retriever.retrieve(query, top_k)

    context = "\n\n".join([d["content"] for d in docs]) if docs else " "

    answer = llm.generate(query, context)

    return {
        "query": query,
        "context": docs,
        "answer": answer,
    }

def rag_advanced(query, retriever, llm, top_k=5, min_score=0.2, return_context=False):

    results = retriever.retrieve(
        query,
        top_k=top_k,
        score_threshold=min_score
    )

    if not results:
        return {
            "answer": "No relevant context found.",
            "sources": [],
            "confidence": 0.0,
            "context": ""
        }

    # build context
    context = "\n\n".join([doc["content"] for doc in results])

    # build sources safely (NO metadata assumption)
    sources = [{
        "score": doc["score"],
        "preview": doc["content"][:300] + "..."
    } for doc in results]

    confidence = max(doc["score"] for doc in results)

    # consistent LLM call (same as run_rag)
    answer = llm.generate(query, context)

    output = {
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }

    if return_context:
        output["context"] = context

    return output