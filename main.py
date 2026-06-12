from scripts.loaders import load_pdfs
from scripts.chunking import split_documents
from scripts.embeddings import EmbeddingManager
from scripts.vectorstore import VectorStore
from scripts.retriever import RAGRetriever
from scripts.llm import OpenAILLM


def run_rag(retriever, llm, query: str, top_k: int = 5):

    docs = retriever.retrieve(query, top_k)

    context = "\n\n".join([d["content"] for d in docs])

    answer = llm.generate(query, context)

    return {
        "query": query,
        "context": docs,
        "answer": answer,
    }

def main():

    docs = load_pdfs("./data")

    # chunking
    chunks = split_documents(docs)

    # embeddings
    embedder = EmbeddingManager()
    embeddings = embedder.embed([c.page_content for c in chunks])

    # vector DB
    store = VectorStore()
    store.add(chunks, embeddings)

    # retriever
    retriever = RAGRetriever(store, embedder)

    # LLM
    llm = OpenAILLM()

    # query
    query = "What is sales projection?"
    result = run_rag(retriever, llm, query)

    print("\nANSWER:\n", result["answer"])


if __name__ == "__main__":
    main()