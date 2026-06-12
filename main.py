from scripts.loaders import load_pdfs
from scripts.chunking import split_documents
from scripts.embeddings import EmbeddingManager
from scripts.vectorstore import VectorStore
from scripts.retriever import RAGRetriever
from scripts.llm import OpenAILLM
from scripts.runRAG import rag_simple, rag_advanced 

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

    #result = rag_simple(retriever, llm, query)
    #print("\nANSWER:\n", result["answer"])

    
    result = rag_advanced("Hard Negative Mining Technqiues", retriever, llm, top_k=3, min_score=0.1, return_context=True)
    print("Answer:", result['answer'])
    print("Sources:", result['sources'])
    print("Confidence:", result['confidence'])
    print("Context Preview:", result['context'][:300])


if __name__ == "__main__":
    main()