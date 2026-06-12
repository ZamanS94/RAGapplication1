from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size=300, chunk_overlap=10):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = splitter.split_documents(documents)

    print(f"Split {len(documents)} docs into {len(chunks)} chunks")

    if chunks:
        print("\nExample chunk:")
        print(chunks[0].page_content[:100])

    return chunks