from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(pdf_directory: str):
    all_documents = []
    pdf_dir = Path(pdf_directory)

    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()

        for doc in documents:
            doc.metadata["source_file"] = pdf_file.name
            doc.metadata["file_type"] = "pdf"

        all_documents.extend(documents)
        print(f"Loaded {len(documents)} pages")

    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents