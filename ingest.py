import os
import lancedb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB

# 1. SETUP
# Define where the database will be saved
DB_URI = "./lancedb_data"
# Define which model to use for embeddings (runs locally!)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_documents():
    print("🚀 Starting Ingestion Pipeline...")

    # 2. LOAD PDF
    # Looks for any PDF in the 'data' folder
    data_folder = "./data"
    documents = []
    for file in os.listdir(data_folder):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, file)
            print(f"📄 Loading: {file}")
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    if not documents:
        print("❌ No PDFs found in 'data/' folder!")
        return

    # 3. CHUNK (SPLIT) DOCUMENTS
    # M.Tech Level: recursive splitting keeps context better than simple splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"🧩 Split {len(documents)} pages into {len(chunks)} chunks.")

    # 4. EMBED & STORE
    print("🧠 Loading Embedding Model (this might take a moment)...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("💾 Storing in LanceDB...")
    # This automatically connects to LanceDB, creates the table, converts text to vectors, and saves it.
    db = lancedb.connect(DB_URI)
    
    # We use the LangChain wrapper for LanceDB to make retrieval easier later
    vector_store = LanceDB.from_documents(
        documents=chunks,
        embedding=embedding_model,
        connection=db,
        table_name="rag_test"
    )

    print("✅ Ingestion Complete! Database is ready.")

if __name__ == "__main__":
    ingest_documents()