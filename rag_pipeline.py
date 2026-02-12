import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

# --- NEW 2026 PHOENIX SETUP ---
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# Initialize Tracing (This connects to the Phoenix server)
tracer_provider = register()
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
# ------------------------------

DB_URI = "./lancedb_data"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def chat_with_data_setup(model_name="gemma3:4b"):
    """
    Creates a RAG chain for a SPECIFIC model.
    """
    
    # 1. SETUP EMBEDDINGS & DB
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = lancedb.connect(DB_URI)
    
    # Connect to existing table
    vector_store = LanceDB(
        connection=db,
        embedding=embedding_model,
        table_name="rag_test" 
    )

    # 2. SETUP DYNAMIC LLM
    llm = OllamaLLM(model=model_name)

    # 3. STRICT SYSTEM PROMPT
    template = """
    You are a precise technical auditor. Answer the question strictly based on the context below.
    
    Rules:
    1. If the answer is not in the context, say "I don't know".
    2. Do not hallucinate facts.
    3. Keep the answer concise (under 4 sentences).
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # 4. BUILD CHAIN
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 8}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return qa_chain