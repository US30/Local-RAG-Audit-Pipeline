import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

# 1. SETUP
DB_URI = "./lancedb_data"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Make sure you have this model pulled in Ollama (ollama pull llama3.1:8b)
LLM_MODEL = "llama3.1:8b" 

def chat_with_data():
    print("🧠 Loading Retrieval Pipeline...")

    # 2. CONNECT TO DATABASE
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = lancedb.connect(DB_URI)
    
    # Connect LangChain to the existing LanceDB table
    vector_store = LanceDB(
        connection=db,
        embedding=embedding_model,
        table_name="rag_test"
    )

    # 3. SETUP THE LLM (OLLAMA)
    # This connects to your local Ollama server running on localhost:11434
    llm = Ollama(model=LLM_MODEL)

    # 4. DEFINE THE PROMPT
    # This tells the LLM to behave like a technical expert
    template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer technical and concise.
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # 5. CREATE THE CHAIN
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    print("✅ Pipeline Ready! (Type 'exit' to quit)")
    print("-" * 50)

    # 6. INTERACTIVE LOOP
    while True:
        query = input("\n❓ Ask a question: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        # Run the RAG pipeline
        print("Thinking...")
        result = qa_chain.invoke({"query": query})
        
        print(f"\n💡 Answer: {result['result']}")
        print("\n🔍 Sources:")
        for doc in result['source_documents']:
            print(f"- {doc.page_content[:100]}...") # Print first 100 chars of source

if __name__ == "__main__":
    chat_with_data()