import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

# GLOBAL CONFIG
DB_URI = "./lancedb_data"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1:8b" 

def chat_with_data_setup():
    """Returns the QA Chain object for use in other scripts"""
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = lancedb.connect(DB_URI)
    vector_store = LanceDB(
        connection=db,
        embedding=embedding_model,
        table_name="rag_test"
    )
    llm = OllamaLLM(model=LLM_MODEL)
    
    template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 12}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain

# This function keeps the interactive chat working
def interactive_chat():
    qa_chain = chat_with_data_setup()
    print("✅ Pipeline Ready! (Type 'exit' to quit)")
    while True:
        query = input("\n❓ Ask a question: ")
        if query.lower() in ["exit", "quit"]: break
        result = qa_chain.invoke({"query": query})
        print(f"\n💡 Answer: {result['result']}")

if __name__ == "__main__":
    interactive_chat()