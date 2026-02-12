import lancedb
from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# --- DEEPEVAL IMPORTS ---
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# 1. SETUP OBSERVABILITY
tracer_provider = register()
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

DB_URI = "./lancedb_data"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- CUSTOM LOCAL JUDGE (For DeepEval) ---
class LocalOllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3.1:8b"):
        self.model = OllamaLLM(model=model_name)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.model.invoke(prompt)

    def get_model_name(self):
        return "Ollama Judge"

def run_audit_metrics(query, actual_output, retrieval_context):
    """
    Runs DeepEval metrics using Llama 3 as the Judge.
    Returns: Dict with scores and reasons.
    """
    # Initialize the Judge (Llama 3 is best for judging)
    judge_llm = LocalOllamaJudge("llama3.1:8b")
    
    # Define Metrics
    faithfulness = FaithfulnessMetric(
        threshold=0.7, 
        model=judge_llm, 
        include_reason=True
    )
    relevancy = AnswerRelevancyMetric(
        threshold=0.7, 
        model=judge_llm, 
        include_reason=True
    )
    
    # Create Test Case
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    
    # Measure
    faithfulness.measure(test_case)
    relevancy.measure(test_case)
    
    return {
        "faithfulness_score": faithfulness.score,
        "faithfulness_reason": faithfulness.reason,
        "relevancy_score": relevancy.score,
        "relevancy_reason": relevancy.reason
    }

# --- EXISTING RAG SETUP ---
def chat_with_data_setup(model_name="gemma3:4b"):
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = lancedb.connect(DB_URI)
    
    vector_store = LanceDB(
        connection=db,
        embedding=embedding_model,
        table_name="rag_test" 
    )

    llm = OllamaLLM(model=model_name)

    template = """
    You are a precise technical auditor. Answer the question strictly based on the context below.
    If the answer is not in the context, say "I don't know".
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return qa_chain