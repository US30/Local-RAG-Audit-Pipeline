import lancedb
import json
import re
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

# --- ROBUST LOCAL JUDGE ---
class LocalOllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3.1:8b"):
        # We tell Ollama to prefer JSON mode
        self.model = OllamaLLM(model=model_name, format="json")

    def load_model(self):
        return self.model

    def _clean_output(self, text: str) -> str:
        """
        Fixes the 'Invalid JSON' error by finding the first '{' and last '}'
        and removing everything else.
        """
        try:
            # If it's already valid JSON, return it
            json.loads(text)
            return text
        except json.JSONDecodeError:
            # Otherwise, use Regex to find the JSON block
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return match.group(0)
            return text

    def generate(self, prompt: str) -> str:
        raw_output = self.model.invoke(prompt)
        return self._clean_output(raw_output)

    async def a_generate(self, prompt: str) -> str:
        raw_output = await self.model.ainvoke(prompt)
        return self._clean_output(raw_output)

    def get_model_name(self):
        return "Ollama Llama3 Judge"

def run_audit_metrics(query, actual_output, retrieval_context):
    """
    Runs DeepEval metrics using Llama 3 as the Judge.
    """
    judge_llm = LocalOllamaJudge("llama3.1:8b")
    
    # Define Metrics with stricter threshold
    faithfulness = FaithfulnessMetric(
        threshold=0.5,  # Lowered slightly for local models
        model=judge_llm, 
        include_reason=True
    )
    relevancy = AnswerRelevancyMetric(
        threshold=0.5, 
        model=judge_llm, 
        include_reason=True
    )
    
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    
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

    llm = OllamaLLM(model=model_name, temperature=0)

    template = """
    You are a precise technical auditor. 
    Task: Answer the question strictly based on the context below.
    
    Guidelines:
    1. Look for exact numbers, names, and quotes in the context.
    2. If the context is empty or irrelevant, strictly say "I don't know".
    3. Do not use outside knowledge.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return qa_chain