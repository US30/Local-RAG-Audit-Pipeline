import json
import pandas as pd
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from eval_utils import OllamaEvaluator
from rag_pipeline import chat_with_data_setup # We will need to slightly modify rag_pipeline.py to import this

# 1. SETUP LOCAL JUDGE
# We use Llama 3 as the judge to grade Llama 3. 
# In a real research paper, you might use a larger model (e.g., Llama 70B) as the judge.
local_judge = OllamaEvaluator(model_name="llama3.1:8b")

# 2. DEFINE METRICS
faithfulness = FaithfulnessMetric(
    threshold=0.7, 
    model=local_judge, 
    include_reason=True
)
relevancy = AnswerRelevancyMetric(
    threshold=0.7, 
    model=local_judge, 
    include_reason=True
)

def main():
    print("🧪 Starting Automated Benchmarking...")
    
    # Load Test Data
    with open("test_dataset.json", "r") as f:
        test_data = json.load(f)

    # Initialize RAG Pipeline (We need a function that returns the chain, not runs the loop)
    # *Note: We need to modify rag_pipeline.py slightly to allow importing the chain*
    qa_chain = chat_with_data_setup() 

    test_cases = []
    
    print(f"📊 Evaluating {len(test_data)} test cases...")
    
    for entry in test_data:
        question = entry["input"]
        expected_answer = entry["actual_output"]
        
        print(f"   - Testing: {question[:40]}...")
        
        # 1. Get Actual RAG Result
        result = qa_chain.invoke({"query": question})
        actual_output = result["result"]
        
        # 2. Extract Retrieved Context
        retrieval_context = [doc.page_content for doc in result["source_documents"]]
        
        # 3. Create Test Case for DeepEval
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected_answer,
            retrieval_context=retrieval_context
        )
        test_cases.append(test_case)

    # 4. RUN EVALUATION
    print("👨‍⚖️  Judge is grading the answers (this takes time)...")
    results = evaluate(
        test_cases, 
        metrics=[faithfulness, relevancy],
    )
    
    # 5. SAVE RESULTS (Robust Version)
    print("💾 Saving results to CSV...")
    df_rows = []
    
    for r in results:
        # Check if 'r' is a standard object or a tuple (fixing the crash)
        if isinstance(r, tuple):
            # If it's a tuple, the TestResult object is likely the first item
            res_obj = r[0] if hasattr(r[0], 'input') else r[1]
        else:
            res_obj = r

        # Safely extract reason (sometimes it's missing)
        faith_reason = res_obj.metrics_data[0].reason if res_obj.metrics_data else "N/A"
        rel_reason = res_obj.metrics_data[1].reason if len(res_obj.metrics_data) > 1 else "N/A"

        df_rows.append({
            "Input": getattr(res_obj, 'input', 'N/A'),
            "Actual Output": getattr(res_obj, 'actual_output', 'N/A'),
            "Faithfulness Score": res_obj.metrics_data[0].score,
            "Faithfulness Reason": faith_reason,
            "Relevancy Score": res_obj.metrics_data[1].score,
            "Relevancy Reason": rel_reason
        })
    
    df = pd.DataFrame(df_rows)
    df.to_csv("benchmark_results.csv", index=False)
    print("✅ Evaluation Complete! Results saved to 'benchmark_results.csv'")
if __name__ == "__main__":
    main()