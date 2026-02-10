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
    
   # 5. SAVE RESULTS (Object-Attribute Fix)
    print("💾 Saving results to CSV...")
    
    # 1. Extract the list from the object
    # The error showed us the data is inside 'test_results'
    if hasattr(results, 'test_results'):
        final_results = results.test_results
    elif isinstance(results, list):
        final_results = results
    else:
        print(f"❌ Error: Unexpected format {type(results)}")
        print(results)
        return

    # 2. Extract Data
    df_rows = []
    for res_obj in final_results:
        # Safely get metrics (handle missing/empty metrics)
        faith_score = 0
        faith_reason = "N/A"
        rel_score = 0
        rel_reason = "N/A"

        # Check if metrics_data exists and has items
        if hasattr(res_obj, 'metrics_data') and res_obj.metrics_data:
            # Try to find Faithfulness (usually first)
            if len(res_obj.metrics_data) > 0:
                faith_score = res_obj.metrics_data[0].score
                faith_reason = res_obj.metrics_data[0].reason
            # Try to find Relevancy (usually second)
            if len(res_obj.metrics_data) > 1:
                rel_score = res_obj.metrics_data[1].score
                rel_reason = res_obj.metrics_data[1].reason

        df_rows.append({
            "Input": getattr(res_obj, 'input', 'N/A'),
            "Actual Output": getattr(res_obj, 'actual_output', 'N/A'),
            "Faithfulness Score": faith_score,
            "Faithfulness Reason": faith_reason,
            "Relevancy Score": rel_score,
            "Relevancy Reason": rel_reason
        })
    
    # 3. Save
    if df_rows:
        df = pd.DataFrame(df_rows)
        df.to_csv("benchmark_results.csv", index=False)
        print(f"✅ Success! Saved {len(df_rows)} rows to 'benchmark_results.csv'")
    else:
        print("❌ Extracted list was empty. No rows to save.")

if __name__ == "__main__":
    main()