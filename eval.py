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
    
   # 5. SAVE RESULTS (Deep Search Version)
    print("💾 Saving results to CSV...")
    df_rows = []

    for i, r in enumerate(results):
        target_obj = None
        
        # Strategy A: Is 'r' the object itself?
        if hasattr(r, 'input'):
            target_obj = r
        
        # Strategy B: Is it inside a tuple or list?
        elif isinstance(r, (list, tuple)):
            for sub_item in r:
                if hasattr(sub_item, 'input'):
                    target_obj = sub_item
                    break
        
        # Strategy C: If we still haven't found it, print the structure to debug
        if target_obj is None:
            print(f"❌ Result #{i}: Could not find TestResult object!")
            print(f"   Type: {type(r)}")
            print(f"   Content: {r}")
            continue

        # Extract Metrics
        # We use a helper to avoid index errors if metrics are missing
        faith_m = target_obj.metrics_data[0] if len(target_obj.metrics_data) > 0 else None
        rel_m = target_obj.metrics_data[1] if len(target_obj.metrics_data) > 1 else None

        df_rows.append({
            "Input": getattr(target_obj, 'input', 'N/A'),
            "Actual Output": getattr(target_obj, 'actual_output', 'N/A'),
            "Faithfulness Score": faith_m.score if faith_m else 0,
            "Faithfulness Reason": faith_m.reason if faith_m else "N/A",
            "Relevancy Score": rel_m.score if rel_m else 0,
            "Relevancy Reason": rel_m.reason if rel_m else "N/A"
        })
    
    if df_rows:
        df = pd.DataFrame(df_rows)
        df.to_csv("benchmark_results.csv", index=False)
        print(f"✅ Success! Saved {len(df_rows)} rows to 'benchmark_results.csv'")
    else:
        print("❌ No results were saved. Check the logs above.")


if __name__ == "__main__":
    main()