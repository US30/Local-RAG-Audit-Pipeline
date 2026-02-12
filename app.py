import streamlit as st
import time
from rag_pipeline import chat_with_data_setup, run_audit_metrics

# PAGE CONFIG
st.set_page_config(page_title="RAG Model Arena", page_icon="⚔️", layout="wide")
st.title("⚔️ Local RAG Arena: The Battle of Algorithms")
st.markdown("### M.Tech Project: Real-Time Audit with DeepEval")

# SIDEBAR: SETUP
st.sidebar.header("⚙️ Arena Setup")
available_models = ["llama3.1:8b", "gemma3:4b", "mistral", "llama3.2-vision:latest"]
selected_models = st.sidebar.multiselect(
    "Select 2 Models to Compare",
    available_models,
    default=["llama3.1:8b", "mistral"],
    max_selections=2
)

# --- SESSION STATE SETUP ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_results" not in st.session_state:
    st.session_state.last_results = {} # Stores the RAG answers

# 1. DISPLAY CHAT HISTORY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. HANDLE INPUT & GENERATION
if prompt := st.chat_input("Enter a technical question..."):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if len(selected_models) < 1:
        st.error("Please select at least 1 model.")
    else:
        # Clear previous results for a new query
        st.session_state.last_results = {}
        
        cols = st.columns(len(selected_models))
        
        # Run Generation Loop
        for i, model_name in enumerate(selected_models):
            with cols[i]:
                st.subheader(f"🤖 {model_name}")
                status = st.empty()
                status.info("Generating...")
                
                try:
                    start_ts = time.time()
                    chain = chat_with_data_setup(model_name=model_name)
                    response = chain.invoke({"query": prompt})
                    duration = time.time() - start_ts
                    
                    answer_text = response["result"]
                    source_docs = [doc.page_content for doc in response["source_documents"]]
                    
                    status.success(f"⏱️ {duration:.2f}s")
                    st.markdown(f"**Answer:** {answer_text}")
                    
                    # SAVE TO STATE (Crucial Step!)
                    st.session_state.last_results[model_name] = {
                        "answer": answer_text,
                        "context": source_docs,
                        "query": prompt
                    }

                    # Add Assistant Message to history (Optional, keeps chat clean)
                    st.session_state.messages.append({"role": "assistant", "content": f"**{model_name}:** {answer_text}"})

                except Exception as e:
                    st.error(f"Error: {e}")

# 3. HANDLE AUDIT BUTTONS (Outside the input loop)
# This part runs on every refresh, checking if we have results to audit
if st.session_state.last_results:
    st.divider()
    st.subheader("🕵️ Audit Controls")
    
    cols = st.columns(len(selected_models))
    for i, model_name in enumerate(selected_models):
        if model_name in st.session_state.last_results:
            with cols[i]:
                result_data = st.session_state.last_results[model_name]
                
                # The Audit Button
                if st.button(f"⚖️ Audit {model_name}", key=f"btn_{model_name}"):
                    with st.spinner(f"Running DeepEval on {model_name}..."):
                        try:
                            scores = run_audit_metrics(
                                query=result_data["query"],
                                actual_output=result_data["answer"],
                                retrieval_context=result_data["context"]
                            )
                            
                            # Display Metrics
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric(
                                    label="Faithfulness", 
                                    value=f"{scores['faithfulness_score']:.2f}",
                                    delta="Pass" if scores['faithfulness_score'] > 0.7 else "-Fail"
                                )
                            with col_b:
                                st.metric(
                                    label="Relevancy", 
                                    value=f"{scores['relevancy_score']:.2f}",
                                    delta="Pass" if scores['relevancy_score'] > 0.7 else "-Fail"
                                )
                            
                            st.info(f"Reason: {scores['faithfulness_reason']}")
                            
                        except Exception as e:
                            st.error(f"Audit Failed: {e}")