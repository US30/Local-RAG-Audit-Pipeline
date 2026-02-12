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
st.sidebar.info("💡 Tip: Click 'Run Live Audit' after generating answers to see Faithfulness scores.")

# MAIN CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter a technical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if len(selected_models) < 1:
        st.error("Please select at least 1 model.")
    else:
        cols = st.columns(len(selected_models))
        
        for i, model_name in enumerate(selected_models):
            with cols[i]:
                st.subheader(f"🤖 {model_name}")
                status = st.empty()
                
                try:
                    # 1. GENERATION PHASE
                    status.info("Generating...")
                    start_ts = time.time()
                    chain = chat_with_data_setup(model_name=model_name)
                    response = chain.invoke({"query": prompt})
                    duration = time.time() - start_ts
                    
                    answer_text = response["result"]
                    source_docs = [doc.page_content for doc in response["source_documents"]]
                    
                    status.success(f"⏱️ {duration:.2f}s")
                    st.markdown(f"**Answer:** {answer_text}")
                    
                    # 2. AUDIT PHASE (Button to prevent OOM)
                    audit_key = f"audit_{model_name}_{int(time.time())}"
                    if st.button(f"⚖️ Run Live Audit for {model_name}", key=audit_key):
                        with st.spinner("🕵️ Judge (Llama 3) is analyzing truthfulness..."):
                            # Run DeepEval
                            scores = run_audit_metrics(
                                query=prompt,
                                actual_output=answer_text,
                                retrieval_context=source_docs
                            )
                            
                            # Display Metrics
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric(
                                    label="Faithfulness", 
                                    value=f"{scores['faithfulness_score']:.2f}",
                                    delta="Pass" if scores['faithfulness_score'] > 0.7 else "-Fail",
                                    delta_color="normal"
                                )
                            with col_b:
                                st.metric(
                                    label="Relevancy", 
                                    value=f"{scores['relevancy_score']:.2f}",
                                    delta="Pass" if scores['relevancy_score'] > 0.7 else "-Fail",
                                    delta_color="normal"
                                )
                                
                            with st.expander("📉 View Audit Reason"):
                                st.caption(f"**Faithfulness Reason:** {scores['faithfulness_reason']}")
                                st.caption(f"**Relevancy Reason:** {scores['relevancy_reason']}")

                    # Context Dropdown
                    with st.expander("📚 Retrieved Context"):
                        for doc in source_docs:
                            st.caption(f"...{doc[:200]}...")
                            st.divider()

                except Exception as e:
                    st.error(f"Error: {e}")