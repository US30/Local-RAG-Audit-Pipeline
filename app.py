import streamlit as st
import time
from rag_pipeline import chat_with_data_setup

# PAGE CONFIG
st.set_page_config(page_title="RAG Model Arena", page_icon="⚔️", layout="wide")

st.title("⚔️ Local RAG Arena: The Battle of Algorithms")
st.markdown("### M.Tech Project: Auditing Llama vs. Gemma vs. Mistral")

# SIDEBAR: SETUP
st.sidebar.header("⚙️ Arena Setup")

# 1. Model Selector
# Make sure you have pulled these models in Ollama!
available_models = [
    "llama3.1:8b", 
    "gemma3:4b",            
    "mistral", 
    "llama3.2-vision:latest"
]

selected_models = st.sidebar.multiselect(
    "Select 2 Models to Compare",
    available_models,
    default=["llama3.1:8b", "gemma3:4b"],
    max_selections=2
)

st.sidebar.markdown("---")
st.sidebar.success("📡 **Observability:** [Open Arize Phoenix](http://localhost:6006)")
st.sidebar.caption("View traces to see retrieval latency and embedding visualization.")

# MAIN CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# INPUT HANDLING
if prompt := st.chat_input("Enter a technical question for the audit..."):
    # Show User Query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # CHECK SELECTION
    if len(selected_models) < 1:
        st.error("Please select at least 1 model from the sidebar.")
    else:
        # PREPARE COLUMNS FOR BATTLE
        cols = st.columns(len(selected_models))
        
        # RUN MODELS IN PARALLEL (Conceptually)
        for i, model_name in enumerate(selected_models):
            with cols[i]:
                st.subheader(f"🤖 {model_name}")
                status = st.empty()
                status.info("Thinking...")
                
                try:
                    start_ts = time.time()
                    
                    # 1. Initialize Specific Model Chain
                    chain = chat_with_data_setup(model_name=model_name)
                    
                    # 2. Run Inference
                    response = chain.invoke({"query": prompt})
                    
                    end_ts = time.time()
                    duration = end_ts - start_ts
                    
                    # 3. Display Result
                    status.success(f"⏱️ {duration:.2f}s")
                    st.markdown(response["result"])
                    
                    # 4. Show Evidence (Audit Trail)
                    with st.expander("📚 Retrieved Context"):
                        for doc in response["source_documents"][:2]:
                            st.caption(f"...{doc.page_content[:200]}...")
                            st.divider()
                            
                except Exception as e:
                    status.error("Failed")
                    st.error(f"Error: {e}")
                    st.caption("Did you run `ollama pull` for this model?")