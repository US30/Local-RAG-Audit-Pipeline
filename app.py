import streamlit as st
import pandas as pd
import os
from rag_pipeline import chat_with_data_setup

# PAGE CONFIG
st.set_page_config(page_title="RAG Audit Pipeline", page_icon="🕵️", layout="wide")

st.title("🕵️ Local RAG Audit Pipeline")
st.markdown("### M.Tech Project: Automated Compliance & Hallucination Detection")

# SIDEBAR: Metric History
st.sidebar.header("📊 Audit History")
if os.path.exists("benchmark_results.csv"):
    df = pd.read_csv("benchmark_results.csv")
    
    # Calculate pass rates
    pass_rate = len(df[df["Faithfulness Score"] > 0.7]) / len(df) * 100
    
    st.sidebar.metric("Latest Pass Rate", f"{pass_rate:.1f}%")
    st.sidebar.metric("Faithfulness", f"{df['Faithfulness Score'].mean():.2f}")
    st.sidebar.metric("Relevancy", f"{df['Relevancy Score'].mean():.2f}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Recent Failures")
    # Show failed rows
    failed = df[df["Faithfulness Score"] < 0.7]
    if not failed.empty:
        for idx, row in failed.iterrows():
            st.sidebar.error(f"❌ {row['Input'][:30]}...")
            st.sidebar.caption(f"Reason: {row['Faithfulness Reason'][:100]}...")
    else:
        st.sidebar.success("No Recent Failures!")
else:
    st.sidebar.warning("No benchmark data found. Run eval.py first.")

# MAIN CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your document..."):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking & Auditing..."):
            try:
                # Initialize Pipeline
                qa_chain = chat_with_data_setup()
                
                # Get Answer + Source Docs
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                sources = response["source_documents"]
                
                # Display Answer
                st.markdown(answer)
                
                # Display "Audit Evidence" (Sources)
                with st.expander("🔍 View Retrieved Context (Audit Trail)"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:**")
                        st.caption(doc.page_content)
                        st.divider()

                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")