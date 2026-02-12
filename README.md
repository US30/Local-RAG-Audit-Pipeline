# 🕵️ Local RAG Audit Pipeline: Multi-Model Evaluation Arena

### M.Tech Project | Automated Compliance & Hallucination Detection
**Author:** Utkarsh  
**Institution:** NMIMS (Data Science & Business Analytics)  
**Status:** Prototype Complete (Phase 5)

---

## 📖 Overview
This project implements an **End-to-End Local RAG (Retrieval-Augmented Generation) System** designed to audit and benchmark Large Language Models (LLMs) on consumer hardware (RTX 2070 Super, 8GB VRAM). 

Unlike standard chatbots, this pipeline focuses on **Observability and Compliance**. It features a "Model Arena" where users can compare different LLM architectures (Llama vs. Gemma vs. Mistral) side-by-side, auditing their responses in real-time for **Faithfulness** (Hallucination Check) and **Relevancy** using a strictly local "Judge" model.

---

## 🚀 Key Features
* **⚔️ Multi-Model Arena:** Run two models (e.g., *Gemma 3* vs. *Mistral*) simultaneously to compare latency and accuracy.
* **⚖️ Automated Auditing:** Integrated **DeepEval** framework to score answers on Faithfulness & Relevancy using a local LLM Judge.
* **📡 Full Observability:** Real-time tracing of retrieval chunks and embedding latency using **Arize Phoenix** (OpenTelemetry).
* **⚡ "Thin-Stack" Architecture:** Uses **LanceDB** (Serverless Vector DB) to minimize RAM usage, leaving resources for the models.
* **🖼️ Multimodal Ready:** Support for ingesting text and images using **Llama 3.2 Vision** and joint embeddings.

---

## 🛠️ Tech Stack

| Component | Technology | Reasoning |
| :--- | :--- | :--- |
| **Orchestration** | **LangChain** | For chaining retrieval and generation logic. |
| **Vector Database** | **LanceDB** | Embedded, serverless DB. Zero setup, low RAM footprint. |
| **Inference Engine** | **Ollama** | Runs quantized GGUF models locally with hardware acceleration. |
| **Frontend** | **Streamlit** | Interactive dashboard for the "Battle Arena" UI. |
| **Evaluation** | **DeepEval** | Unit testing framework for LLM outputs (Faithfulness/Relevancy). |
| **Observability** | **Arize Phoenix** | Visualizing traces, spans, and retrieval quality. |
| **Embeddings** | **Sentence-Transformers** | `all-MiniLM-L6-v2` (Fast, local CPU execution). |

---

## 🤖 The Model Zoo (Benchmark Targets)

We evaluate models across different sizes to analyze the **Performance vs. Resource** trade-off on an 8GB VRAM GPU.

| Model | Params | Role | Performance Note |
| :--- | :--- | :--- | :--- |
| **Gemma 3** | 4B | **Challenger** | ⚡ **Fastest**. Fits entirely in VRAM. Excellent efficiency. |
| **Llama 3.1** | 8B | **Benchmark** | ⚖️ **Balanced**. The industry standard for local reasoning. |
| **Mistral** | 7B | **Reasoning** | 🧠 **Logic**. High accuracy but slower generation. |
| **Mistral Small** | 22B | **Stress Test** | 🐢 **Slow**. Exceeds VRAM; demonstrates offloading to system RAM. |

---

## ⚙️ Installation & Setup

### Prerequisites
* **OS:** Windows 10/11 (WSL2 Recommended) or Linux.
* **Hardware:** NVIDIA GPU (Minimum 6GB VRAM recommended).
* **Software:** Python 3.10+, Ollama.

### 1. Clone & Environment
```bash
git clone https://github.com/yourusername/Local-RAG-Audit.git
cd Local-RAG-Audit
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# Key packages: langchain, lancedb, streamlit, deepeval, arize-phoenix, sentence-transformers, openinference-instrumentation-langchain
```

### 3. Pull Local Models

Ensure Ollama is running, then pull the required models:
```bash
ollama pull llama3.1:8b
ollama pull gemma3:4b
ollama pull mistral
ollama pull llama3.2-vision
```

---

## 🏃 Usage Guide

### Step 1: Ingest Data (Build the Knowledge Base)

Place your PDF documents (e.g., *Attention Is All You Need*) in the `data/` folder.
```bash
python ingest.py
# This chunks the PDF and stores vectors in ./lancedb_data
```

### Step 2: Start Observability Server (Terminal 1)

Launch Arize Phoenix to track your RAG pipeline's internal logic.
```bash
python -m phoenix.server.main serve
# Dashboard will be available at http://localhost:6006
```

### Step 3: Launch the Arena (Terminal 2)

Start the Streamlit dashboard.
```bash
streamlit run app.py
```

---

## 📊 Methodology: The Audit Loop

1. **Retrieval:** The system searches LanceDB for the top k=3 chunks semantically related to the user query.

2. **Generation:** Two selected models (e.g., Llama 3 vs. Gemma 3) generate answers simultaneously in separate threads.

3. **Visual Audit:** The user inspects the raw retrieved context in the "Evidence" dropdown.

4. **Algorithmic Audit:**
   - User clicks "⚖️ Audit".
   - The system spins up a Judge Agent (Llama 3.1, Temperature=0).
   - DeepEval calculates:
     - **Faithfulness:** Is the answer derived strictly from the retrieved context?
     - **Relevancy:** Does the answer address the user's prompt?

---

## 📈 Hardware Performance Findings

Tested on **NVIDIA RTX 2070 Super (8GB VRAM) + 16GB System RAM**.

* **Sweet Spot:** Models under 8B parameters (Llama 3, Gemma 3) run 100% on GPU, achieving 40+ tokens/sec.
* **The Bottleneck:** Models >10B (Mistral Small 22B) spill over into system RAM, reducing speed to <5 tokens/sec and causing UI latency.
* **Conclusion:** For local enterprise deployment, **Quantized 8B models** offer the best balance of accuracy and latency.

---

## 🔮 Future Scope

* **Hybrid Search:** Implement keyword-based search (BM25) alongside vector search to improve retrieval of specific acronyms.
* **Agentic Workflow:** Upgrade from a RAG chain to a LangGraph Agent that can search the web if local documents are insufficient.
* **Fine-Tuning:** Create a LoRA adapter specifically for "Audit Style" responses to reduce the need for prompt engineering.

---

## 📜 License

MIT License. Created for Academic Purposes at NMIMS.