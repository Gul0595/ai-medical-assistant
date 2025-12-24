# 🩺 AI Medical Knowledge Assistant (GenAI + RAG)

An **educational GenAI-powered medical assistant** that provides **safe, non-diagnostic explanations** for user-reported symptoms.  
The system uses a **Retrieval-Augmented Generation (RAG)** pipeline with **semantic similarity search**, **severity assessment**, and **red-flag detection**, all wrapped in an interactive **Streamlit web app**.

> ⚠️ This project is for **educational purposes only** and does **not provide medical advice or diagnosis**.

---

## 🚀 Key Features

- 🔍 **Retrieval-Augmented Generation (RAG)**
  - Retrieves relevant medical descriptions using sentence embeddings
  - Prevents hallucinations by grounding responses in reference data

- 🧠 **GenAI Medical Explanation**
  - Uses an open-source LLM (FLAN-T5)
  - Generates clear, educational explanations in simple language

- 🚦 **Severity & Red-Flag Scoring**
  - Rule-based triage system
  - Classifies symptoms as **LOW / MODERATE / HIGH**
  - Detects medical red flags (e.g., chest pain, breathing difficulty)

- 🛡️ **Medical Safety Design**
  - Conservative prompting
  - Explicit medical disclaimer
  - Avoids diagnosis or prescriptions

- 🌐 **Interactive Web App**
  - Built with Streamlit
  - User-friendly interface for symptom input
  - Real-time response generation

---

## 🧱 Tech Stack

- **Python 3.10**
- **Streamlit** – Web interface
- **Sentence-Transformers** – Text embeddings
- **Transformers (FLAN-T5)** – Language model
- **Scikit-learn** – Cosine similarity search
- **Pandas / NumPy** – Data handling

---

## 🏗️ System Architecture

