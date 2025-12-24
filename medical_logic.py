from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from functools import lru_cache
import pandas as pd
import numpy as np

# ===============================
# LOAD DATA
# ===============================
desc = pd.read_csv("symptom_Description.csv")

medical_knowledge = [
    f"Disease: {row['Disease']}. Description: {row['Description']}"
    for _, row in desc.iterrows()
]

# ===============================
# EMBEDDINGS
# ===============================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(medical_knowledge, show_progress_bar=False)

# ===============================
# GENAI MODEL (CACHED)
# ===============================
@lru_cache(maxsize=1)
def load_generator():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    )

generator = load_generator()

# ===============================
# RETRIEVAL (NO FAISS)
# ===============================
def retrieve_medical_context(user_input, k=3, min_similarity=0.4):
    query_embedding = embed_model.encode([user_input])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    valid_indices = [
        i for i, score in enumerate(similarities) if score >= min_similarity
    ]

    top_indices = sorted(
        valid_indices, key=lambda i: similarities[i], reverse=True
    )[:k]

    return [medical_knowledge[i] for i in top_indices]

# ===============================
# GENAI RESPONSE
# ===============================
def generate_medical_response(user_input):
    context = "\n".join(retrieve_medical_context(user_input))

    prompt = f"""
This is NOT medical advice.

The user reports:
{user_input}

Only discuss conditions commonly associated with these symptoms.
If uncertain, state that multiple causes are possible.

Medical reference:
{context}

Possible associated conditions (educational only):
"""

    response = generator(prompt, do_sample=True, temperature=0.3)
    return response[0]["generated_text"]

# ===============================
# SEVERITY SCORING
# ===============================
RED_FLAG_KEYWORDS = [
    "chest pain",
    "shortness of breath",
    "difficulty breathing",
    "confusion",
    "loss of consciousness",
    "persistent vomiting",
    "seizure",
]

def assess_severity(user_input):
    text = user_input.lower()

    for keyword in RED_FLAG_KEYWORDS:
        if keyword in text:
            return "HIGH", "🚨 High-risk symptoms detected. Seek immediate medical care."

    if len(user_input.split(",")) >= 3:
        return "MODERATE", "🟡 Multiple symptoms detected. Medical consultation advised."

    return "LOW", "🟢 Symptoms appear mild. Monitor closely."

# ===============================
# FORMAT RESPONSE
# ===============================
def format_medical_response(raw_text, severity, severity_msg):
    return f"""
### 🩺 Possible Associated Conditions (Educational Only)
{raw_text}

### 🚦 Severity Assessment
**Level:** {severity}  
{severity_msg}

### 📌 Medical Disclaimer
This information is for educational purposes only and is not medical advice.
Please consult a qualified healthcare professional.
"""
