import pandas as pd
import requests
import json

# ------------------------------
# Config
# ------------------------------
MODEL = "phi"
OLLAMA_URL = "http://localhost:11434/api/generate"
DOCUMENT = "data/traffic_data.csv"

# ------------------------------
# Search Function
# ------------------------------
def search_traffic(query, file_path):
    df = pd.read_csv(file_path)

    results = df[
        df.apply(lambda row: query.lower() in str(row).lower(), axis=1)
    ]

    return results.to_dict(orient="records")

# ------------------------------
# LLM Call
# ------------------------------
def ask_llm(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    return response.json()["response"]

# ------------------------------
# RAG Workflow
# ------------------------------
def rag_query(user_query):
    results = search_traffic(user_query, DOCUMENT)

    context = json.dumps(results, indent=2)

    role = """
    You are a traffic analysis assistant.

    Your job:
    - Analyze the traffic data provided
    - Answer the question clearly in plain English
    - DO NOT write code
    - DO NOT explain programming
    - ONLY give a short, direct answer based on the data
    """

    prompt = f"""
    {role}

    Traffic Data:
    {context}

    Question:
    {user_query}

    Answer:
    """

    return ask_llm(prompt)
# ------------------------------
# Run Test
# ------------------------------
queries = [
    "Where is traffic the worst?",
    "Which roads have incidents?",
    "Which segment has the highest congestion?"
]
for q in queries:
    print("Query:", q)
    print("Answer:", rag_query(q))
    print("---")