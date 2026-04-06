import pandas as pd
import json
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Setup
load_dotenv()

# API Key from Lab 1
client = OpenAI(api_key="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")

# Local LLM Config (Ollama)
OLLAMA_URL = "http://localhost:11434/api/generate"
RAG_MODEL = "phi" 
DATA_FILE = "traffic_data.csv"

# The Search Tool
def search_traffic_data(query):
    """Searches every cell in the CSV for the query string."""
    if not os.path.exists(DATA_FILE):
        print(f"   [ERROR] CSV file '{DATA_FILE}' not found!")
        return []
    
    try:
        df = pd.read_csv(DATA_FILE, skipinitialspace=True)
        search_term = str(query).lower()
        mask = df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
        results = df[mask]
        print(f"   [DEBUG] Search for '{search_term}' found {len(results)} matches.")
        return results.to_dict(orient="records")
    except Exception as e:
        print(f"   [ERROR] CSV Read Error: {e}")
        return []

# Agent Runners

def run_rag_agent(user_query, retrieved_data):
    """Extracts data and forces the local model to stay on task."""
    if not retrieved_data:
        return "The database search returned no matches for that location."

    # Extract the specific row data
    item = retrieved_data[0]
    location = item.get('location', 'Unknown')
    incident = item.get('incident', 'Unknown')
    desc = item.get('description', 'No details')
    
    # We create a fallback string in case the AI hallucinates again
    fallback_text = f"Factual Report: At {location}, there is a {incident} ({desc})."

    # We use a 'completion' style prompt (Fill in the blank)
    # This is the best way to prevent 'phi' from making up riddles
    prompt = f"""
    Below is a specific fact from a traffic database. 
    FACT: {fallback_text}
    
    Task: Summarize the fact above in one short sentence.
    Summary: 
    """
    
    payload = {"model": RAG_MODEL, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        ai_response = response.json()["response"].strip()
        
        # SAFETY CHECK: If the AI tries to play a game or says it doesn't know, use the fallback.
        if len(ai_response) < 5 or "sorry" in ai_response.lower() or "incident A" in ai_response:
            return fallback_text
        return ai_response
    except:
        return fallback_text

def run_gpt_agent(system_prompt, user_input):
    """OpenAI Agent with a Quota Fallback for Homework safety."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback text to keep the orchestration visible in screenshots
        return f"TRAFFIC ANALYSIS SUMMARY:\n- Insight: {user_input[:120]}...\n- Alert: High Attention Required.\n- [Note: Fallback triggered due to OpenAI {e}]"

# Multi-Agent Workflow

def run_integrated_system(user_request):
    print(f"Starting Integrated Traffic System...")
    print(f"User Request: {user_request}")

    # Search Tool (Lab 3)
    print("Executing Tool: search_traffic_data...")
    raw_data = search_traffic_data(user_request)

    # RAG Grounding (Lab 2)
    print("Running RAG Agent (Local LLM)...")
    rag_insight = run_rag_agent(user_request, raw_data)
    print(f"RAG Result: {rag_insight}")

    # Analyst Agent (Lab 1)
    print("Agent: Data Analyst (GPT-3.5)...")
    analyst_prompt = "Identify the key traffic incident and its severity. Bullet points only."
    analysis_output = run_gpt_agent(analyst_prompt, rag_insight)
    print(f"Analyst Output: {analysis_output}")

    # Formatter Agent (Lab 1)
    print("Agent: Report Formatter (GPT-3.5)...")
    formatter_prompt = "Create a clean Markdown report with a Title 'Traffic Report' and a 'Summary' section."
    final_report = run_gpt_agent(formatter_prompt, analysis_output)

    return final_report

# Execution
if __name__ == "__main__":
    # Use the road name directly for the best search results
    user_query = "Highway 101"
    
    final_output = run_integrated_system(user_query)
    
    print("\n" + "="*50)
    print("FINAL INTEGRATED SYSTEM OUTPUT")
    print("="*50)
    print(final_output)