import pandas as pd
import os
from functions import agent_run

# --- TASK 1: DEFINE THE TOOL ---

def search_traffic_data(query):
    """Tool to search the CSV for incidents or road names."""
    file_path = "traffic_data.csv"
    
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
        
    try:
        df = pd.read_csv(file_path)
        # Filter rows where the query matches any cell
        results = df[df.apply(lambda row: query.lower() in str(row).lower(), axis=1)]
        
        if results.empty:
            return f"No traffic incidents found for '{query}'."
            
        return results.to_dict(orient="records")
    except Exception as e:
        return f"Error processing CSV: {e}"

# Metadata: Must be valid JSON Schema
tool_traffic = {
    "type": "function",
    "function": {
        "name": "search_traffic_data",
        "description": "Search a database for traffic incidents, road closures, or congestion by road name.",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "The keyword or road name to search for (e.g., 'I-95')"
                }
            }
        }
    }
}

# Map tool names to the actual functions
traffic_fns = {"search_traffic_data": search_traffic_data}

# --- TASK 2: 2-AGENT WORKFLOW ---

def main():
    print("Starting Traffic Agent Workflow...")

    # Agent 1: The Researcher (Uses Llama3 for tool reliability)
    print("📡 Agent 1 (Dispatcher): Searching data...")
    raw_results = agent_run(
        role="You are a Traffic Dispatcher. Use the search tool to find facts about road conditions.",
        task="Check for any reported issues on I-95.",
        tools=[tool_traffic],
        available_functions=traffic_fns,
        model="llama3.2"
    )

    print(f"Data Retrieved: {raw_results}")

    # Agent 2: The Narrator (Uses Smollm for quick formatting)
    print("Agent 2 (Reporter): Drafting alert...")
    final_alert = agent_run(
        role="You are a Radio Traffic Reporter. Write a short, urgent 2-sentence alert for drivers.",
        task=f"The dispatcher found this data: {raw_results}. Write a public update.",
        model="smollm2:1.7b"
    )

    print("\n" + "="*50)
    print(" OFFICIAL TRAFFIC BROADCAST:")
    print("-" * 50)
    print(final_alert)
    print("="*50)

if __name__ == "__main__":
    main()