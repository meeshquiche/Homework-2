import pandas as pd
import os
from functions import agent_run, df_as_text

# --- TASK 1: DEFINE THE TOOL ---

def search_traffic_data(query):
    """Tool to search the CSV for incidents or road names."""
    
    # SAFETY CHECK: If Llama sends a dict instead of a string, extract the value
    if isinstance(query, dict):
        query = query.get('value', query.get('query', next(iter(query.values())) if query else ""))
    
    query = str(query)
    file_path = "traffic_data.csv" 
    
    if not os.path.exists(file_path):
        return f"Error: '{file_path}' not found."
        
    try:
        df = pd.read_csv(file_path)
        # Filter rows where the query matches any cell
        results = df[df.apply(lambda row: query.lower() in str(row).lower(), axis=1)]
        
        # Return as a list of dictionaries (standard for tool outputs)
        return results.to_dict(orient="records")
    except Exception as e:
        return f"Error processing CSV: {e}"

# Metadata for the Model
tool_traffic = {
    "type": "function",
    "function": {
        "name": "search_traffic_data",
        "description": "Search for traffic incidents or road names in the database.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "The specific road or incident to search for (e.g. 'I-95')"
                }
            },
            "required": ["query"]
        }
    }
}

traffic_fns = {"search_traffic_data": search_traffic_data}

# --- TASK 2: 2-AGENT WORKFLOW ---

def main():
    print("🚀 Starting Traffic Agent Workflow...")

    # Configuration
    MODEL_DISPATCHER = "llama3.2"
    MODEL_REPORTER = "smollm2:1.7b"

    # --- AGENT 1: THE DISPATCHER (DATA FETCHING) ---
    print("📡 Agent 1 (Dispatcher): Searching data...")
    
    # output="tools" returns a LIST of dictionaries
    result1 = agent_run(
        role="You are a Traffic Dispatcher. Use the search tool to find facts about road conditions.",
        task="Check for any reported issues on I-95.",
        tools=[tool_traffic],
        available_functions=traffic_fns,
        model=MODEL_DISPATCHER,
        output="tools"
    )

    # --- THE FIX: CONVERTING LIST TO DATAFRAME BEFORE PRINTING ---
    print("\n📊 Agent 1 Result (Data Fetch):")
    
    # Check if result1 is a list (returned by our tool)
    if isinstance(result1, list) and len(result1) > 0:
        # 1. Convert the list to a DataFrame
        df1 = pd.DataFrame(result1)
        
        print(f"✅ Retrieved {len(df1)} records:")
        
        # 2. NOW .head() works because df1 is a DataFrame!
        print(df1.head())
        
        # 3. Use our helper function to turn the table into text for Agent 2
        context_text = df_as_text(df1)
    else:
        print("⚠️ No data was retrieved or result is empty.")
        context_text = "No incidents reported."

    # --- AGENT 2: THE REPORTER (NARRATION) ---
    print("\n📢 Agent 2 (Reporter): Drafting alert...")
    
    final_alert = agent_run(
        role="You are a Radio Traffic Reporter. Write a short, urgent 2-sentence alert for drivers.",
        task=f"The dispatcher found this data:\n{context_text}\nWrite a public update.",
        model=MODEL_REPORTER
    )

    # --- FINAL OUTPUT ---
    print("\n" + "="*50)
    print("🚨 OFFICIAL TRAFFIC BROADCAST:")
    print("-" * 50)
    print(final_alert)
    print("="*50)

if __name__ == "__main__":
    main()