import requests
import json
import pandas as pd

# Configuration
DEFAULT_MODEL = "llama3"
PORT = 11434
OLLAMA_HOST = f"http://localhost:{PORT}"
CHAT_URL = f"{OLLAMA_HOST}/api/chat"

# 1. AGENT FUNCTION ###################################

def agent(messages, model=DEFAULT_MODEL, output="text", tools=None, available_functions=None):
    """
    Core agent function. Sends messages to Ollama and executes tool calls if requested.
    """
    # 1. Build the Request Body
    body = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    # Only add tools if the list is provided and not empty
    if tools and len(tools) > 0:
        body["tools"] = tools
    
    # 2. Send to Ollama
    response = requests.post(CHAT_URL, json=body)
    
    # 3. Enhanced Error Handling
    if response.status_code != 200:
        print(f"Ollama Error ({response.status_code}): {response.text}")
        response.raise_for_status()

    result = response.json()
    message = result.get("message", {})

    # 4. Handle Tool Calls (The Execution Loop)
    if "tool_calls" in message:
        tool_calls = message["tool_calls"]
        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            args = tool_call["function"]["arguments"]
            
            # Execute the actual Python function
            if available_functions and name in available_functions:
                print(f"Executing tool: {name}({args})")
                fn_result = available_functions[name](**args)
                tool_call["output"] = fn_result
        
        if output == "tools":
            return tool_calls
        
        # Return the output of the last tool call
        return tool_calls[-1].get("output")

    # If no tool was called, return the text content
    return message.get("content", "")

def agent_run(role, task, tools=None, available_functions=None, model=DEFAULT_MODEL, output="text"):
    """
    Wrapper updated to support the 'output' parameter (e.g., 'tools' or 'text').
    """
    messages = [
        {"role": "system", "content": role},
        {"role": "user", "content": task}
    ]
    # We added 'output=output' here so it passes the instruction to the main agent
    return agent(messages, model=model, tools=tools, available_functions=available_functions, output=output)


# 2. DATA CONVERSION FUNCTION ###################################

def df_as_text(df):
    """
    Convert a pandas DataFrame to a markdown table string.
    
    Parameters:
    -----------
    df : pandas.DataFrame or list of dicts
        The data to convert to a markdown table
    
    Returns:
    --------
    str
        A markdown-formatted table string
    """
    # If the input is a list of dicts (common from tool outputs), convert to DF first
    if isinstance(df, list):
        df = pd.DataFrame(df)
    
    # Use pandas to_markdown (requires 'pip install tabulate')
    try:
        return df.to_markdown(index=False)
    except Exception:
        # Fallback if tabulate isn't installed
        return str(df)