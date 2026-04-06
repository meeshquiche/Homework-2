# 02_function_calling.py
# Basic Function Calling Example (Updated with Multiple Tools)
# Tim Fraser / Updated for Lab Stage 2

import requests 
import json

# 0. SETUP ###################################

# Select model (Ensure this model supports tools)
MODEL = "smollm2:1.7b"

# Set the port where Ollama is running
PORT = 11434
OLLAMA_HOST = f"http://localhost:{PORT}"
CHAT_URL = f"{OLLAMA_HOST}/api/chat"

# 1. DEFINE FUNCTIONS TO BE USED AS TOOLS ###################################

def add_two_numbers(x, y):
    """Add two numbers together."""
    return x + y

def multiply_numbers(x, y):
    """Multiply two numbers together."""
    return x * y

# 2. DEFINE TOOL METADATA ###################################

tool_add_two_numbers = {
    "type": "function",
    "function": {
        "name": "add_two_numbers",
        "description": "Add two numbers",
        "parameters": {
            "type": "object",
            "required": ["x", "y"],
            "properties": {
                "x": {"type": "number", "description": "first number"},
                "y": {"type": "number", "description": "second number"}
            }
        }
    }
}

tool_multiply_numbers = {
    "type": "function",
    "function": {
        "name": "multiply_numbers",
        "description": "Multiply two numbers together",
        "parameters": {
            "type": "object",
            "required": ["x", "y"],
            "properties": {
                "x": {"type": "number", "description": "first factor"},
                "y": {"type": "number", "description": "second factor"}
            }
        }
    }
}

# 3. CREATE CHAT REQUEST WITH TOOLS ###################################

# Test it with a multiplication problem to see it pick the new tool
messages = [
    {"role": "user", "content": "What is 12 times 8?"}
]

# Build the request body with BOTH tools available
body = {
    "model": MODEL,
    "messages": messages,
    "tools": [tool_add_two_numbers, tool_multiply_numbers],
    "stream": False
}

print(f"📡 Sending request to {MODEL}...")
response = requests.post(CHAT_URL, json=body)
response.raise_for_status()
result = response.json()

# 4. EXECUTE THE TOOL CALL ###################################

if "tool_calls" in result.get("message", {}):
    tool_calls = result["message"]["tool_calls"]
    
    for tool_call in tool_calls:
        func_name = tool_call["function"]["name"]
        # Parse arguments from the string returned by the LLM
        func_args = tool_call["function"]["arguments"]
        
        # If args come back as a string, parse to dict
        if isinstance(func_args, str):
            func_args = json.loads(func_args)
        
        print(f"⚙️ LLM requested: {func_name}({func_args})")
        
        # Get the function from the global scope and execute
        func = globals().get(func_name)
        if func:
            output = func(**func_args)
            print(f"✅ Tool result: {output}")
        else:
            print(f"❌ Error: Function {func_name} not found.")
else:
    # If the LLM just answered with text instead of a tool call
    print("💬 LLM Response:", result["message"]["content"])