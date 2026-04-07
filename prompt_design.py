import requests
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

# -----------------------------
# Load API key
# -----------------------------
load_dotenv()
client = OpenAI(api_key="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")

# -----------------------------
# Helper function to call LLM
# -----------------------------
def run_agent(system_prompt, user_input):
   response = client.chat.completions.create(
    model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.3
    )
   return response.choices[0].message.content


# -----------------------------
# Agent 1: Data Analyst
# -----------------------------
analyst_prompt = """
You are a data analyst specializing in traffic congestion.

Your task:
- Analyze the provided congestion data
- Identify the top 3 most congested locations
- Identify any trends (increasing/decreasing congestion)
- Highlight unusual patterns or spikes

Output format (STRICT):
- Bullet points only
- Concise and data-driven
- No explanations or opinions
"""


# -----------------------------
# Agent 2: Insight Generator
# -----------------------------
insight_prompt = """
You are an AI assistant that explains data insights in plain English.

Your task:
- Convert the analyst’s bullet points into a clear summary
- Explain what is happening in traffic conditions
- Keep it simple and easy to understand

Output format (STRICT):
- 1–2 short paragraphs
- No bullet points
- No technical jargon
"""


# -----------------------------
# Agent 3: Report Formatter
# -----------------------------
formatter_prompt = """
You are a report formatter.

Your task:
- Format the provided summary into a clean markdown report

Output format (STRICT):
- Title: "Traffic Congestion Report"
- Section: "Summary"
- Use markdown headers (#, ##)
"""


# -----------------------------
# Step 1: Fetch Data from API
# -----------------------------
def fetch_data():
    url = "http://127.0.0.1:8000/congestion/current"
    response = requests.get(url)
    return response.json()


# -----------------------------
# Step 2: Run Multi-Agent Flow
# -----------------------------
def run_workflow():
    print("\n--- Fetching Data ---")
    data = fetch_data()
    data_str = json.dumps(data, indent=2)

    print("\n--- Agent 1: Data Analyst ---")
    analyst_output = run_agent(analyst_prompt, data_str)
    print(analyst_output)

    print("\n--- Agent 2: Insight Generator ---")
    insight_output = run_agent(insight_prompt, analyst_output)
    print(insight_output)

    print("\n--- Agent 3: Report Formatter ---")
    final_report = run_agent(formatter_prompt, insight_output)
    print(final_report)

    return final_report


# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":
    run_workflow()