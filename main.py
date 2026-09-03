import os
import requests
from typing import TypedDict
from fastapi import FastAPI
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI()

# 1. State Clipboard Definitions
class AgentState(TypedDict):
    topic: str
    search_results: str
    summary: str

# 2. Initialize Cloud LLM 
# OLD CODE:
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# NEW CODE: Update to the current version
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.3)

# 3. Node A: Code-driven NewsAPI MCP Tool Simulation
def fetch_news_node(state: AgentState):
    topic = state["topic"]
    api_key = os.environ.get("NEWSAPI_KEY")
    
    # 1. ADD THIS LINE: Explicitly define a real browser identity header
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    url = f"https://newsapi.org/v2/everything?q={topic}&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
    
    # 2. UPDATE THIS LINE: Pass the headers variable into the request
    response = requests.get(url, headers=headers).json()
    articles = response.get("articles", [])
    
    formatted_text = ""
    for art in articles:
        title = art.get("title", "No Title")
        source = art.get("source", {}).get("name", "Unknown Source")
        description = art.get("description", "No Description Available")
        formatted_text += f"- [{source}] {title}: {description}\n"
        
    return {"search_results": formatted_text if formatted_text else "No recent articles found."}

# 4. Node B: Code-driven LLM Summarization
def summarize_news_node(state: AgentState):
    results = state["search_results"]
    prompt = f"Analyze and provide a concise, high-level developer digest of these news items:\n\n{results}"
    
    response = llm.invoke(prompt)
    return {"summary": str(response.content)}

# 5. Connect the Graph Elements
workflow = StateGraph(AgentState)
workflow.add_node("fetch_news", fetch_news_node)
workflow.add_node("summarize_news", summarize_news_node)

workflow.add_edge(START, "fetch_news")
workflow.add_edge("fetch_news", "summarize_news")
workflow.add_edge("summarize_news", END)

compiled_agent = workflow.compile()

# FastAPI Endpoint to invoke the pipeline
@app.get("/trigger-summary")
def trigger_agent(topic: str = "Artificial Intelligence"):
    initial_state = {"topic": topic, "search_results": "", "summary": ""}
    output = compiled_agent.invoke(initial_state)
    return {"summary": output["summary"]}

