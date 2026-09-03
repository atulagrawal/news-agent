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
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# 3. Node A: Code-driven NewsAPI MCP Tool Simulation
def fetch_news_node(state: AgentState):
    topic = state["topic"]
    
    # Read the token from Render's cloud environment variables
    api_key = os.environ.get("NEWSAPI_KEY")
    
    # NewsAPI REST Endpoint mimicking the MCP 'searchArticles' schema
    url = f"https://newsapi.org{topic}&sortBy=publishedAt&pageSize=5&apiKey={api_key}"
    
    response = requests.get(url).json()
    articles = response.get("articles", [])
    
    # Process structured fields cleanly into a state string
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

