import os
from typing import TypedDict, Annotated
from fastapi import FastAPI
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

app = FastAPI()

# 1. Define the explicit Data State passing through the graph
class AgentState(TypedDict):
    topic: str
    search_results: str
    summary: str

# 2. Initialize Cloud Tools & LLM (Requires Environment Variables in Cloud)
search_tool = TavilySearchResults(max_results=5)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# 3. Node A: Code-driven Fetch Tool
def fetch_news_node(state: AgentState):
    topic = state["topic"]
    # Programmatically fire search request
    raw_results = search_tool.invoke({"query": f"latest news about {topic}"})
    
    # Process text arrays cleanly into state strings
    formatted_text = "\n".join([f"- {r['title']}: {r['content']}" for r in raw_results])
    return {"search_results": formatted_text}

# 4. Node B: Code-driven LLM Summarization
def summarize_news_node(state: AgentState):
    results = state["search_results"]
    prompt = f"Analyze and provide a concise, high-level developer digest of these news items:\n\n{results}"
    
    response = llm.invoke(prompt)
    return {"summary": str(response.content)}

# 5. Build the LangGraph Workflow Schema
workflow = StateGraph(AgentState)
workflow.add_node("fetch_news", fetch_news_node)
workflow.add_node("summarize_news", summarize_news_node)

# Chain the nodes explicitly
workflow.add_edge(START, "fetch_news")
workflow.add_edge("fetch_news", "summarize_news")
workflow.add_edge("summarize_news", END)

# Compile into a production execution application graph
compiled_agent = workflow.compile()

# FastAPI Web Endpoint to safely invoke the workflow remotely
@app.get("/trigger-summary")
def trigger_agent(topic: str = "Artificial Intelligence"):
    initial_state = {"topic": topic, "search_results": "", "summary": ""}
    output = compiled_agent.invoke(initial_state)
    return {"summary": output["summary"]}

