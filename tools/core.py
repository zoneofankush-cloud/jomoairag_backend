#=================================================================
#====================  ENV VARIABLE SETUP  =======================
from dotenv import load_dotenv
import os
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


#=================================================================
#=======================  Module Imports =========================
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch
from langchain.tools import tool

from pydantic import BaseModel, Field
from fastapi import HTTPException

#=================================================================
#=======================  SEARCH TOOL SETUP ======================
@tool
async def search_tool(n: int, query: str):
    """Search the context asked by user if needed"""
    try:
        tavily_search = TavilySearch(max_results=n, topic="general", tavily_api_key=TAVILY_API_KEY)
        return await tavily_search.ainvoke(query)
    except Exception:
        try:
            return await DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=n)).ainvoke(query)
        except Exception:
            return "No results found"


#=================================================================
#=======================  HISTORY TOOL SETUP =====================
class HistoryInput(BaseModel):
    limit: int = Field(default=10, description="Number of History user required (e.g. 10, 50)")

@tool(args_schema=HistoryInput)
async def demand_history(limit: int, config: RunnableConfig):
    """Guess the number as per user input and Fetches the past chat history of the user if they ask about previous messages."""
    
    # CORRECT LOCAL IMPORTS:
    from backend.main import workflow, check_ownership  # workflow main.py me global hai!
    from backend.graph import get_text
    
    thread_id = config['configurable'].get("thread_id") # type: ignore
    user_id = config['configurable'].get("user_id") # type: ignore
    
    if not await check_ownership(thread_id, user_id):
        raise HTTPException(status_code=404, detail="Thread not found")
        
    config_ = RunnableConfig(configurable={"thread_id": thread_id, "user_id": user_id})
    
    state_history = [state async for state in workflow.aget_state_history(config=config_)]
    
    if not state_history:
        return "No previous history found."
        
    messages = state_history[0].values.get("messages", [])
    
    history = []
    for message in messages[-limit:]:
        if message.type in {"human", "ai"} and get_text(message.content):
            history.append(f"{message.type}: {get_text(message.content)}")
            
    return "\n".join(history)

#=================================================================
from langchain_core.runnables import RunnableConfig

class TimeInput(BaseModel):
    timezone: str = Field(default="", description="Optional timezone string (e.g. 'America/New_York', 'UTC'). Leave empty for user's local time.")

@tool(args_schema=TimeInput)
async def time_tool(timezone: str, config: RunnableConfig):
    """ use this tool to get current time or time in a specific timezone like UTC and location of user"""
    
    # Hamesha config se location aur default timezone nikal lo
    user_location = config.get('configurable', {}).get('location', 'Unknown')
    user_timezone = config.get('configurable', {}).get('timezone', 'UTC')
    
    # Agar AI ne koi specific timezone nahi manga, toh user ka default timezone use karo
    if not timezone:
        timezone = user_timezone
        
    try:
        from zoneinfo import ZoneInfo
        import datetime
        local_time = datetime.datetime.now(ZoneInfo(timezone)).strftime("%A, %Y-%m-%d %H:%M:%S %Z")
    except Exception:
        import time
        local_time = str(time.localtime())
        
    return f"Time for {timezone} timezone is: {local_time}. (User is currently in {user_location})"
