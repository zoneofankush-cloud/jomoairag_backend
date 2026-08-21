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
import asyncio, ipaddress, socket, trafilatura
from urllib.parse import urlparse
from langchain_core.tools import tool
from langchain_community.utilities import SearxSearchWrapper
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

@tool
async def search_engine(query: str, n: int):
    """Search the web, returns top N results (title, url, snippet).
    Use first for any current/factual info. Snippet is usually enough —
    only use fetch_webpage after this if snippet lacks needed detail.
    """
    try:
        searx = SearxSearchWrapper(searx_host="http://searxng:8080")
        result = await searx.aresults(query=query, num_results=n)
        return result
    except Exception as e:
        return {"error": f"Search failed: {e}"}

# Blacklist for Internal and Cloud Metadata IPs
BLOCKED_HOSTS = {"localhost", "127.0.0.1", "169.254.169.254"}

def is_safe_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False

    hostname = parsed.hostname
    if not hostname or hostname in BLOCKED_HOSTS:
        return False

    try:
        ip = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip)
        # Blocks internal network, loopback, and Oracle cloud metadata IPs
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
            return False
    except socket.gaierror:
        return False

    return True


def _extract_sync(url: str) -> str:
    if not is_safe_url(url):
        return "ERROR: URL is blocked for security reasons (SSRF Protection)."
        
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""
    text = trafilatura.extract(downloaded)
    return text or ""

@tool
async def fetch_webpage(url: str) -> str:
    """Fetch full text of one webpage. Use only after search_engine,
    and only for 1-2 most relevant URLs when snippets weren't enough.
    URL must come from search_engine results.
    """
    try:
        content = await asyncio.to_thread(_extract_sync, url)
        if not content:
            return "Could not extract readable content from this page."
            
        # Isolate content to prevent Prompt Injection
        return f"<fetched_content>\n{content}\n</fetched_content>"
    except Exception as e:
        return f"Error fetching page: {e}"


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
