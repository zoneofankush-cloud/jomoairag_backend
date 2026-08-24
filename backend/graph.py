# graph.py
#=================================================================
#========================  PATH IMPORTS  =========================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sandbox import python_docker_sandbox
from tools.core import search_tool, demand_history, time_tool, search_engine, fetch_webpage
from tools.emails import Send_Email, fetch_emails
from tools.finance import stock_price
from tools.files import read_document,load_pdf_document,ipynb_file_reader,\
                        read_image, read_excel_file, read_html_file, read_xml_file
from rag.retriever import DecissionNode, deep_search_node, general_query_node, dont_know_node
from postgres.my_vector import My_Retriever
from postgres.user_vector import User_Retriever

#=================================================================
#=======================  Module Imports =========================
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from typing import List
import time

def get_text(content):
    if isinstance(content, list):
        return "".join(x.get("text", "") for x in content if isinstance(x, dict))
    else:
        return content

#=================================================================
#=======================  Persistent Nodes =======================
class UserFact(BaseModel):
    topic: str = Field(..., description="Single topic (e.g., 'Location', 'Career Goal', 'Tech Stack')")
    info: str = Field(..., description="Details about that specific topic")

class check_message(BaseModel):
    should_record: bool = Field(..., description="Return True if User has provided any New Info")
    extracted_facts: List[UserFact] = Field(default_factory=list, description="List of all distinct facts found in the message")

async def memory_state(state: MessagesState, store: BaseStore, config: RunnableConfig):
    selected_llm = config['configurable']['llm'] # type: ignore
    structured_model = selected_llm.with_structured_output(check_message)

    user_id = config['configurable']['user_id'] # type: ignore
    user_message = None
    for message in reversed(state["messages"]):
        if getattr(message, "type", None) in {"human", "user"}:
            user_message = get_text(message.content)
            break
    user_details = ('profile', user_id, 'details')
    user_info = await store.asearch(user_details)
    user_old_info = ''.join([f"\n- {x.key}: {x.value['data']}" for x in user_info])
    prompt = f"""user has given this information: {user_message}\n
    check is this info match with already save info:{user_old_info}\n,
    if yes return False otherwise return True
    if it is mix of match like old info partially match the new info 
    return True and give a consised info about user by excluding the user's old info in list.
    """
    ai_response = await structured_model.ainvoke(prompt)
    if ai_response.should_record == True:
        for fact in ai_response.extracted_facts:
            await store.aput(user_details, fact.topic, {'data': fact.info})
    return {}

tools = [search_tool, search_engine, fetch_webpage, demand_history, time_tool, stock_price, ipynb_file_reader, read_image, 
         read_excel_file, read_html_file, read_xml_file,
         python_docker_sandbox, fetch_emails, Send_Email, read_document, load_pdf_document]

tool_node = ToolNode(tools)


#=================================================================
#===================== Dynamic Tool Selection ====================
def select_active_tools(user_query: str):
    user_query = user_query.lower()
    
    active_tools = [search_tool, search_engine, fetch_webpage, demand_history, time_tool]
    
    if any(keyword in user_query for keyword in ["stock", "price", "market", "finance"]):
        active_tools.append(stock_price)
        
    if any(keyword in user_query for keyword in ["email", "mail", "inbox"]):
        active_tools.extend([fetch_emails, Send_Email])
        
    if any(keyword in user_query for keyword in ["file", "pdf", "image", "excel", "csv", "html", "xml", "read", "document"]):
        active_tools.extend([
            read_document, load_pdf_document, read_image, 
            read_excel_file, read_html_file, read_xml_file, ipynb_file_reader
        ])
        
    if any(keyword in user_query for keyword in ["code", "run", "python", "script", "sandbox"]):
        active_tools.append(python_docker_sandbox)
        
    return active_tools


#=================================================================
#=========================  Get Time =============================
def get_local_time(user_tz):
    try:
        from zoneinfo import ZoneInfo
        import datetime
        return datetime.datetime.now(ZoneInfo(user_tz)).strftime("%A, %Y-%m-%d %H:%M:%S %Z")
    except Exception:
        import time
        return str(time.localtime())
    
    
#=================================================================
#=========================  Chat State ===========================
async def chat_node(state: MessagesState, store: BaseStore, config: RunnableConfig):
    selected_llm = config['configurable']['llm'] # type: ignore
    user_id = config['configurable']['user_id'] # type: ignore
    
    if not state['messages']:
        return {'messages': []}
        
    user_query_msg = state['messages'][-1]
    user_query_text = get_text(user_query_msg.content)

    #=================================================================
    #=========================  Direct Response ======================
    user_query_lower = user_query_text.lower().strip()
    simple_greetings = ["hi", "hello", "hey", "thanks", "ok"]
    if user_query_lower in simple_greetings:
        try:
            response = await selected_llm.ainvoke(user_query_text)
            return {'messages': [response]}
        except Exception as e:
            return {'messages': [AIMessage(content=f"[ERROR] Could not get a response: {str(e)}")]}

    #=================================================================
    #========== Sliding Window (ONLY FOR PAST HISTORY) ===============
    MAX_MESSAGES = 10
    past_messages = state['messages'][:-1] 
    recent_history = past_messages[-MAX_MESSAGES:] if past_messages else []

    safe_messages = []
    for msg in recent_history:
        text_content = get_text(msg.content)
        if text_content and len(text_content) > 4000:
            truncated_text = text_content[:4000] + "\n\n...[Context truncated]"
            safe_messages.append(msg.model_copy(update={"content": truncated_text}))
        else:
            safe_messages.append(msg)
            
    #=================================================================  
    #==================== API CRASH PROTECTION =======================
    while len(safe_messages) > 0 and not isinstance(safe_messages[0], HumanMessage):
        safe_messages.pop(0)

    #=================================================================
    #================= FORMATTING HISTORY STRING =====================
    history_string = "\n".join(f"{m.type}: {get_text(m.content)}" for m in safe_messages)

    #=================================================================
    #=============================== Retriever =======================
    vectorsearch = config['configurable'].get('vectorsearch', 'System Vector Embedding')

    if vectorsearch == 'System Vector Embedding':
        retriever = await My_Retriever(user_id)
    elif vectorsearch == 'Custom Vector Embedding':
        retriever = await User_Retriever(user_id)
    else:
        retriever = await My_Retriever(user_id) 

    #=================================================================
    #==================== RAG PROCESSING ON CURRENT QUERY ============
    smart_router, doc = await DecissionNode(retriever, user_query_text, user_id, selected_llm, history_string)

    sys = "" 
    human = None 
      
    if smart_router == 'deep_search':
        sys, human = await deep_search_node(query=user_query_text, doc=doc, history=history_string, user_id=user_id)
    elif smart_router == 'general_query':
        sys, human = await general_query_node(query=user_query_text, history=history_string, user_id=user_id)
    else:
        # LLM KO BYPASS KARO (Short-Circuit Optimization)
        direct_answer = await dont_know_node(user_id)
        return {'messages': [AIMessage(content=direct_answer)]}
        
    #=================================================================
    #==================  Gather and Config User Info =================
    user_location = config['configurable']['location'] # type: ignore
    user_tz = config['configurable']['timezone'] # type: ignore
    current_time_str = get_local_time(user_tz)
    user_details = ('profile', user_id, 'details')
    user_info = await store.asearch(user_details)

    if user_info:
        info = ''.join([f"\n- {x.key}: {x.value['data']}" for x in user_info])
    else:
        info = ""
    
    sys_config = {"name": "jomoai", "version": 'v-01', "established": "15 August 2026", "trained by": "Ankush Gautam"}
    
    sys_msg = SystemMessage(content=f"{str(sys_config)}\nuser_info: \n{info} \n- current_location: {user_location}\n- current_time: {current_time_str}\n{sys}")
    human_msg = HumanMessage(content=human)
    
    #=================================================================
    #========================= FINAL CALL ============================
    active_tools = select_active_tools(user_query_text)
    llm_tool = selected_llm.bind_tools(active_tools)
    
    try:
        response = await llm_tool.ainvoke([sys_msg, human_msg], config)
        return {'messages': [response]}
    except Exception as e:
        return {'messages': [AIMessage(content=f"[ERROR] Could not get a response: {str(e)}")]}
    
#=================================================================
#=======================  GRAPH STATE ============================
async def graph_state(checkpoint, store):
    graph = StateGraph(MessagesState)
    graph.add_node('chat_node', chat_node)
    graph.add_node("store_node", memory_state) 
    graph.add_node('tools', tool_node)
    graph.add_edge(START, 'chat_node')
    graph.add_conditional_edges('chat_node', tools_condition,{
        "tools": "tools", END: "store_node"
        })
    graph.add_edge('tools', 'chat_node')
    graph.add_edge('store_node', END)
    workflow = graph.compile(checkpointer=checkpoint, store=store)
    return workflow

#=================================================================
#=========================  TITLE STATE ==========================
class PromptSummary(BaseModel):
        prompt: str = Field(..., description="Give a 3 to 5 words discription of prompt")

async def chat_heading(prompt, selected_llm):    
    structured_llm = selected_llm.with_structured_output(PromptSummary)
    prompt_template = PromptTemplate(
        template="Based on user following prompt summarise it in 3 to 5 words: {user_prompts}",
        input_variables=["user_prompts"]
    )
    chain = prompt_template | structured_llm
    result = await chain.ainvoke({"user_prompts": prompt})
    return result.prompt