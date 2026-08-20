import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field
from typing import Literal
from postgres.connection import async_pool

def get_text(content):
    if isinstance(content, list):
        return " ".join(x.get("text", "") for x in content if isinstance(x, dict))
    else:
        return content

async def file_info(user_id:str):
    await async_pool.open()
    async with async_pool.connection() as conn: 
        cur = await conn.execute("SELECT * FROM files_info WHERE user_id = %s", (user_id,),)
        res = await cur.fetchone()
        if res is not None:
            PDF = res['pdf_name']
            Subject = res['subject']
            return PDF, Subject
        else:
            return " ", " "

class WhichVector(BaseModel):
    vectorsearch: Literal['System Vector Embedding', 'Custom Vector Embedding']

class schema_router(BaseModel):
    smart_router: Literal['deep_search', 'general_query', 'dont_know'] = Field(
        description="Whether the query needs deep_search, general_query or dont_know"
    )

async def DecissionNode(retriever, query: str, user_id: str, model, history: str):
    # FIX: model directly pass ho raha hai, 'config' variable use karne ki zaroorat nahi
    structured_model = model.with_structured_output(schema_router)
    
    import asyncio
    doc = await asyncio.to_thread(retriever.invoke, query)
    doc = "\n\n".join(x.page_content for x in doc)
    
    PDF, Subject = await file_info(user_id)
    
    # FIX: history bhi bahar se pass ho rahi hai, 'state' variable use karne ki zaroorat nahi
    prompt = f""" 
    You are {Subject} expert to judge and answer the following:\n
    you are being provided following History of chat, user's new query and retrived data from vecotor search
    (The data is extracted is from {PDF}) as per user question:\n
    User Question:  {query}\n
    Chat History:   {history}\n
    Retrieved Data: {doc}\n
    Judge how to answer this:
    - deep_search:   If user's query and retrieved data are closely related mark it as deep_search\n
    - general_query: if you really think the question do not need such extra vector searchd data and you can answer this using
                    basic {Subject} knowledge without the retrieved data consider it general_query.\n
    - dont_know:     If the question do not belongs to the relavant vector search 
                    and is blank and completely irrelevant to {PDF}reply dont_know.
    """
    smart_router = (await structured_model.ainvoke(prompt)).smart_router

    return smart_router, doc


async def deep_search_node(query: str, doc: str, history: str, user_id:str):
    PDF, Subject = await file_info(user_id)
    system = f'You are a {Subject} expert for {PDF}'
    human  = f"""as per the following user query, chat history retrieved from database and retrived data from vector search database, answer the question properly: \n
    If query is Blank or Irrelevant say kindly ask {Subject} Realated Questions only. \n
    Answer the question properly: \n
    query: {query} \n
    Vecotr Serched Data: {doc}\n
    chat history: {history}"""
    return system, human

async def general_query_node(query: str, history: str, user_id:str):
    _, Subject = await file_info(user_id)
    system = f'You are a {Subject} expert and reply only for {Subject} question, otherwise you say you do not know'
    human = f"""as per the following user query and chat history retrieved from database, answer the question properly: \n
    If query is Blank or Irrelevant say kindly ask {Subject} Realated Questions only. \n
    Answer the question properly: \n
    query: {query} \n
    chat history: {history}"""
            
    return system, human
    
async def dont_know_node(user_id:str):
    _, Subject = await file_info(user_id)
    return f"Kindly ask {Subject} Realated Questions only."