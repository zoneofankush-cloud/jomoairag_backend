import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postgres.files import delete_file
from security.enc_dec import decrypt_value, encrypt_value
from postgres.connection import async_pool, cleanup_old_user_data

import urllib.parse
import ipaddress
import asyncio
import os
import asyncpg
from pydantic import BaseModel, field_validator, Field
from typing import Literal
from fastapi import HTTPException
from langchain_community.document_loaders import DirectoryLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import create_engine
from langchain_community.storage import SQLStore
from langchain_classic.storage import create_lc_store
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from fastapi import UploadFile, File, HTTPException, APIRouter, Depends, Form
from security.jwt import get_current_user
router = APIRouter()

# ==========================
# PYDANTIC (WITH DROPDOWN)
# ==========================
class EmbedderState(BaseModel):
    Parent_Chunk_Size: int = Field(..., ge=500, lt=5000)
    Parent_Chunk_Overlap: int = Field(..., ge=50, lt=400)   
    Child_Chunk_Size: int = Field(..., ge=100, lt=450)
    Child_Chunk_Overlap: int = Field(..., ge=10, lt=90)
    batch_size: int = Field(..., ge=10, lt=200)
    embedder: Literal[
        'gemini-embedding-001', 
        'gemini-embedding-2-preview', 
        'text-embedding-3-small', 
        'text-embedding-3-large'
    ]
    api_key: str

# ==========================
# EMBEDDER LOGIC
# ==========================
def user_embedder(embedder: str, api_key: str):
    if embedder == 'gemini-embedding-001':
        return GoogleGenerativeAIEmbeddings(model='gemini-embedding-001', api_key=api_key) # type: ignore
    if embedder == 'gemini-embedding-2-preview':
        return GoogleGenerativeAIEmbeddings(model='gemini-embedding-2-preview', api_key=api_key) # type: ignore
    if embedder == 'text-embedding-3-small':
        return OpenAIEmbeddings(model='text-embedding-3-small', api_key=api_key) # type: ignore
    if embedder == 'text-embedding-3-large':
        return OpenAIEmbeddings(model='text-embedding-3-large', api_key=api_key)             # type: ignore


async def veryfy_embedder(embedder_model: str, api_key: str):
    embedder = user_embedder(embedder_model, api_key)
    try:
        assert embedder is not None
        test = await embedder.aembed_query("hello")
        if test:
            return True
    except:
        pass
    return False

class VerifyEmbedderReq(BaseModel):
    embedder: str
    api_key: str

@router.post("/verify_embedder")
async def verify_embedder_api(req: VerifyEmbedderReq, user_id: str = Depends(get_current_user)):
    is_valid = await veryfy_embedder(req.embedder, req.api_key)
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid API Key or Embedder Model.")
    return {"message": "Embedder verified successfully."}

# ==========================
# CONFIG FUNCTIONS
# ==========================
@router.post("/set_custom_vector_config")
async def store_user_config(embedd: EmbedderState, user_id: str = Depends(get_current_user)):
    key = embedd.api_key
    model = embedd.embedder
    if not await veryfy_embedder(model, key):
        raise HTTPException(status_code=401, detail="Invalid Embedder API KEY")
    
    secret_key = await encrypt_value(key)
    # db_connection is no longer provided by user, store a dummy value to keep schema intact
    hashed_db = await encrypt_value("system_default_db")
    
    await async_pool.open()
    async with async_pool.connection() as conn:
        await conn.execute("""
            INSERT INTO user_config 
            (user_id, parent_chunk_size, parent_chunk_overlap, child_chunk_size, child_chunk_overlap, batch_size, db_connection, embedder, api_key)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                parent_chunk_size = EXCLUDED.parent_chunk_size,
                parent_chunk_overlap = EXCLUDED.parent_chunk_overlap,
                child_chunk_size = EXCLUDED.child_chunk_size,
                child_chunk_overlap = EXCLUDED.child_chunk_overlap,
                batch_size = EXCLUDED.batch_size,
                db_connection = EXCLUDED.db_connection,
                embedder = EXCLUDED.embedder,
                api_key = EXCLUDED.api_key
        """, (
            user_id, embedd.Parent_Chunk_Size, embedd.Parent_Chunk_Overlap, 
            embedd.Child_Chunk_Size, embedd.Child_Chunk_Overlap, embedd.batch_size, 
            hashed_db, model, secret_key
        ))
    return {"message": "The Config has been encrypted and saved"}

@router.put("/update_custom_vector_config")
async def change_user_config(embedd: EmbedderState, user_id: str = Depends(get_current_user)):
    key = embedd.api_key
    model = embedd.embedder
    if not await veryfy_embedder(model, key):
        raise HTTPException(status_code=401, detail="Invalid Embedder API KEY")
    
    secret_key = await encrypt_value(key)
    # Use dummy value for schema compatibility
    hashed_db = await encrypt_value("system_default_db")

    async with async_pool.connection() as conn:
        await conn.execute("""
            UPDATE user_config 
            SET parent_chunk_size = %s, parent_chunk_overlap = %s, child_chunk_size = %s, 
                child_chunk_overlap = %s, batch_size = %s, db_connection = %s, embedder = %s, api_key = %s
            WHERE user_id = %s
        """, (
            embedd.Parent_Chunk_Size, embedd.Parent_Chunk_Overlap, 
            embedd.Child_Chunk_Size, embedd.Child_Chunk_Overlap, embedd.batch_size, 
            hashed_db, model, secret_key, user_id
        ))
    return {"message": "The Config has been updated"}

@router.get("/get_custom_vector_config")
async def fetch_user_config(user_id: str = Depends(get_current_user)):
    await async_pool.open()
    async with async_pool.connection() as conn:
        cur = await conn.execute("SELECT * FROM user_config WHERE user_id = %s", (user_id,))
        res = await cur.fetchone()
    return res

# ==========================
# 1. MAIN EMBEDDER API
# ==========================
@router.post("/Start_User_Embedder")
async def User_Embedder(user_id: str = Depends(get_current_user)):
    config = await fetch_user_config(user_id)
    selected_embedder = config['embedder'] # type: ignore
    # Use the system default database instead of user provided db
    user_db_url = os.environ.get("MY_DATABASE_URL")
    if not user_db_url:
        raise HTTPException(status_code=500, detail="System Database URL not configured")
    
    # 2. FIX: config['key'] ki jagah config['api_key'] 
    key = config['api_key']  # type: ignore
    api_key = await decrypt_value(key)
    
    embedder = user_embedder(selected_embedder, api_key)

    # Clean up old vectors and history for this user
    await cleanup_old_user_data(user_id)

    path = f'uploads/{user_id}'
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No files uploaded. Please upload a PDF first.")
        
    loader = DirectoryLoader(
        path=path, glob="*.pdf", 
        loader_cls=PyMuPDF4LLMLoader,   # type: ignore
        loader_kwargs={"header": False, "footer": False})
    document = loader.load()
    
    for doc in document:
        doc.metadata["user_id"] = user_id

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=config['parent_chunk_size'], chunk_overlap=config['parent_chunk_overlap']) # type: ignore
    child_splitter  = RecursiveCharacterTextSplitter(chunk_size=config['child_chunk_size'], chunk_overlap=config['child_chunk_overlap']) # type: ignore

    db_engine = create_engine(
        user_db_url.replace("postgresql://", "postgresql+psycopg://"),
        pool_pre_ping=True,
        pool_recycle=1800
    )

    vector_store = PGVector(
        embeddings=embedder, # type: ignore
        collection_name=f"custom_{user_id}_{selected_embedder.replace('-', '_')}",
        connection=db_engine, 
        use_jsonb=True
    )

    raw_store = SQLStore(namespace=f"custom_parent_{user_id}_{selected_embedder.replace('-', '_')}", engine=db_engine)
    Store = create_lc_store(raw_store)

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=Store,  # type: ignore
        child_splitter=child_splitter,
        parent_splitter=parent_splitter   
    )
    
    try:
        batch_size = config['batch_size'] # type: ignore
        for i in range(0, len(document), batch_size):
            batch_chunks = document[i : i + batch_size]
            await asyncio.to_thread(retriever.add_documents, batch_chunks)
            
        return {"message": "✅ Custom Embedding Successfully Completed!"}

    except Exception as e:
        print(f"Error during embedding: {e}")
        raise HTTPException(status_code=500, detail='Embedding could not be completed, try again.')
        
    finally:
        await delete_file(user_id)
        
# ===========================
# 2. MAIN RETRIEVER API
# ===========================
@router.get("/User_Retriever")
async def User_Retriever(user_id: str = Depends(get_current_user)):
    config = await fetch_user_config(user_id)
    if not config:
        raise HTTPException(status_code=404, detail="Info not found")
    
    selected_embedder = config['embedder'] # type: ignore
    
    # 1. Use system default database
    user_db_url = os.environ.get("MY_DATABASE_URL")
    if not user_db_url:
        raise HTTPException(status_code=500, detail="System Database URL not configured")
    
    # 2. FIX: Yahan api_key decrypt karni hai
    key = config['api_key'] # type: ignore
    api_key = await decrypt_value(key)

    # 3. Yahan embedder load hoga
    embedder = user_embedder(selected_embedder, api_key)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=config['parent_chunk_size'], chunk_overlap=config['parent_chunk_overlap'], add_start_index=True) # type: ignore
    child_splitter  = RecursiveCharacterTextSplitter(chunk_size=config['child_chunk_size'], chunk_overlap=config['child_chunk_overlap'], add_start_index=True) # type: ignore

    async_db_engine = create_async_engine(
        user_db_url.replace("postgresql://", "postgresql+psycopg://"),
        pool_pre_ping=True,
        pool_recycle=1800
    )

    vector_store = PGVector(
        embeddings=embedder, # type: ignore
        collection_name=f"custom_{user_id}_{selected_embedder.replace('-', '_')}",
        connection=async_db_engine,
        use_jsonb=True,
        async_mode=True,
    )

    sync_db_engine = create_engine(
        user_db_url.replace("postgresql://", "postgresql+psycopg://"),
        pool_pre_ping=True,
        pool_recycle=1800
    )
    raw_store = SQLStore(namespace=f"custom_parent_{user_id}_{selected_embedder.replace('-', '_')}", engine=sync_db_engine)
    Store = create_lc_store(raw_store)

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=Store,  # type: ignore
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_type='mmr',  # type: ignore
        search_kwargs={'filter': {'user_id': user_id}}
    )
    return retriever