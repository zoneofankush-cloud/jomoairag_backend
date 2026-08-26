from dotenv import load_dotenv
import os, asyncio
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
DBConnection= os.getenv("MY_DATABASE_URL")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from security.jwt import get_current_user
from postgres.files import delete_file
from postgres.connection import cleanup_old_user_data
from fastapi import Depends, APIRouter
from security.jwt import get_current_user
router = APIRouter()


from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.storage import SQLStore
from langchain_classic.storage import create_lc_store
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from fastapi import HTTPException
from sqlalchemy import create_engine

embedder_model = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-small-en-v1.5",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token
    )

db_engine = create_engine(
    DBConnection.replace("postgresql://", "postgresql+psycopg://"),
    pool_pre_ping=True,
    pool_recycle=1800
)

vector_store = PGVector(
        embeddings=embedder_model,
        collection_name="parent_child_collection",
        connection=db_engine,
        use_jsonb=True,
    )

raw_store = SQLStore(namespace="parent_docs", engine=db_engine)
Store = create_lc_store(raw_store)

# ==========================
# 1. MAIN EMBEDDER FUNCTION:
# ==========================
@router.post("/Start_System_Embedder")
async def My_Embedder(user_id: str = Depends(get_current_user)):
    # Clean up old vectors and history for this user
    await cleanup_old_user_data(user_id)

    folder = f'uploads/{user_id}'
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail="No files uploaded. Please upload a PDF first.")
        
    loader   = DirectoryLoader(
        path=folder, 
        glob="*.pdf", 
        loader_cls=PyMuPDF4LLMLoader,  # type: ignore
        loader_kwargs={"header": False, "footer": False}
        )
    document = loader.load()
    
    for doc in document:
        doc.metadata["user_id"] = user_id

    # Chunking 
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter  = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

    # Retriever Setup
    embedder = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=Store, # type: ignore
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    
    try:
        batch_size = 50
        for i in range(0, len(document), batch_size):
            batch_chunks = document[i : i + batch_size]
            await asyncio.to_thread(embedder.add_documents, batch_chunks)
        return {"message": "✅ Embedding Successfully Completed!"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during embedding: {e}")
        raise HTTPException(status_code=500, detail='Embedding could not be completed, try again.')
        
    finally:
        await delete_file(user_id)


# ===========================
# 2. MAIN RETRIEVER FUNCTION:
# ===========================
async def My_Retriever(user_id:str):
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, add_start_index=True)
    child_splitter  = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, add_start_index=True)

    # 5. Retriever Setup
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=Store, # type: ignore
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_type= 'mmr', # type: ignore
        search_kwargs={'filter': {'user_id': user_id}}        
    )
    return retriever