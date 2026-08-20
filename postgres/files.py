import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from security.jwt import get_current_user
from postgres.connection import async_pool

from fastapi import UploadFile, File, HTTPException, APIRouter, Depends, Form
from typing import Literal
import uuid, os, shutil

router = APIRouter()

def get_folder_size(folder_path: str) -> int:
    total_size = 0
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            fp = os.path.join(folder_path, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size

@router.post("/upload_pdf")
async def upload_file(subject: str = Form(...), 
                      pdf_name: str = Form(...), 
                      vectorsearch: Literal['System Vector Embedding', 'Custom Vector Embedding'] = Form(...),
                      file: UploadFile = File(...), 
                      user_id = Depends(get_current_user)):
    
    if file.size > (1024 * 1024 * 15) or not file.filename.lower().endswith('.pdf'): # type: ignore
        raise HTTPException(status_code=400, detail='File size or type is not allowed')
    file_name = f'{uuid.uuid4().hex}.pdf' 

    folder = f'uploads/{user_id}'
    os.makedirs(folder, exist_ok=True) 

    path = f"{folder}/{file_name}"
    # Naya logic: Total Size Limit Check
    current_file_size = file.size or 0
    existing_folder_size = get_folder_size(folder)
    MAX_TOTAL_SIZE = 15 * 1024 * 1024  # 15 MB
    
    if (existing_folder_size + current_file_size) > MAX_TOTAL_SIZE:
        raise HTTPException(status_code=413, detail="Total upload size exceeds the 15MB limit. Please embed or delete existing files first.")

    content = await file.read() 
    with open(path, "wb") as f:
        f.write(content)

    await async_pool.open()
    async with async_pool.connection() as conn:
        await conn.execute("""
            INSERT INTO files_info (user_id, pdf_name, subject, vectorsearch) 
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id) 
            DO UPDATE SET 
                pdf_name = EXCLUDED.pdf_name, 
                subject = EXCLUDED.subject, 
                vectorsearch = EXCLUDED.vectorsearch
        """, (user_id, pdf_name, subject, vectorsearch))

    return {"message": "File uploaded safely!"}

@router.delete('/delete_pdf')
async def delete_file(user_id:str = Depends(get_current_user)):
    path = f'uploads/{user_id}'
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except:
        pass

@router.get('/check_embedding_status')
async def check_embedding_status(user_id: str = Depends(get_current_user)):
    try:
        await async_pool.open()
        async with async_pool.connection() as conn:
            cur = await conn.execute("SELECT pdf_name, subject FROM files_info WHERE user_id = %s", (user_id,))
            res = await cur.fetchone()
            if res:
                return {"is_embedded": True, "pdf_name": res['pdf_name'], "subject": res['subject']}
            return {"is_embedded": False}
    except Exception as e:
        return {"is_embedded": False, "error": str(e)}