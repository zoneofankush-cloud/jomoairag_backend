import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jwt, os, datetime
from typing import Optional
from fastapi import Header, HTTPException
from postgres.connection import async_pool

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"

async def delete_jwt_record(user_id: str):
    async with async_pool.connection() as conn:
        await conn.execute("DELETE FROM jwt_blacklist WHERE user_id = %s", (user_id,))
        await conn.commit()

async def insert_jwt_record(user_id: str, token: str):
    async with async_pool.connection() as conn:
        await conn.execute("INSERT INTO jwt_blacklist (user_id, token, blacklist) VALUES (%s, %s, FALSE)", (user_id, token))
        await conn.commit()

async def verify_jwt_record(user_id: str, token: str) -> bool:
    async with async_pool.connection() as conn:
        cur = await conn.execute("SELECT token, blacklist FROM jwt_blacklist WHERE user_id = %s", (user_id,))
        row = await cur.fetchone()
        if not row:
            return False
        if row['blacklist']:
            return False
        if row['token'] != token:
            return False
        return True

async def get_jwt_token(user_id):
    try:
        payload = {"user_id": str(user_id), "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)}
        token = jwt.encode(payload, key = str(SECRET_KEY), algorithm=ALGORITHM)
        await delete_jwt_record(str(user_id))
        await insert_jwt_record(str(user_id), token)
        return token
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"SECRET KEY not defined properly: error: {e}")

async def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    if authorization is None or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(' ')[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]) # type: ignore
        user_id = payload['user_id']
        
        if not await verify_jwt_record(user_id, token):
            raise HTTPException(status_code=401, detail="Token has been revoked or logged out from another device.")
            
        async with async_pool.connection() as conn:
            cur = await conn.execute("SELECT 1 FROM users WHERE id = %s", (user_id,))
            if not await cur.fetchone():
                raise HTTPException(status_code=401, detail="User account no longer exists")
                
        return user_id
    except jwt.ExpiredSignatureError:       
            raise HTTPException(status_code=401, detail="Token expired, please log in again")
    except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

