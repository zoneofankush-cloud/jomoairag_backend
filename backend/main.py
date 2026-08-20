#=================================================================
#====================  ENV VARIABLE SETUP  =======================
from dotenv import load_dotenv
import os
load_dotenv()

JWT_SECRET_KEY=os.getenv("JWT_SECRET_KEY")
PASS_KEY=os.getenv("PASS_KEY")
SENDER_EMAIL=os.getenv("SENDER_EMAIL")
ENDPOINT_URL=os.getenv("ENDPOINT_URL")
ACCESS_KEY_ID=os.getenv("ACCESS_KEY_ID")
SECRET_ACCESS_KEY=os.getenv("SECRET_ACCESS_KEY")
BUCKET_NAME=os.getenv("BUCKET_NAME")

#=================================================================
#========================  PATH IMPORTS  =========================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.graph import graph_state, get_text, chat_heading
from postgres.connection import memory_saver_state, memory_store_state, async_pool
from security.enc_dec import decrypt_value, encrypt_value
from tools.emails import get_gmail_auth_detail
from model.gemini import jomoai

from postgres.files import router as files_router
from postgres.user_vector import router as User_vector_router
from postgres.my_vector import router as my_vector_router

#=================================================================
#========================  Helper Main Func  =====================

from backend.user.data import get_user_info
from backend.user.user import email_exists, password_fetcher, fetch_id, check_ownership
from security.jwt import delete_jwt_record, insert_jwt_record, verify_jwt_record, get_jwt_token, get_current_user
from security.otp import generate_otp, build_email, create_row, update_row, delete_row, otp_status, time_remaining

#=================================================================
#=======================  Module Imports =========================
import uuid, jwt, datetime, bcrypt, httpx, shutil, re, aioboto3, asyncio, smtplib, numpy as np

from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException, FastAPI, Depends, Header, UploadFile, File, Response, Request
from fastapi.responses import StreamingResponse

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from pydantic import BaseModel, EmailStr, Field, field_validator, SecretStr
from typing import Optional
from enum import Enum

# (If you want advanced Semantic caching later, you can use RedisSemanticCache)

from datetime import date
import datetime as dt

from email.message import EmailMessage

#=================================================================
#========================  API Setup ============================
import redis.asyncio as redis

redis_client = None
    
    # ... baki aapka graph aur db ka connection code ...
def setup_database_schema():
    import os
    from langchain_community.storage import SQLStore
    DBConnection = os.getenv("DATA_BASE_URL")
    if DBConnection:
        raw_store = SQLStore(namespace="parent_docs", db_url=DBConnection)
        raw_store.create_schema()
        print("✅ Database schema created successfully!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global workflow, redis_client
    # Running sync DB schema creation in a separate thread so it doesn't block
    await asyncio.to_thread(setup_database_schema)
    
    redis_client = await redis.from_url("redis://redis:6379", decode_responses=True)
    await async_pool.open()
    checkpointer = await memory_saver_state()
    store = await memory_store_state()
    workflow = await graph_state(checkpointer, store)
    yield

app = FastAPI(
    lifespan=lifespan,
    #docs_url=None,
    #redoc_url=None,
    #openapi_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", "https://jomoai.pages.dev", "https://jomoairag.pages.dev"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://.*\.jomoai\.pages\.dev|https://.*\.jomoairag\.pages\.dev",
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Chat-Title"],
)

class PromptValue(BaseModel):
    prompt: str
    timezone: str = "UTC"



#=================================================================
#========================  RAG SETUP  ============================
app.include_router(files_router)
app.include_router(User_vector_router)
app.include_router(my_vector_router)
    
#=================================================================
#========================  JWT Module ============================
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"


async def chat_shape(user_id):
    async with async_pool.connection() as conn:
        cur = await conn.execute("""
SELECT COUNT(DISTINCT t.thread_id) as total_chats, ROUND(SUM(pg_column_size(c.*)) / (1024.0 * 1024.0), 2) AS size_in_mb
FROM threads t JOIN checkpoints c ON t.thread_id::text = c.thread_id WHERE t.user_id = %s""", (user_id,))
        row = await cur.fetchone()
        return (row['total_chats'] or 0), (row['size_in_mb'] or 0) # type: ignore

async def inject_title(title, thread_id, user_id):
    async with async_pool.connection() as conn:
        await conn.execute("UPDATE threads SET title = %s WHERE thread_id = %s AND user_id = %s", (title, thread_id, user_id))

async def _generate_title_bg(prompt, llm, thread_id, user_id):
    try:
        title = await chat_heading(prompt, llm)
        await inject_title(title, thread_id, user_id)
    except Exception:
        pass

async def update_daily_tokens(user_id: str, tokens: int):
    async with async_pool.connection() as conn:
        await conn.execute("""
            INSERT INTO daily_usage (user_id, tokens, date)
            VALUES (%s, %s, CURRENT_DATE)
            ON CONFLICT (user_id, date)
            DO UPDATE SET tokens = daily_usage.tokens + EXCLUDED.tokens
        """, (user_id, tokens))
        await conn.commit()

async def get_size(user_id):
    async with session.client(
                service_name="s3",
                endpoint_url=ENDPOINT_URL,
                aws_access_key_id=ACCESS_KEY_ID,
                aws_secret_access_key=SECRET_ACCESS_KEY,
                region_name="auto",
            ) as s3_client: # type: ignore
        data = await s3_client.list_objects_v2(Bucket = BUCKET_NAME, Prefix= f"upload/{user_id}")
        contents = data.get('Contents')
        if not contents:
            size = 0
        else:
            size = sum(x.get('Size', 0) for x in contents) / (1024 * 1024)
            
        if size <= 100:
            return True

#=================================================================
#========================  Threads Setups =========================    
@app.post("/threads")
async def create_thread(user_id=Depends(get_current_user)):
    # check if rag setup is complete before creating a new thread
    async with async_pool.connection() as conn:
        cur = await conn.execute("SELECT 1 FROM files_info WHERE user_id = %s", (user_id,))
        if not await cur.fetchone():
            raise HTTPException(status_code=403, detail="RAG setup incomplete. Please upload and embed a document first.")

    # check if the user has reached the maximum number of chats before creating a new thread 
    counts, size = await chat_shape(user_id)
    if counts > 15 or size > 50:
        return f"You have hit the size or Number of Chats Delete some chats to continue"
    else:
        thread_id = uuid.uuid4()
        async with async_pool.connection() as conn:
            await conn.execute("INSERT INTO threads (thread_id, user_id) VALUES (%s, %s)", (thread_id, user_id))
            await conn.commit() 
        return {"thread_id": str(thread_id)}

@app.get("/threads")
async def get_all_threads(user_id = Depends(get_current_user)):
    async with async_pool.connection() as conn:
        cur = await conn.execute("SELECT thread_id, title FROM threads WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
        threads = await cur.fetchall()
    return [{"id": str(t['thread_id']), "title": t['title'] or "New Conversation"} for t in threads]

# FIXED [Issue #2]: Added DELETE /threads endpoint to handle deleting chat threads safely
@app.delete("/threads")
async def delete_thread(thread_id: str, user_id = Depends(get_current_user)):
    async with async_pool.connection() as conn:
        cur = await conn.execute("DELETE FROM threads WHERE thread_id = %s AND user_id = %s RETURNING thread_id", (thread_id, user_id))
        deleted = await cur.fetchone()
        if not deleted:
            raise HTTPException(status_code=404, detail="Thread not found")
        await conn.commit()
    return {"status": "deleted", "thread_id": thread_id}

@app.post("/response")
async def response(prompt: PromptValue, thread_id: str, model_version: str, request: Request, user_id=Depends(get_current_user)):
    row = await check_ownership(thread_id, user_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    llm = await model_selection(model_version, user_id)
    
    # Get IP-based Location and Timezone
    user_loc, user_tz = await get_user_info(request, prompt.timezone)

    # Redis se get karna (Extremely fast, usually <1ms)
    vector_type = await redis_client.get(f"user:{user_id}:vectorsearch")
    if not vector_type:
        # DB se fetch karein (Agar Redis me nahi hai to)
        async with async_pool.connection() as conn:
            cur = await conn.execute("SELECT vectorsearch FROM files_info WHERE user_id = %s", (user_id,))
            res = await cur.fetchone()
        vector_type = res['vectorsearch'] if res else 'System Vector Embedding'       
        # Redis me save karein (e.g., 24 hours expiry ke sath)
        await redis_client.setex(f"user:{user_id}:vectorsearch", 86400, vector_type)

    
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id, "llm": llm, 
                               "timezone": user_tz, "location": user_loc, "vectorsearch": vector_type}}
    initial_state = {"messages": [{"role": "user", "content": prompt.prompt}]}

    #==========================================================================
    #===================== First Message Detection ============================
    # row['title'] None hoga jab thread naya hoga (first message)
    is_first_message = row['title'] is None

    async def stream():
        tokens = 0
        try:
            async for chunk, metadata in workflow.astream(initial_state, config=config, stream_mode="messages"): # type: ignore
                #==============================================================================
                #=========================== Proper Text Streaming ============================
                if metadata.get("langgraph_node") == "chat_node" and chunk.type in ("ai", "AIMessageChunk"): # type: ignore
                    if isinstance(chunk.content, str) and chunk.content: # type: ignore
                        yield chunk.content # type: ignore
                    elif isinstance(chunk.content, list): # type: ignore
                        text = get_text(chunk.content) # type: ignore
                        if text:
                            yield text

                #==========================================================================
                #=========================== Token Extractions ============================        
                if model_version == "jomoai":
                    if getattr(chunk, "usage_metadata", None):
                        tokens_used = chunk.usage_metadata.get('total_tokens', 0) # type: ignore
                        tokens += tokens_used
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            yield f"\n\n[SYSTEM_ERROR]: {str(e)}\n\nTRACEBACK:\n{tb_str}"
        finally:
            if tokens > 0:
                asyncio.create_task(update_daily_tokens(user_id, tokens))
        #==========================================================================
        #=========================== Title ========================================
            # Title generation — stream khatam hone ke BAAD background mein
            if is_first_message:
                asyncio.create_task(_generate_title_bg(prompt.prompt, llm, thread_id, user_id))

    return StreamingResponse(stream(), media_type="text/plain")

@app.get("/history")
async def history(thread_id: str, user_id=Depends(get_current_user)):
    if not await check_ownership(thread_id, user_id):
        raise HTTPException(
            status_code=404, detail="Thread not found")
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    state_history = [state async for state in workflow.aget_state_history(config=config)] # type: ignore
    if not state_history:
        return []
    messages = state_history[0].values.get("messages", [])
    return [{message.type: get_text(message.content)} for message in messages if message.type in {"human", "ai"} and get_text(message.content)]

#================================================================
#========================  User Auth ============================
class User(BaseModel):
    email: EmailStr = Field(..., description="Email address of the user", examples=["example@example.com"])
    password: str = Field(
        ...,
        min_length=12,
        max_length=25,
        examples=["Password1234#"],
        description="Password must be at least 12 characters long and contain at least one uppercase letter, " \
                    "one lowercase letter, one digit, and one special character."
    )
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?/' for c in v):
            raise ValueError('Password must contain at least one special character')
        return v

class New_Password(BaseModel):
    password: str = Field(
        ...,
        min_length=12,
        max_length=25,
        examples=["Password1234#"],
        description="Password must be at least 12 characters long and contain at least one uppercase letter, " \
                    "one lowercase letter, one digit, and one special character."
    )
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?/' for c in v):
            raise ValueError('Password must contain at least one special character')
        return v

@app.post("/register")
async def register_user(user: User):
    if not await otp_status(user.email):
        return {"status": "First verify your email before proceeding."}
    await delete_row(user.email)
    password = user.password
    hashed = bcrypt.hashpw(password.encode(), salt=bcrypt.gensalt()).decode()
    if await email_exists(user.email):
        raise HTTPException(status_code=401, detail="Email already exists")
    else:
        try:
            async with async_pool.connection() as conn:
                await conn.execute("INSERT INTO public.users (id, email, password) VALUES (%s, %s, %s)", (str(uuid.uuid4()),user.email, hashed))
                await conn.commit()
            return {"status": "registered"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to register user, might be server error: {str(e)}")

class TokenData(BaseModel):
    token: str

@app.post('/logout')
async def logout(data: TokenData):
    async with async_pool.connection() as conn:
        try:
            await conn.execute("UPDATE jwt_blacklist SET blacklist = TRUE WHERE token = %s", (data.token,))
            await conn.commit()
        except Exception:
            pass
    return {"status": "success", "message": "Logged out securely"}

@app.post('/login')
async def login(user: User):
    if not await email_exists(user.email): #1
        raise HTTPException(status_code=401, detail="Email does not exists")
    else:
        try:
            password = await password_fetcher(user.email) #2
            if not bcrypt.checkpw(user.password.encode(), password.encode()): # type: ignore
                raise HTTPException(status_code=401, detail="Invalid email or password")
            user_id = await fetch_id(user.email) #3
            token = await get_jwt_token(user_id)  
            return {"access_token": token, "token_type": "bearer"}
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to login user, might be server error: {str(e)}")


class UserOtp(BaseModel):
    email: EmailStr = Field(..., description="Email address of the user", examples=["example@example.com"])

@app.post('/login_with_otp')
async def login_with_otp(user: UserOtp):
    if not await email_exists(user.email):
        raise HTTPException(status_code=401, detail="Email does not exists")
    if not await otp_status(user.email):
                return {"status": "First verify your email before proceeding."}
    else:
        try:
            user_id = await fetch_id(user.email) 
            token = await get_jwt_token(user_id)  
            await delete_row(user.email)
            return {"access_token": token, "token_type": "bearer"}
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to login user, might be server error: {str(e)}")

@app.post('/change_password')
async def change_password(user: User, new_password: New_Password):
    if not await otp_status(user.email):
            return {"status": "First verify your email before proceeding."}
    await delete_row(user.email)
    email = user.email
    password = user.password
    await login(User(email=email, password=password))
    new_password_value = new_password.password
    if password == new_password_value:
        raise HTTPException(status_code=400, detail="New password cannot be the same as old password")
    hashed = bcrypt.hashpw(new_password_value.encode(), bcrypt.gensalt()).decode()
    async with async_pool.connection() as conn:
        await conn.execute("UPDATE users SET password = %s WHERE email = %s", (hashed, email))
        await conn.commit()
    return {"status": "password changed"}

@app.post('/forgot_password')
async def forgot_password(email_id: UserOtp, new_password: New_Password):
    email = email_id.email
    if not await email_exists(email):
            raise HTTPException(status_code=401, detail="Email not exists")
    if not await otp_status(email):
            return {"status": "First verify your email before proceeding."}
    await delete_row(email)
    new_password_value = new_password.password
    hashed = bcrypt.hashpw(new_password_value.encode(), bcrypt.gensalt()).decode()
    async with async_pool.connection() as conn:
        await conn.execute("UPDATE users SET password = %s WHERE email = %s", (hashed, email))
        await conn.commit()
    return {"status": "password changed successfully"}

@app.delete('/delete_user')
async def delete_user(user: User):
    if not await otp_status(user.email):
            return {"status": "First verify your email before proceeding."}
    await delete_row(user.email)
    email = user.email
    password = user.password
    await login(User(email=email, password=password))
    async with async_pool.connection() as conn:
        await conn.execute("DELETE FROM users WHERE email = %s", (email,))
        await conn.commit()
    return {"status": "user deleted"}

#================================================================
#======================== User Profile ==========================
class UserProfile(BaseModel):
    name: str
    email: EmailStr
    avatar_url: Optional[str] = None
    dob: Optional[date] = None 
    age: Optional[int] = None
    country_code: Optional[str] = None 
    mobile: Optional[str] = None 
    linkedin: Optional[str] = None
    github: Optional[str] = None
    instagram: Optional[str] = None
    job_title: Optional[str] = None

@app.post("/users_profile/insert")
async def user_profile_insert(user: UserProfile, user_id: str = Depends(get_current_user)):
    async with async_pool.connection() as conn:
        await conn.execute("""
            INSERT INTO user_profile 
            (user_id, name, email, avatar_url, dob, age, country_code, mobile, linkedin, github, instagram, job_title)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO NOTHING
        """, (
            user_id, user.name, user.email, user.avatar_url, user.dob, user.age, 
            user.country_code, user.mobile, user.linkedin, user.github, user.instagram, user.job_title
        ))
        return {"status": "Profile created successfully"}

@app.get("/users_profile")
async def get_user_profile(user_id: str = Depends(get_current_user)):
    async with async_pool.connection() as conn:
        cur = await conn.execute("SELECT * FROM user_profile WHERE user_id = %s", (user_id,))
        row = await cur.fetchone()
        
        if row is not None:
            return dict(row)
        return {"error": "Profile not found"}

@app.post("/users_profile/update")
async def update_user_profile(user: UserProfile, user_id: str = Depends(get_current_user)):
    async with async_pool.connection() as conn:
        cur = await conn.execute("""
            UPDATE user_profile 
            SET name = %s, email = %s, avatar_url = %s, dob = %s, age = %s, 
                country_code = %s, mobile = %s, linkedin = %s, github = %s, 
                instagram = %s, job_title = %s, updated_at = NOW()
            WHERE user_id = %s
            RETURNING *
        """, (
            user.name, user.email, user.avatar_url, user.dob, user.age,
            user.country_code, user.mobile, user.linkedin, user.github, 
            user.instagram, user.job_title, user_id
        ))
        
        row = await cur.fetchone()
        if row is not None:
            return dict(row)
        return {"error": "Update failed or user not found"}

#=================================================================
#======================== OTP Authentication =====================
class EmailFormat(BaseModel):
    id: EmailStr

@app.post("/send_otp")
async def Send_Email(email: EmailFormat) -> dict:
    if await email_exists(email.id):
        return {"status": "Email already exists"}
    if await time_remaining(email.id):
        return {"status": "Please try after some time."}
        
    await delete_row(email.id)
    otp = generate_otp()
    subject, content = build_email(otp)
    msg = EmailMessage()
    msg['From'] = SENDER_EMAIL
    msg['To'] = email.id
    msg['Subject'] = subject
    msg.set_content(content)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, PASS_KEY) # type: ignore
            smtp.send_message(msg)
    except Exception as e:
        return {'status': f"can not authenticate with email otp right now, error: {str(e)}"}

    try:
        await create_row(email.id, otp)
    except Exception as e:
        return {'status': f"otp cannot be updated, error: {str(e)}"}

    return {
        'status': f"Otp sent to your Email {email.id}, otp valid for 5 minutes",  # FIXED: missing f-string add kiya
    }

@app.post("/send_otp_to_registered_user")
async def Send_Email_to_registered_user(email: EmailFormat) -> dict:
    if not await email_exists(email.id):
            return {"status": "Email not exists"}
    if await time_remaining(email.id):
        return {"status": "Please try after some time."}
        
    await delete_row(email.id)
    otp = generate_otp()
    subject, content = build_email(otp)
    msg = EmailMessage()
    msg['From'] = SENDER_EMAIL
    msg['To'] = email.id
    msg['Subject'] = subject
    msg.set_content(content)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(str(SENDER_EMAIL), str(PASS_KEY))
            smtp.send_message(msg)
    except Exception as e:
        return {'status': f"can not authenticate with email otp right now, error: {str(e)}"}

    try:
        await create_row(email.id, otp)
    except Exception as e:
        return {'status': f"otp cannot be updated, error: {str(e)}"}

    return {
        'status': f"Otp sent to your Email {email.id}, otp valid for 5 minutes",  # FIXED: missing f-string add kiya
    }
    
@app.get("/verify_otp")
async def veryfy_otp(email_id, otp: str):
    async with async_pool.connection() as conn:
        cur = await conn.execute(
            "SELECT * FROM otp_validation WHERE email = %s",
            (email_id,),
        )
        row = await cur.fetchone()

        if row is None:
            return {'error': 'No OTP record found'}

        if dt.datetime.now() >= row['expiry_time']:
            await delete_row(email_id)
            return {'error': 'OTP expired'}

        if row['attempt_count'] > 3:
            await delete_row(email_id)
            return False

        if bcrypt.checkpw(otp.encode(), row['encrypted_otp'].encode()):
            async with async_pool.connection() as conn:
                await conn.execute("UPDATE otp_validation SET status = %s WHERE email = %s", (True, email_id,),)
                await conn.commit()
            return True

        await update_row(email_id)
        return {'error': 'Invalid OTP'}

#=================================================================
#======================== Gmail Addition =========================
class GmailAuth(BaseModel):
    id: EmailStr = Field(..., description="Gmail address of the user", examples=["example@gmail.com"])
    pass_key: str = Field(..., description="App password generated for Gmail access", examples=["xdhf nmap bxsa mpvo"])

    @field_validator("id")
    @classmethod
    def must_be_gmail(cls, value: str) -> str:
        if not value.lower().endswith("@gmail.com"):
            raise ValueError("Only @gmail.com addresses are allowed")
        return value

    @field_validator("pass_key")
    @classmethod
    def validate_app_password_format(cls, value: str) -> str:
        pattern = r"^[a-z]{4} [a-z]{4} [a-z]{4} [a-z]{4}$"
        if not re.match(pattern, value.strip().lower()):
            raise ValueError(
                "App password must be 16 letters in the format 'xxxx xxxx xxxx xxxx'"
            )
        return value.strip().lower()

@app.post("/gmail_auth_detail")
async def gmail_auth_detail(gmail: GmailAuth, user_id: str = Depends(get_current_user)):
    if await check_email_exists(gmail.id):
        return {"status": "Gmail already exists Generate otp to verify ownership"}
    hashed = await encrypt_value(gmail.pass_key)
    async with async_pool.connection() as conn:
        await conn.execute("INSERT INTO user_gmail (user_id, email, pass_key) VALUES (%s, %s, %s)", 
                           (user_id, gmail.id, hashed, ), )
        await conn.commit()
    return {"status": "success", "message": "Gmail credentials saved successfully"}

@app.post("/gmail_auth_detail_re_register")
async def gmail_auth_detail_re_register(gmail: GmailAuth, user_id: str = Depends(get_current_user)):
    if not await otp_status(gmail.id):
        return {"status": "First verify your email before proceeding."}
    await delete_row(gmail.id)
    await delete_gmail_data_by_email(gmail.id)
    try:
        hashed = await encrypt_value(gmail.pass_key)
        async with async_pool.connection() as conn:
            await conn.execute("INSERT INTO user_gmail (user_id, email, pass_key) VALUES (%s, %s, %s)", 
                            (user_id, gmail.id, hashed, ), )
            await conn.commit()
        return {"status": "success", "message": "Gmail credentials saved successfully"}
    except Exception as e:
        return {"data base not reachable, error": str(e)}

@app.get("/gmail_auth_detail")
async def get_gmail_data(user_id: str = Depends(get_current_user)):
    result = await get_gmail_auth_detail(user_id)
    if not result:
        return {"email": None}
    email, _ = result
    return {"email": email}

@app.delete("/gmail_auth_detail")
async def delete_gmail_data(user_id: str = Depends(get_current_user)):
    async with async_pool.connection() as conn:
        await conn.execute("DELETE FROM user_gmail WHERE user_id = %s", (user_id, ))
        await conn.commit()
    return {"status": "success", "message": "Gmail credentials deleted successfully"}

async def check_email_exists(email: str):
    async with async_pool.connection() as conn:
        cur = await conn.execute("SELECT 1 FROM user_gmail WHERE email = %s", (email,))
        return await cur.fetchone() is not None

async def delete_gmail_data_by_email(email: str):
    async with async_pool.connection() as conn:
        await conn.execute("DELETE FROM user_gmail WHERE email = %s", (email,))
        await conn.commit()

#================================================================
#======================== File Manager ==========================
session = aioboto3.Session()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    try:
        if await get_size(user_id):
            size = 0
            chunk_size = 1024 * 1024  # 1 MB
            contents = bytearray()

            while chunk := await file.read(chunk_size):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="File too large")
                contents.extend(chunk)

            prefix = uuid.uuid4()
            unique_filename = f"{prefix}_{file.filename}"  
            user_upload_dir = os.path.join("app", "uploads", f"{user_id}")
            os.makedirs(user_upload_dir, exist_ok=True)
            new_file_path = os.path.join(user_upload_dir, unique_filename)
            
            with open(new_file_path, "wb") as f:
                f.write(contents)

            key = f"upload/{user_id}/{unique_filename}"
            async with session.client(
                service_name="s3",
                endpoint_url=ENDPOINT_URL,
                aws_access_key_id=ACCESS_KEY_ID,
                aws_secret_access_key=SECRET_ACCESS_KEY,
                region_name="auto",
            ) as client: # type: ignore
                with open(new_file_path, "rb") as f:
                    await client.upload_fileobj(f, Bucket=os.getenv("BUCKET_NAME"), Key=key)
            return {"file_path": new_file_path, "original_name": file.filename}

        else:
            return f"You Have hit the Maximum Limit to upload files, First delete them to Upload New Files"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not upload file: {str(e)}")

@app.get("/list_files")
async def list_files(user_id: str = Depends(get_current_user)):
    try:
        async with session.client(
            service_name="s3",
            endpoint_url=os.getenv("ENDPOINT_URL"),
            aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
            region_name="auto",
        ) as client: # type: ignore
            x = await client.list_objects_v2(Bucket=os.getenv("BUCKET_NAME"), Prefix=f"upload/{user_id}")
        return {"file_lists": [v["Key"] for v in x.get("Contents", [])]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list files: {str(e)}")

@app.get("/download_file")
async def download_file(file_name: str, user_id: str = Depends(get_current_user)):
    key = f"upload/{user_id}/{file_name}"
    try:
        async with session.client(
            service_name="s3",
            endpoint_url=os.getenv("ENDPOINT_URL"),
            aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
            region_name="auto",
        ) as client: # type: ignore
            response    = await client.get_object(Bucket=os.getenv("BUCKET_NAME"), Key=key)
            file_bytes  = await response["Body"].read()

        return Response(content=file_bytes, media_type="application/octet-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not download file: {str(e)}")
    
@app.delete("/delete_file")
async def delete_file(file_name: str, user_id: str = Depends(get_current_user)):
    key = f"upload/{user_id}/{file_name}"
    try:
        async with session.client(
            service_name="s3",
            endpoint_url=os.getenv("ENDPOINT_URL"),
            aws_access_key_id=os.getenv("ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("SECRET_ACCESS_KEY"),
            region_name="auto",
        ) as client: # type: ignore
            await client.delete_object(Bucket=os.getenv("BUCKET_NAME"), Key=key)

        return {"status": "deleted", "file_name": file_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not delete file: {str(e)}")

#================================================================
#======================== USER LLM SELETOR ======================
class LLMProvider(str, Enum):
    openai = "openai"
    anthropic = "anthropic"
    gemini = "gemini"

ALLOWED_MODELS = {
    LLMProvider.openai: {"gpt-5.6-sol","gpt-5.6-terra","gpt-5.6-luna","gpt-5.4","gpt-5.5"},
    LLMProvider.anthropic: {"claude-opus-4-8","claude-sonnet-4-6","claude-haiku-4-5-20251001"},
    LLMProvider.gemini: {"gemini-3.1-pro-preview","gemini-3.5-flash","gemini-3.6-flash"},
}

class UserLLMSelector(BaseModel):
    llm_name: LLMProvider
    model_name: str
    api_key: str

    @field_validator("model_name")
    @classmethod
    def validate_model_for_provider(cls, v, info):
        provider = info.data.get("llm_name")
        if provider and v not in ALLOWED_MODELS[provider]:
            raise ValueError(
                f"'{v}' is not a valid model for provider '{provider.value}'. "
                f"Allowed: {sorted(ALLOWED_MODELS[provider])}"
            )
        return v

@app.get('/llm_selector')
async def get_llm_selector(user_id: str = Depends(get_current_user)):
    async with async_pool.connection() as conn:
        cur = await conn.execute("SELECT llm_name, model_name FROM user_llms WHERE user_id = %s", (user_id,))
        row = await cur.fetchone()
        if row:
            return {"llm_name": row['llm_name'], "model_name": row['model_name']}
        return {"llm_name": None, "model_name": None}

@app.post('/llm_selector')
async def llm_selector(llm: UserLLMSelector, user_id: str = Depends(get_current_user)):
    llm_name = llm.llm_name.value
    model_name = llm.model_name
    api_key = llm.api_key
    hashed = await encrypt_value(api_key)
    async with async_pool.connection() as conn:
        cur = await conn.execute("SELECT 1 FROM user_llms WHERE user_id = %s", (user_id,))
        exists = await cur.fetchone()
        if exists:
            await conn.execute("UPDATE user_llms SET llm_name = %s, model_name = %s, encrypted_api_key = %s WHERE user_id = %s", 
                               (llm_name, model_name, hashed, user_id))
        else:
            await conn.execute("INSERT INTO user_llms (user_id, llm_name, model_name, encrypted_api_key) VALUES (%s, %s, %s, %s)", 
                               (user_id, llm_name, model_name, hashed))
        await conn.commit()
        return {"message": "Model Updated and API Key Encrypted"}
    
@app.delete('/llm_selector')
async def delete_llm_selector(user_id: str = Depends(get_current_user)):
    async with async_pool.connection() as conn:
        await conn.execute("DELETE FROM user_llms WHERE user_id = %s", (user_id,))
        await conn.commit()
        return {"message": "Model preference deleted"}

async def ping_validation(llm_name: str, model_version: str, key: str, user_id):
    try:
        if llm_name == "gemini":
            llm = ChatGoogleGenerativeAI(api_key=SecretStr(key), model=model_version)
            ping = await llm.ainvoke("ping")
            return ping
        elif llm_name == "openai":
            llm = ChatOpenAI(model=model_version, api_key=SecretStr(key))
            ping = await llm.ainvoke("ping")
            return ping
        else:
            llm = ChatAnthropic(model_name=model_version, api_key=SecretStr(key), timeout=None, stop=None)
            ping = await llm.ainvoke("ping")
            return ping
    except Exception:
        return None

@app.post('/model_tester')
async def model_tester(user_id: str = Depends(get_current_user)):
    async with async_pool.connection() as conn:
           cur = await conn.execute("SELECT * FROM user_llms WHERE user_id = %s", (user_id,))
           row = await cur.fetchone()
           if row is None:
               return {"message": "No model found for the user."}

    llm_name, model_name, encrypted_api_key = row['llm_name'], row['model_name'], row['encrypted_api_key'] # type: ignore
    api_key = await decrypt_value(encrypted_api_key)
    if await ping_validation(llm_name, model_name, api_key, user_id) is None:
        async with async_pool.connection() as conn:
            await conn.execute("DELETE FROM user_llms WHERE user_id = %s", (user_id,))
            await conn.commit()
            return {"message": "API KEY MIGHT BE INVALID, Try with Valid API KEY"}
    return {"message": "Model is ready to use."}


#================================================================
#========================  Model Selection =======================
@app.get('/token_usage')
async def total_token_usage(user_id: str = Depends(get_current_user)):
    async with async_pool.connection() as conn:
        cur = await conn.execute(
            "SELECT tokens FROM daily_usage WHERE user_id = %s AND date = CURRENT_DATE", 
            (user_id,)
        )
        row = await cur.fetchone()
        tokens = row["tokens"] if row else 0
        return {"total_tokens": tokens}

async def model_selection(model_version: str,user_id: str):
    usage = await total_token_usage(user_id= user_id)
    if model_version == "jomoai":
        if usage.get("total_tokens", 0) >= 100000:
            # FIXED [Issue #5]: Changed status code from 404 to 429 (Too Many Requests) for daily quota limit hit
            raise HTTPException(status_code=429, detail="Daily quota limit hit, select your own model")
        return jomoai

    async with async_pool.connection() as conn:
        cur = await conn.execute("SELECT * FROM user_llms WHERE user_id = %s", (user_id,))
        cur = await cur.fetchone()

    if cur is None:
        # FIXED [Issue #5]: Changed status code from 404 to 429 for quota limit hit
        raise HTTPException(status_code=429, detail="You have hit Your Daily Quota Limit use your Own model to Continue")

    llm_name = cur['llm_name']
    encrypted_key = cur['encrypted_api_key']
    key = await decrypt_value(encrypted_key)    
    if llm_name == "gemini":
        llm = ChatGoogleGenerativeAI(api_key=SecretStr(key), model=model_version)
        return llm
    elif llm_name == "openai":
        llm = ChatOpenAI(model=model_version, api_key=SecretStr(key))
        return llm
    elif llm_name == "anthropic":
        llm = ChatAnthropic(model_name=model_version, api_key=SecretStr(key), timeout=None, stop=None)
        return llm
    else:
        raise HTTPException(status_code=404, detail="Your LLM is not Responding check with your provider")