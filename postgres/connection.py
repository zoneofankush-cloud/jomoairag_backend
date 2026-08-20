#=================================================================
#====================  ENV VARIABLE SETUP  =======================
from dotenv import load_dotenv
import os
load_dotenv()

DBConnection = os.getenv("DATA_BASE_URL")
if not DBConnection:
    raise ValueError("DATA_BASE_URL environment variable is not set")

#=================================================================
#=======================  Module Imports =========================
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from typing import Any, cast

async_pool = AsyncConnectionPool(
        conninfo=DBConnection,
        max_size=5,
        min_size=1,
        max_idle=60,
        check=AsyncConnectionPool.check_connection,
        kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row},
        open=False
    )

async def memory_saver_state():
    try:
        await async_pool.open()          
        saver = AsyncPostgresSaver(conn=cast(Any, async_pool))
        await saver.setup()             
        return saver
    except Exception as e:
        raise ValueError(f"Error creating memory saver state: {e}")

async def memory_store_state():
    try:
        await async_pool.open()          
        store = AsyncPostgresStore(conn=cast(Any, async_pool))
        await store.setup()    
        return store
    except Exception as e:
        raise ValueError(f"Error creating memory store state: {e}")

async def cleanup_old_user_data(user_id: str):
    try:
        await async_pool.open()
        async with async_pool.connection() as conn:
            # 1. Check if user already exists (i.e. has embedded data/threads)
            cur = await conn.execute("SELECT 1 FROM langchain_pg_embedding WHERE cmetadata->>'user_id' = %s LIMIT 1", (user_id,))
            has_vectors = await cur.fetchone()
            
            cur2 = await conn.execute("SELECT thread_id FROM threads WHERE user_id = %s", (user_id,))
            rows = await cur2.fetchall()
            thread_ids = [str(r['thread_id']) for r in rows]
            
            # If data exists, wipe it out
            if has_vectors or thread_ids:
                # 1. Delete vectors
                await conn.execute("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'user_id' = %s", (user_id,))
                
                # 2. Delete raw chunks
                namespace_pattern = f'%{user_id}%'
                await conn.execute("DELETE FROM langchain_key_value_stores WHERE namespace LIKE %s", (namespace_pattern,))
                
                # 3. Delete LangGraph Checkpoints
                if thread_ids:
                    await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = ANY(%s)", (thread_ids,))
                    await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ANY(%s)", (thread_ids,))
                    await conn.execute("DELETE FROM checkpoints WHERE thread_id = ANY(%s)", (thread_ids,))
                
                # 4. Delete Chat Threads
                await conn.execute("DELETE FROM threads WHERE user_id = %s", (user_id,))
                
                # 5. Delete User's LSTM Memory from public.store
                await conn.execute("DELETE FROM store WHERE prefix LIKE %s", (f"%{user_id}%",))
                
            # Note: Not touching users, user_profile, jwt_blacklist, user_gmail, etc.
    except Exception as e:
        print(f"Error during cleanup: {e}")