import datetime
from postgres.connection import async_pool

async def check_ownership(thread_id, user_id):
    async with async_pool.connection() as conn:
        cur = await conn.execute("SELECT thread_id, title FROM threads WHERE thread_id = %s AND user_id = %s", (thread_id, user_id,),)
        thread = await cur.fetchone()
        return thread

async def email_exists(email: str) -> bool:
    async with async_pool.connection() as conn:
        cur = await conn.execute("SELECT 1 FROM public.users WHERE email = %s", (email,))
        email_id = await cur.fetchone()
        return email_id is not None

async def password_fetcher(email: str):
    async with async_pool.connection() as conn:
        cur = await conn.execute("SELECT password FROM public.users WHERE email = %s", (email,))
        password = await cur.fetchone()
    if password is None:
        return None
    else:
        return password['password']

async def fetch_id(email):
    async with async_pool.connection() as conn:
        curr = await conn.execute("SELECT id FROM public.users WHERE email = %s", (email,))
        id = await curr.fetchone()
        if id is None:
               return None
        else:
            return id['id']

