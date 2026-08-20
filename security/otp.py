import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postgres.connection import async_pool
import bcrypt, numpy as np, datetime as dt

def generate_otp() -> str:
    # FIXED: har baar naya OTP generate hota hai, module-level static value nahi
    return str(np.random.randint(100000, 999999))

def build_email(otp: str) -> tuple[str, str]:
    subject = "OTP Verification - Jomo AI"
    content = f"""
Dear User,
Your One Time Password (OTP) is: {otp}
Use This OTP to validate your email within 5 minutes.

Regards,
JOMO AI
"""
    return subject, content

async def create_row(email, otp: str) -> dict:
    hashed = bcrypt.hashpw(otp.encode(), salt=bcrypt.gensalt()).decode() 
    async with async_pool.connection() as conn:
        await conn.execute(
            "INSERT INTO otp_validation (email, expiry_time, encrypted_otp, attempt_count) VALUES (%s, %s, %s, 0)",
            (email, dt.datetime.now() + dt.timedelta(minutes=5), hashed),
        )
        await conn.commit()
    return {'message': f"Otp generated and sent to {email}"}

async def update_row(email):
    async with async_pool.connection() as conn:
        await conn.execute(
            "UPDATE otp_validation SET attempt_count = attempt_count + 1 WHERE email = %s",
            (email,),
        )
        await conn.commit()  # FIXED: commit missing tha pehle

async def delete_row(email):
    async with async_pool.connection() as conn:
        await conn.execute(
            "DELETE FROM otp_validation WHERE email = %s",
            (email,),
        )
        await conn.commit()  # FIXED: commit missing tha pehle

async def otp_status(email_id):
    async with async_pool.connection() as conn:
        cur = await conn.execute(
            "SELECT status, expiry_time FROM otp_validation WHERE email = %s",
            (email_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return False
        if dt.datetime.now() >= row['expiry_time']:
            return False
        return row['status']

async def time_remaining(email_id):
    async with async_pool.connection() as conn:
            cur = await conn.execute(
                "SELECT expiry_time FROM otp_validation WHERE email = %s",
                (email_id,),
            )
            row = await cur.fetchone()
            if not row:
                  return False
            return dt.datetime.now() < row['expiry_time']

