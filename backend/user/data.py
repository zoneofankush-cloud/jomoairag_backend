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


from fastapi import Request
import httpx


async def get_user_info(request: Request, fallback_tz: str):
    ip = request.headers.get("CF-Connecting-IP") or request.headers.get("X-Forwarded-For")
    if not ip and request.client:
        ip = request.client.host
    
    # In local testing (127.0.0.1), IP API might fail or return local data. We handle exceptions.
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.get(f"http://ip-api.com/json/{ip}")
            data = res.json()
            if data.get("status") == "success":
                loc = f"{data.get('city')}, {data.get('country')}"
                tz = data.get("timezone", fallback_tz)
                return loc, tz
    except Exception:
        pass
    return "Unknown", fallback_tz


