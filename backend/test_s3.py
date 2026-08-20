import asyncio
import aioboto3
import os
from botocore.client import Config
from dotenv import load_dotenv

load_dotenv("backend/.env")
ENDPOINT_URL = os.getenv("ENDPOINT_URL")
ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")

async def test_s3():
    session = aioboto3.Session()
    try:
        async with session.client(
            service_name="s3",
            endpoint_url=ENDPOINT_URL,
            aws_access_key_id=ACCESS_KEY_ID,
            aws_secret_access_key=SECRET_ACCESS_KEY,
            region_name="auto",
        ) as s3_client:
            data = await s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix="upload/test")
            print("SUCCESS!", data.get("Contents"))
    except Exception as e:
        print("ERROR:", repr(e))

asyncio.run(test_s3())
