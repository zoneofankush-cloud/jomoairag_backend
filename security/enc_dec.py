#=================================================================
#====================  ENV VARIABLE SETUP  =======================
import os
from dotenv import load_dotenv
load_dotenv()
FERNET_KEY = os.getenv("FERNET_KEY")
if not FERNET_KEY:
    raise ValueError("FERNET_KEY not set in .env")

#=================================================================
#=======================  Module Imports =========================
from cryptography.fernet import Fernet

#=================================================================
#=======================  SECURITY SETUPS ========================
fernet = Fernet(FERNET_KEY.encode())
async def encrypt_value(value: str) -> str:
    return fernet.encrypt(value.encode()).decode()

async def decrypt_value(token: str) -> str:
    return fernet.decrypt(token.encode()).decode()
