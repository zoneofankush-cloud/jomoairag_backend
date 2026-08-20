#=================================================================
#========================  PATH IMPORTS  =========================
from security.enc_dec import decrypt_value
from postgres.connection import async_pool


#=================================================================
#=======================  Module Imports =========================
import imaplib, smtplib, asyncio, email as email_lib
from langchain.tools import tool
from email.message import EmailMessage
from pydantic import BaseModel, EmailStr, Field
from langchain_core.runnables import RunnableConfig

#=================================================================
#=======================  BASE SETUP =============================
class Email(BaseModel):
    to: EmailStr = Field(..., description="Email address where user like to send message")
    subject: str = Field(..., description="Subject of the email")
    body: str = Field(..., description="Body of the email")

#=================================================================
#======================= AUTHORIZATION ============================
async def get_gmail_auth_detail(user_id: str):
    async with async_pool.connection() as conn:
        curr = await conn.execute("SELECT * FROM user_gmail WHERE user_id = %s", (user_id,))
        gmail_record = await curr.fetchone()
        if not gmail_record:
            return None 
        pass_key = await decrypt_value(gmail_record['pass_key'])
        if isinstance(pass_key, dict):
            pass_key = pass_key.get('key', pass_key)
        if isinstance(pass_key, bytes):
            pass_key = pass_key.decode(errors='ignore')
        return gmail_record['email'], pass_key

#=================================================================
#=======================  SEND EKAIL =============================
@tool
async def Send_Email(email_details: Email, config: RunnableConfig):
    """ Generate Email as per user reqirements """
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return {"error": "Authentication failed. User ID not found."}

    try:
        auth_detail = await get_gmail_auth_detail(user_id)
    except Exception as e:
        return {"error": f"Could not fetch Gmail credentials: {str(e)}"}

    if not auth_detail:
        return {"error": "Please save your Gmail credentials in the Account settings first."}
    SENDER_EMAIL, pass_key = auth_detail

    msg = EmailMessage()
    msg['From'] = SENDER_EMAIL
    msg['To'] = email_details.to
    msg['Subject'] = email_details.subject
    body = email_details.body
    msg.set_content(body)

    def _send():
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, pass_key)
            smtp.send_message(msg)

    try:
        await asyncio.to_thread(_send)
        return {
            "status": "success",
            "email": email_details.to,
            "subject": email_details.subject,
            "body": body
        }
    except Exception as e:
        return {"error": f"Email could not be sent: {str(e)}"}

#=================================================================
#=======================  FETCH EMAILS ===========================
@tool
async def fetch_emails(n: int, config: RunnableConfig):
    """Fetch the latest n emails from Gmail. 
    If gmail does not belong to the owner, tells the human it is a security issue. 
    Do not try to fetch or prompt the data from an email which does not belong to you."""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return {"error": "Authentication failed. Trying to fetch another emails leads to revoke and delete your account peemanently"}

    try:
        auth_detail = await get_gmail_auth_detail(user_id)
    except Exception as e:
        return {"error": f"Could not fetch Gmail credentials: {str(e)}"}

    if not auth_detail:
        return {"error": "Please save your Gmail credentials in the Account settings first."}

    SENDER_EMAIL, pass_key = auth_detail

    def _fetch():
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(SENDER_EMAIL, pass_key)
        mail.select("inbox")
        _, data = mail.search(None, "ALL")
        ids = data[0].split()
        email_list = []
        for x in ids[::-1][:n]:
            _, msg_data = mail.fetch(x, "(RFC822)")
            msg = email_lib.message_from_bytes(msg_data[0][1]) # type: ignore
            body = None
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(errors="ignore") # type: ignore
                        break
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore") # type: ignore

            msg_body = body if body else "(no plain text body found)"
            email_list.append({
                "From": msg["From"],
                "Subject": msg["Subject"],
                "body": msg_body,
            })
        mail.logout()
        return email_list

    try:
        return await asyncio.to_thread(_fetch)
    except Exception as e:
        return {"error": f"Failed to fetch emails: {str(e)}"}