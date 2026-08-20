#=================================================================
#====================  ENV VARIABLE SETUP  =======================
import os
from dotenv import load_dotenv
load_dotenv()

# We rely on Docker Compose's environment variable for GOOGLE_APPLICATION_CREDENTIALS
#=================================================================
#=======================  Module Imports =========================
from langchain_google_genai import ChatGoogleGenerativeAI

#=================================================================
#=======================  JOMO AI SETUP ==========================

# Use Vertex AI for the default JOMO AI model
jomoai = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    vertexai=True,
    project="project-45267852-1c84-4327-ba7",
    location="global",
    streaming=True
)