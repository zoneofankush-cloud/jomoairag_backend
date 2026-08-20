# files.py

#=================================================================
#=======================  Module Imports =========================
import json, asyncio, base64
from io import BytesIO
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from PIL import Image
import pandas as pd
import os

ALLOWED_UPLOAD_DIR = os.path.abspath(os.path.join(os.getcwd(), "app", "uploads"))

def _is_path_safe(user_path: str) -> bool:
    if not user_path:
        return False
    real_path = os.path.realpath(user_path)
    allowed_real = os.path.realpath(ALLOWED_UPLOAD_DIR)
    return real_path == allowed_real or real_path.startswith(allowed_real + os.sep)


def get_text(content):
    if isinstance(content, list):
        return "".join(x.get("text", "") for x in content if isinstance(x, dict))
    else:
        return content


#=================================================================
#=======================  File Handler ===========================
@tool
async def read_document(file_path: str) -> str:
    """Extract and return all text content from a file (DOCX, or plain text)."""
    if not _is_path_safe(file_path):
        return "[ERROR] Access denied: file_path is outside the allowed uploads directory"

    def _read():
        if file_path.endswith('.docx'):
            from docx import Document
            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

    try:
        text = await asyncio.to_thread(_read)
        if len(text) > 8000:
            text = text[:8000] + "\n...[Content truncated due to length]"
        return text
    except Exception as e:
        return f"[ERROR reading file]: {str(e)}"

@tool
async def load_pdf_document(pdf_file: str) -> str:
    """ Load a .pdf document from a file path. to extract the text from .pdf file """
    if not _is_path_safe(pdf_file):
        return "[ERROR] Access denied: pdf_file is outside the allowed uploads directory"

    def _load():
        file = PyMuPDF4LLMLoader(file_path=pdf_file)
        doc = file.load()
        return " ".join(x.page_content for x in doc)

    try:
        return await asyncio.to_thread(_load)
    except Exception as e:
        return f"[ERROR loading PDF File, File path is not Correct]: {str(e)}"

@tool
async def ipynb_file_reader(file: str):
    """ Read the IPYNB file and convert it to JSON format  """
    if not _is_path_safe(file):
        return "[ERROR] Access denied: file_path is outside the allowed uploads directory"

    def _read():
        with open(file, 'r') as f:
            data = f.read()
        return json.loads(data)

    try:
        return await asyncio.to_thread(_read)
    except Exception as e:
        return f"[ERROR reading ipynb file]: {str(e)}"

@tool
async def read_image(image_path: str, user_query: str, config: RunnableConfig) -> str:
    """
    Use this tool to analyze, describe, or answer specific questions about a local image file based on a user's prompt.
    Args:
        image_path (str): The absolute or relative path to the image file (supports PNG, JPEG, WebP, BMP).
        user_query (str): The specific question or instruction regarding what to look for or analyze in the image.
    Returns:
        str: The AI model's textual answer or analysis matching the user_query.
    """
    if not _is_path_safe(image_path):
        return "[ERROR] Access denied: image_path is outside the allowed uploads directory"
    llm = config["configurable"]["llm"] # type: ignore

    def _encode():
        img = Image.open(image_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    try:
        img_base64 = await asyncio.to_thread(_encode)
        message = HumanMessage(
            content=[
                {"type": "text", "text": user_query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                },
            ]
        )
        response = await llm.ainvoke([message])
        return get_text(response.content)
    except Exception as e:
        return f"[ERROR reading image]: {str(e)}"

@tool
async def read_excel_file(file: str) -> str:
    """
    Use this tool to read the contents of a local Excel file (.xlsx or .xls) and convert it into a JSON string format.
    Args:
        file (str): The absolute or relative system path to the Excel file.
    Returns:
        str: A JSON string representation of the Excel spreadsheet data, structured by columns or rows.
    """
    if not _is_path_safe(file):
        return "[ERROR] Access denied: file_path is outside the allowed uploads directory"
    try:
        df = await asyncio.to_thread(pd.read_excel, file)
        return df.to_json()
    except Exception as e:
        return f"[ERROR reading excel file]: {str(e)}"

@tool
async def read_html_file(file: str) -> str:
    """
    Use this tool to read a local HTML file (.html) containing tables and convert the first table into a JSON string format.

    Args:
        file (str): The absolute or relative system path to the HTML file.

    Returns:
        str: A JSON string representation of the first table found in the HTML file, or "[]" if no tables exist.
    """
    if not _is_path_safe(file):
        return "[ERROR] Access denied: file_path is outside the allowed uploads directory"
    try:
        dfs = await asyncio.to_thread(pd.read_html, file)
        return dfs[0].to_json() if dfs else "[]"
    except Exception as e:
        return f"[ERROR reading html file]: {str(e)}"

@tool
async def read_xml_file(file: str) -> str:
    """
    Use this tool to read the contents of a local XML file (.xml) and convert its structured data into a JSON string format.

    Args:
        file (str): The absolute or relative system path to the XML file.

    Returns:
        str: A JSON string representation of the XML data, or "[]" if the file is empty or invalid.
    """
    if not _is_path_safe(file):
        return "[ERROR] Access denied: file_path is outside the allowed uploads directory"
    try:
        df = await asyncio.to_thread(pd.read_xml, file)
        return df.to_json()
    except Exception:
        return "[]"