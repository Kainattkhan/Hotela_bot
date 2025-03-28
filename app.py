from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import re
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY or not HUGGINGFACE_ACCESS_TOKEN:
    raise ValueError("Missing API keys! Ensure GROQ_API_KEY and HUGGINGFACE_ACCESS_TOKEN are set.")

# Initialize FastAPI
app = FastAPI()

# Lazy loading of documents and vector database
documents = None
split_documents = None
vectordb = None
retriever = None
retriever_tool = None

class ChatRequest(BaseModel):
    query: str

def fetch_website_content(url: str) -> str:
    """Fetch and clean text content from the given website."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text_content = soup.get_text(separator=" ").strip()
        return re.sub(r"\s+", " ", text_content)
    except requests.RequestException as e:
        return ""

def initialize_retriever():
    """Lazy load document retriever."""
    global documents, split_documents, vectordb, retriever, retriever_tool
    if documents is None:
        pdf_loader = PyPDFLoader("./hotela.pdf")
        word_loader = UnstructuredWordDocumentLoader("./business.docx")
        documents = pdf_loader.load() + word_loader.load()

        hotela_content = fetch_website_content("https://hotelaapp.com/")
        if hotela_content:
            from langchain.schema import Document  
            hotela_doc = Document(page_content=hotela_content, metadata={"source": "Hotela Website"})
            documents.append(hotela_doc)

        from langchain.schema import Document  
        documents = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in documents]

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_documents = splitter.split_documents(documents)

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'token': HUGGINGFACE_ACCESS_TOKEN}
        )

        vectordb = FAISS.from_documents(split_documents, embedding_model)
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        retriever_tool = create_retriever_tool(retriever, name="Hotelaapp_search", description="Search for information")

def query_groq(question: str) -> str:
    """Query the Groq API."""
    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": question}
            ],
            "temperature": 0.5,
            "max_tokens": 200,
        }
        response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.RequestException as e:
        return f"Error: {str(e)}"

@app.get("/")
def home():
    return {"message": "Hello, FastAPI is running on VPS!"}

@app.post("/chat")
def chat(request: ChatRequest):
    """Chat endpoint."""
    initialize_retriever()

    user_query = request.query.strip().lower()
    if not user_query:
        raise HTTPException(status_code=400, detail="Missing 'query' field")

    greetings_pattern = r"^(hello|hi|hey|how are you|good morning|good evening)[\W]*$"
    if re.match(greetings_pattern, user_query):
        return {"response": "Hello! How can I assist you today?"}

    relevant_context = retriever_tool.run(user_query) if retriever_tool else ""
    if not relevant_context or len(relevant_context) < 10:
        return {"response": "I'm here to help! What would you like to know?"}

    full_prompt = f"Context: {relevant_context}\n\nUser: {user_query}"
    response = query_groq(full_prompt)
    return {"response": response}


