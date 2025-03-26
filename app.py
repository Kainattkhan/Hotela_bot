from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import re
import requests
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

# Flask app
app = Flask(__name__)

# Lazy loading of documents and vector database
documents = None
split_documents = None
vectordb = None
retriever = None
retriever_tool = None

def fetch_website_content(url):
    """Fetch and clean text content from the given website."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        text_content = soup.get_text(separator=" ").strip()
        
        # Remove excessive spaces and new lines
        clean_text = re.sub(r"\s+", " ", text_content)
        
        print("Website content fetched successfully.")
        return clean_text
    except requests.RequestException as e:
        print(f"Error fetching website content: {e}")
        return ""

def initialize_retriever():
    global documents, split_documents, vectordb, retriever, retriever_tool
    if documents is None:
        print("Loading documents...")

        # Load PDF and Word documents
        pdf_loader = PyPDFLoader("./hotela.pdf")
        word_loader = UnstructuredWordDocumentLoader("./business.docx")
        documents = pdf_loader.load() + word_loader.load()

        # Fetch Hotela website content
        hotela_content = fetch_website_content("https://hotelaapp.com/")
        if hotela_content:
            from langchain.schema import Document  # Import Document class
            hotela_doc = Document(page_content=hotela_content, metadata={"source": "Hotela Website"})
            documents.append(hotela_doc)

        # Ensure all documents are properly formatted
        from langchain.schema import Document  # Ensure all documents follow the Document schema
        documents = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in documents]

        # Split documents into smaller chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_documents = splitter.split_documents(documents)

        # Embeddings and Vector Database
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'token': HUGGINGFACE_ACCESS_TOKEN}
        )

        vectordb = FAISS.from_documents(split_documents, embedding_model)
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        retriever_tool = create_retriever_tool(retriever, name="Hotelaapp_search", description="Search for information")

        print("Retriever initialized successfully with Hotela website data.")

# Function to send queries to Groq API
def query_groq(question):
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant. Keep responses short, precise and answer user queries in a casual and helpful tone"},
                {"role": "user", "content": question}
            ],
            "temperature": 0.5,
            "max_tokens": 200,
        }
        response = requests.post(GROQ_API_URL, json=payload, headers=headers,timeout=30)
        response.raise_for_status()
        raw_response = response.json()["choices"][0]["message"]["content"]
        return raw_response.replace("\n\n", " ").replace("\n", " ").strip()
    except requests.RequestException as e:
        print("Error Details:", e.response.text if e.response else str(e))
        return f"Error: {str(e)}"

@app.route("/")
def home():
    return jsonify({"status": "running"})

@app.route("/chat", methods=["POST"])
def chat():
    initialize_retriever()
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415
    
    data = request.get_json()
    user_query = data.get("query", "").strip().lower()
    
    if not user_query:
        return jsonify({"error": "Missing 'query' field"}), 400

    greetings_pattern = r"^(hello|hi|hey|how are you|good morning|good evening)[\W]*$"
    if re.match(greetings_pattern, user_query):
        return jsonify({"response": "Hello! How can I assist you today?"})
    
    relevant_context = retriever_tool.run(user_query) if retriever_tool else ""
    if not relevant_context or len(relevant_context) < 10:
        return jsonify({"response": "I'm here to help! What would you like to know?"})
    
    full_prompt = f"Context: {relevant_context}\n\nUser: {user_query}"
    response = query_groq(full_prompt)
    cleaned_response = re.sub(r"\s+", " ", response).strip()
    return jsonify({"response": cleaned_response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
