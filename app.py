from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import re 
import requests
import asyncio
# from langchain_community.embeddings import HuggingFaceEmbeddings
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

# Load documents (synchronously for better Flask integration)
def load_documents():
    pdf_loader = PyPDFLoader("./hotela.pdf")
    pdf_documents = pdf_loader.load()

    word_loader = UnstructuredWordDocumentLoader("./business.docx")
    word_documents = word_loader.load()

    return pdf_documents + word_documents

# Prepare document chunks
documents = load_documents()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_documents = splitter.split_documents(documents)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_kwargs={'token': HUGGINGFACE_ACCESS_TOKEN}
)

# Create FAISS vector store
vectordb = FAISS.from_documents(split_documents, embedding_model)

# Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Create retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    name="Hotelaapp_search",
    description="Search for information about Hotelaapp.com"
)

# Function to send queries to Groq API
def query_groq(question):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Keep your responses short and precise."},
            {"role": "user", "content": question}
        ],
        "temperature": 0.5,  # Lower temp = more focused responses
        "max_tokens": 200,   
    }
    try:
        response = requests.post(GROQ_API_URL, json=payload, headers=headers)
        response.raise_for_status()  # Raise error for bad status codes
        raw_response = response.json()["choices"][0]["message"]["content"]
        return raw_response.strip()
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Hotela Chatbot API! Use the /chat endpoint to interact."

# Chat endpoint
@app.route("/chat", methods=["POST"])
def chat():
    global retriever_tool  # Lazy-load retriever tool only when needed

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    user_query = data.get("query", "").strip().lower()

    if not user_query:
        return jsonify({"error": "Missing 'query' field"}), 400

    # Handle greetings more flexibly (removes punctuation and matches common greetings)
    greetings_pattern = r"^(hello|hi|hey|how are you|good morning|good evening)[\W]*$"
    
    if re.match(greetings_pattern, user_query):
        return jsonify({"response": "Hello! How can I assist you today?"})

    # Lazy-load retriever tool (reduces memory usage)
    if not retriever_tool:
        retriever_tool = create_retriever_tool(
            retriever,
            name="Hotelaapp_search",
            description="Search for information about Hotelaapp.com"
        )

    # Retrieve relevant context with reduced search size (k=5)
    relevant_context = retriever_tool.run(user_query)

    if not relevant_context or len(relevant_context) < 10:
        return jsonify({"response": "I'm here to help! What would you like to know?"})

    # Query Groq API with context
    full_prompt = f"Context: {relevant_context}\n\nUser: {user_query}"
    response = query_groq(full_prompt)

    # Remove excessive newlines and multiple spaces
    cleaned_response = re.sub(r"\s+", " ", response).strip()

    return jsonify({"response": cleaned_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
