from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import requests
import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import AsyncHtmlLoader, PyPDFLoader, UnstructuredWordDocumentLoader

# Load environment variables
load_dotenv()
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

app = Flask(__name__)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_kwargs={'token': HUGGINGFACE_ACCESS_TOKEN}
)

# FAISS index path (saves memory by keeping data on disk)
FAISS_INDEX_PATH = "./faiss_index"

def load_documents():
    """ Load and process documents only when needed (Lazy Loading) """
    try:
        web_loader = AsyncHtmlLoader(["https://hotelaapp.com"])
        docs = web_loader.load()

        pdf_loader = PyPDFLoader("./hotela.pdf")
        pdf_documents = pdf_loader.load()

        word_loader = UnstructuredWordDocumentLoader("./business.docx")
        word_documents = word_loader.load()

        # Combine all sources
        all_documents = docs + pdf_documents + word_documents

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        return splitter.split_documents(all_documents)

    except Exception as e:
        print(f"Error loading documents: {e}")
        return []  # Return empty list if error occurs

# Load FAISS if exists, otherwise create it
if os.path.exists(FAISS_INDEX_PATH):
    vectordb = FAISS.load_local(FAISS_INDEX_PATH, embedding_model)
else:
    documents = load_documents()
    vectordb = FAISS.from_documents(documents, embedding_model)
    vectordb.save_local(FAISS_INDEX_PATH)  # Save FAISS index to disk

# Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 10})
retriever_tool = create_retriever_tool(retriever, name="Hotelaapp.com_search", description="Search for information about Hotelaapp.com")

def query_groq(question):
    """ Send query to Groq API """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Keep responses short and precise."},
            {"role": "user", "content": question}
        ],
        "temperature": 0.5,
        "max_tokens": 200,
    }
    response = requests.post(GROQ_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    return f"Error: {response.json()}"

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Hotela Chatbot API! Use the /chat endpoint to interact."

@app.route("/chat", methods=["POST"])
def chat():
    """ Handle chat requests """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    user_query = data.get("query", "").strip().lower()

    if not user_query:
        return jsonify({"error": "Missing 'query' field"}), 400

    # Handle greetings flexibly
    greetings_pattern = r"^(hello|hi|hey|how are you|good morning|good evening)[\W]*$"
    if re.match(greetings_pattern, user_query):
        return jsonify({"response": "Hello! How can I assist you today?"})

    # Retrieve relevant context
    relevant_context = retriever_tool.run(user_query)

    if not relevant_context or len(relevant_context) < 10:
        return jsonify({"response": "I'm here to help! What would you like to know?"})

    # Query Groq API
    full_prompt = f"Context: {relevant_context}\n\nUser: {user_query}"
    response = query_groq(full_prompt)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
