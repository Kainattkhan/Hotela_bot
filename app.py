from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import AsyncHtmlLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_core.tools.retriever import create_retriever_tool

# Load environment variables
load_dotenv()
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

print("Loaded GROQ_API_KEY:", GROQ_API_KEY)  # Check if this prints the correct key


# Load documents asynchronously
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
documents = splitter.split_documents(all_documents)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_kwargs={'token': HUGGINGFACE_ACCESS_TOKEN}
)

# Create FAISS vector store
vectordb = FAISS.from_documents(documents, embedding_model)

# Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# Create retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    name="Hotelaapp.com_search",
    description="Search for information about Hotelaapp.com"
)

# Flask app
app = Flask(__name__)

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
        "max_tokens": 200,   # Limit response length
    }
    response = requests.post(GROQ_API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        raw_response = response.json()["choices"][0]["message"]["content"]
        
        # Post-processing: Remove unnecessary formatting
        clean_response = raw_response.replace("\n\n", " ").replace("\n", " ").strip()
        
        return clean_response
    else:
        return f"Error: {response.json()}"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        return jsonify({"error": "Use /chat endpoint for POST requests"}), 405
    return "Welcome to the Hotela Chatbot API! Use the /chat endpoint to interact."

# Flask route for chatbot
@app.route("/chat", methods=["POST"])
def chat():
    if not request.is_json:  # ✅ Check if request is JSON
        return jsonify({"error": "Request must be JSON"}), 415  # Return proper error

    data = request.get_json()  # ✅ Use get_json() to avoid errors
    user_query = data.get("query")

    if not user_query:  # ✅ Check if query is provided
        return jsonify({"error": "Missing 'query' field"}), 400

    # Retrieve relevant context
    relevant_context = retriever_tool.run(user_query)

    if not relevant_context:
        return jsonify({"response": "Sorry, I couldn't find any relevant information."})

    # Build prompt and get response
    full_prompt = f"Context: {relevant_context}\n\nUser: {user_query}"
    response = query_groq(full_prompt)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
