from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import uvicorn
import json
import os
import re
import time
import requests
import logging

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from typing import List, Union
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import FastAPI

logging.basicConfig(level=logging.DEBUG)
# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()
HUGGINGFACE_ACCESS_TOKEN = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY or not HUGGINGFACE_ACCESS_TOKEN:
    raise ValueError("Missing API keys! Ensure GROQ_API_KEY and HUGGINGFACE_ACCESS_TOKEN are set.")

# Lazy loading variables
documents = None
split_documents = None
vectordb = None
retriever = None
retriever_tool = None

# Request Body Model
class ChatRequest(BaseModel):
    query: str

# Helper functions
def setup_selenium_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--single-process")
    chrome_options.add_argument("--disable-accelerated-2d-canvas")
    chrome_options.add_argument("--no-zygote")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36")
    service = Service("/usr/lib/chromium-browser/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(30) 
    return driver

def fetch_website_content(urls):
    if isinstance(urls, str):
        urls = [urls]

    driver = setup_selenium_driver()
    website_documents = []  # Store each page as a document

    for url in urls:
        try:
            driver.get(url)
            time.sleep(10)
            page_text = ""

            # Special handling for blogs page
            if "blogs" in url.lower():
                blog_links = driver.find_elements(By.TAG_NAME, "a")
                blog_urls = []

                for link in blog_links:
                    href = link.get_attribute("href")
                    if href and "/blogs/" in href and href not in blog_urls:
                        blog_urls.append(href)

                logging.info(f"Found {len(blog_urls)} blog articles.")

                # Visit each blog article and extract content
                for blog_url in blog_urls:
                    try:
                        driver.get(blog_url)
                        time.sleep(5)
                        blog_body = driver.find_element(By.TAG_NAME, "body")
                        blog_text = blog_body.text.strip()

                        if blog_text:
                            website_documents.append(Document(
                                page_content=blog_text,
                                metadata={
                                    "source": "Hotela Blog",
                                    "url": blog_url,
                                    "type": "blog"
                                }
                            ))
                            logging.info(f"Fetched blog article: {blog_url}")
                        else:
                            logging.warning(f"No content in blog: {blog_url}")
                    except Exception as e:
                        logging.error(f"Error fetching blog article {blog_url}: {e}")
                        continue

            else:
                # Normal page (home, price, features, aboutus, etc.)
                body_element = driver.find_element(By.TAG_NAME, "body")
                page_text = body_element.text.strip()

                if page_text:
                    website_documents.append(Document(
                        page_content=page_text,
                        metadata={
                            "source": "Hotela Website",
                            "url": url,
                            "type": "website"
                        }
                    ))
                    logging.info(f"Fetched content from {url}")
                else:
                    logging.warning(f"No content found on {url}")

        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            continue

    driver.quit()
    return website_documents

def initialize_retriever():
    """Lazy load document retriever."""
    global documents, split_documents, vectordb, retriever, retriever_tool
    if documents is None:
        logging.info("Initializing retriever...")
        pdf_loader = PyPDFLoader("./hotela.pdf")
        word_loader = UnstructuredWordDocumentLoader("./business.docx")
        documents = pdf_loader.load() + word_loader.load()
        logging.info(f"Loaded {len(documents)} documents from PDF and Word files")

        # Fetch website content
        try:
            logging.info("Fetching website content...")
            hotela_documents = fetch_website_content([
                "https://hotelaapp.com/",
                "https://hotelaapp.com/price",
                "https://hotelaapp.com/features",
                "https://hotelaapp.com/aboutUs",
                "https://hotelaapp.com/blogs"
            ])

            if hotela_documents:
                documents.extend(hotela_documents)
                logging.info(f"Successfully loaded {len(hotela_documents)} website documents. Total documents: {len(documents)}")
            else:
                logging.warning("Warning: Could not fetch website content")

        except Exception as e:
            logging.error(f"Error loading website content: {str(e)}")

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        split_documents = splitter.split_documents(documents)
        logging.info(f"Split documents into {len(split_documents)} chunks")

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'token': HUGGINGFACE_ACCESS_TOKEN}
        )

        vectordb = FAISS.from_documents(split_documents, embedding_model)
        retriever = vectordb.as_retriever(
            search_kwargs={
                "k": 10,
                "filter": None
            }
        )
        retriever_tool = create_retriever_tool(retriever, name="Hotelaapp_search", description="Search for Hotela information")
        logging.info("Retriever initialization complete.")

def query_groq(question: str) -> str:
    """Query the Groq API."""
    try:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for Hotela. Provide concise, accurate responses. Keep responses brief and to the point. Do not use markdown formatting or asterisks. If you're unsure, say so."
                },
                {"role": "user", "content": question}
            ],
            "temperature": 0.2,
            "max_tokens": 150,
            "top_p": 0.9,
            "frequency_penalty": 0.7,
            "presence_penalty": 0.7
        }
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"Error querying Groq API: {e}")
        return "An error occurred while processing your request."

class Plan(BaseModel):
    name: str
    price: str
    description: str

class Feature(BaseModel):
    title: str
    description: str

class Step(BaseModel):
    step_number: int
    instruction: str

class ChatbotStructuredResponse(BaseModel):
    type: str
    content: Union[
        str,
        List[str],
        List[Plan],
        List[Feature],
        List[Step]
    ]

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    try:
        retriever = initialize_retriever()
        # Retrieve context
        related_docs = retriever.invoke(chat_request.query)
        
        if not related_docs:
            return {"response": "I could not find relevant information on the website."}

        context_text = "\n\n".join(doc.page_content for doc in related_docs)
        # Format prompt to Groq
        final_prompt = f"""
        You are a helpful assistant for Hotela.
        You have access to documents and website content.
        When answering the user's question:
        provide your answer in JSON with two fields:
        type: the type of response (plans, features, steps, comparison, etc).
        content: structured properly depending on the type.
        Only return valid JSON. You can include a short friendly introductory sentence before listing points inside 'content'.
        - If the context includes multiple points or features, you must always present them as a bullet-point list.
        - Do not combine points into long sentences.
        - Do not summarize into paragraphs.
        - Keep the original phrasing and structure as much as possible.
        - Be factual, direct, and concise.
        - Do not add any extra marketing language or storytelling.
        - Only use information from the context provided.
        - Speak naturally like a customer support agent.

        Here is the context you should use:
        {context_text}

        Question:
        {chat_request.query}

        Answer:"""

        # Send to Groq
        response = query_groq(final_prompt)

        # Parse response
        try:
            parsed_response = json.loads(response)
        except json.JSONDecodeError:
            # If chatbot returns normal text, wrap it manually
            parsed_response = {
                "type": "text",
                "content": response
            }

        validated_response = ChatbotStructuredResponse(**parsed_response)
        return validated_response.dict()

    except Exception as e:
        logging.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")