##########################################

from langchain_openai import AzureOpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

def get_azure_embedding_model():
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

def get_gemini_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")