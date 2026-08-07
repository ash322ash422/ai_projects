import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def get_llm():
    llm = ChatOpenAI(
        model=AZURE_OPENAI_DEPLOYMENT,
        api_key=AZURE_OPENAI_KEY,
        base_url=AZURE_OPENAI_ENDPOINT,
        temperature=0
    )
    
    return llm

