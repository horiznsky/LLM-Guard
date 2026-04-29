import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY1")
HF_TOKEN = os.getenv("HF_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
# RUNPOD_OLLAMA_URL = os.getenv("RUNPOD_OLLAMA_URL")
RUNPOD_OLLAMA_URL = "https://dgfjfu5f5ijksr-11434.proxy.runpod.net/v1/"