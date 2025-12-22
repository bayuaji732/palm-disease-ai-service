import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # RBF_API_URL = os.getenv("RBF_API_URL")
    # RBF_API_KEY = os.getenv("RBF_API_KEY")
    # RBF_MODEL = os.getenv("RBF_MODEL")

    LLM_MODEL = os.getenv("OLLAMA_MODEL")
    TEMPERATURE = float(os.getenv("TEMPERATURE"))
    SYSTEM_PROMPT_HEALTHY = os.getenv("SYSTEM_PROMPT_HEALTHY")
    PROMPT_TEMPLATE_DISEASE = os.getenv("SYSTEM_PROMPT_DISEASE")