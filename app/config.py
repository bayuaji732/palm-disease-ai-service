import os


# Model Settings
MODEL_PATH = os.getenv("MODEL_PATH")
CONFIDENCE_THRESHOLD = os.getenv("CONFIDENCE_THRESHOLD")
IOU_THRESHOLD = os.getenv("IOU_THRESHOLD")
IMAGE_SIZE = os.getenv("IMAGE_SIZE")

# LLM Settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
LLM_TIMEOUT = os.getenv("LLM_TIMEOUT")
LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE")

# File Upload Settings
MAX_UPLOAD_SIZE = os.getenv("MAX_UPLOAD_SIZE")
ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS")
# Disease Classes (Fixed Dataset)
DISEASE_CLASSES = os.getenv("DISEASE_CLASSSES")