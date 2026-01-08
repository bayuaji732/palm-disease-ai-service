import os
from dotenv import load_dotenv

load_dotenv()

# Model Settings
MODEL_PATH = os.getenv("MODEL_PATH")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.25))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.45))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 640))

# LLM Settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE"))

# File Upload Settings
MAX_UPLOAD_SIZE = 10485760
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]
# Disease Classes (Fixed Dataset)
DISEASE_CLASSES = [
        "Black Scorch",
        "Fusarium Wilt",
        "Healthy sample",
        "Leaf Spots",
        "Magnesium Deficiency",
        "Manganese Deficiency",
        "Parlatoria Blanchardi",
        "Potassium Deficiency",
        "Rachis Blight"
    ]