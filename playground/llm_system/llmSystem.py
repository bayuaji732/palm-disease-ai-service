import ollama
from config import Config

def generate_text(disease: str) -> str:
    """
    Generate Content
    """
    if disease.lower() == "healthy":
        prompt = Config.SYSTEM_PROMPT_HEALTHY
    else:
        prompt = Config.PROMPT_TEMPLATE_DISEASE.format(disease=disease)

    response = ollama.chat(
        model=Config.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": prompt
            }
        ],
        options={
            "temperature": Config.TEMPERATURE
        }
    )
    return response["message"]["content"]

## Execution
if __name__ == "__main__":
    disease = "magnesium deficiency"
    result = generate_text(disease)
    print(result)
