import ollama

def generate_text(prompt: str) -> str:
    """
    Mock LLM response.
    """
    response = ollama.chat(
        model="gemma3:1b", 
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2,}
        )
    return response["message"]["content"]