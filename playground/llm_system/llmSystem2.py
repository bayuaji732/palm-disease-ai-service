from config import Config
import ollama

# Prompt Template
def build_explanation_prompt(label: str, confidence: float) -> str:
    if label.lower() == "healthy":
        return Config.SYSTEM_PROMPT_HEALTHY
    else:
        return Config.PROMPT_TEMPLATE_DISEASE.format(disease=label)

# Generate Content - Ollama
def generate_text(prompt: str) -> str:
    """
    Mock LLM response.
    """
    response = ollama.chat(
        model=Config.LLM_MODEL, 
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": Config.TEMPERATURE,}
        )
    return response["message"]["content"]

# Result
def explain_result(label: str, confidence: float) -> dict:
    """
    Endpoint LLM only.
    """
    prompt = build_explanation_prompt(
        label=label,
        confidence=confidence
    )

    explanation = generate_text(prompt)

    return {
        "classification": {
            "label": label,
            "confidence": confidence
        },
        "explanation": explanation
    }

# Example usage
if __name__ == "__main__":
    # Example usage
    explanation_response = explain_result(label="Magnesium Defficiency", confidence=0.86)
    print(explanation_response)