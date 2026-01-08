def get_confidence_level(confidence: float) -> str:
    """
    Convert numerical confidence to human-readable level.
    
    Args:
        confidence: Float between 0 and 1
        
    Returns:
        Confidence level description
    """
    if confidence >= 0.85:
        return "Very likely"
    elif confidence >= 0.70:
        return "Most likely"
    elif confidence >= 0.50:
        return "Possible"
    else:
        return "Uncertain, needs verification"


def build_explanation_prompt(label: str, confidence: float) -> str:
    """
    Build structured prompt for LLM explanation.
    
    Args:
        label: Detected disease class
        confidence: Prediction confidence (0-1)
        
    Returns:
        Formatted prompt string for LLM
    """
    confidence_level = get_confidence_level(confidence)
    confidence_pct = confidence * 100
    
    prompt = f"""You are an experienced oil palm agronomist. Please provide a professional explanation of the following conditions:
    DETECTION RESULTS:
    - Condition: {label}
    - Confidence Level: {confidence_pct:.1f}% ({confidence_level})

    INSTRUCTIONS:
    Provide responses in the following format (ENGLISH):

    1. CONDITION EXPLANATION (2-3 short sentences):
    - Explain what {label} is
    - List common causes

    2. RECOMMENDED ACTION (3-4 points):
    - Immediate action to be taken
    - Treatment steps
    - DO NOT mention specific chemical dosages or trademarks

    3. PREVENTION (2-3 points):
    - How to prevent this condition in the future

    IMPORTANT:
    - Use language that farmers can easily understand
    - Do not provide information outside the palm oil domain
    - If confidence is low (<50%), include a recommendation for expert consultation
    - Focus on best practices, not commercial products

    Answer professionally and practically."""
    
    return prompt


def build_fallback_explanation(label: str, confidence: float) -> str:
    """
    Generate fallback explanation when LLM fails.
    
    Args:
        label: Detected disease class
        confidence: Prediction confidence
        
    Returns:
        Safe fallback explanation
    """
    confidence_level = get_confidence_level(confidence)
    
    return f"""Detected condition: {label}
Confidence level: {confidence*100:.1f}% ({confidence_level})

Automatic explanation is currently unavailable. Please consult an agronomist or field officer for further analysis.

For general information about {label}, refer to the oil palm cultivation technical guide or contact your nearest agricultural consulting service."""