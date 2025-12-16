from fastapi import APIRouter
from app.services.cv_service import predict
from app.services.llm_service import generate_text
from app.utils.prompt_builder import build_explanation_prompt
from app.schemas.prediction import ClassificationResult, ExplanationResponse

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/predict-image", response_model=ClassificationResult)
def predict_image():
    """
    Endpoint CV only.
    """
    result = predict()
    return result


@router.post("/explain-result", response_model=ExplanationResponse)
def explain_result(result: ClassificationResult):
    """
    Endpoint LLM only.
    """
    prompt = build_explanation_prompt(
        label=result.label,
        confidence=result.confidence
    )

    explanation = generate_text(prompt)

    return {
        "classification": result,
        "explanation": explanation
    }
