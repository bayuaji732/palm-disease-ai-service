from pydantic import BaseModel

class ClassificationResult(BaseModel):
    label: str
    confidence: float

class ExplanationResponse(BaseModel):
    classification: ClassificationResult
    explanation: str
