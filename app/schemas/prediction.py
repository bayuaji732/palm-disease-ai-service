from pydantic import BaseModel, Field, field_validator
from typing import Optional
from app import config


class ClassificationResult(BaseModel):
    """Disease classification result from CV model."""
    
    label: str = Field(..., description="Detected disease class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    
    @field_validator('label')
    @classmethod
    def validate_label(cls, v: str) -> str:
        """Ensure label is from valid disease classes."""
        if v not in config.DISEASE_CLASSES:
            raise ValueError(
                f"Invalid label '{v}'. Must be one of: {config.DISEASE_CLASSES}"
            )
        return v


class ExplanationResponse(BaseModel):
    """Complete response with classification and LLM explanation."""
    
    classification: ClassificationResult
    explanation: str = Field(..., description="LLM-generated explanation and recommendations")
    confidence_level: str = Field(..., description="Human-readable confidence interpretation")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str


class ErrorResponse(BaseModel):
    """Error response model."""
    
    detail: str
    error_type: Optional[str] = None