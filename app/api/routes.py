from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging

from app.services import cv_service, llm_service
from app.schemas.prediction import (
    ClassificationResult,
    ExplanationResponse,
    HealthResponse,
    ErrorResponse
)
from app.utils.prompt_builder import get_confidence_level
from app import config

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify service status.
    
    Returns:
        HealthResponse with service status
    """
    return HealthResponse(
        status="ok"
    )


@router.post("/predict-image",response_model=ClassificationResult, responses={422: {"model": ErrorResponse},500: {"model": ErrorResponse}})
async def predict_image(
    file: UploadFile = File(..., description="Image file (JPG/PNG)")
):
    """
    Computer Vision endpoint - classification only.
    
    Args:
        file: Uploaded image file
        
    Returns:
        ClassificationResult with label and confidence
        
    Raises:
        HTTPException: 422 for invalid input, 500 for processing errors
    """
    try:
        # Validate file extension
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid file type. Allowed: {config.ALLOWED_EXTENSIONS}"
            )
        
        # Read and validate image
        contents = await file.read()
        
        # Check file size
        if len(contents) > config.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"File too large. Max size: {config.MAX_UPLOAD_SIZE/1024/1024}MB"
            )
        
        # Open image
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid image file: {str(e)}"
            )
        cv_service_instace = cv_service.CVService()
        # Validate image
        if not cv_service_instace.validate_image(image):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Image validation failed. Check format and dimensions."
            )
        
        # Perform prediction
        cv_service_instace.load_model()
        result = cv_service_instace.predict(image)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/explain-result", response_model=ExplanationResponse, responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def explain_result(result: ClassificationResult):
    """
    LLM endpoint - explanation only.
    
    Args:
        result: ClassificationResult to explain
        
    Returns:
        ExplanationResponse with classification and explanation
        
    Raises:
        HTTPException: 422 for invalid input, 500 for processing errors
    """
    try:
        # Generate explanation
        llm_service_instance = llm_service.LLMService()
        explanation = llm_service_instance.generate_explanation(
            label=result.label,
            confidence=result.confidence
        )
        
        # Get confidence level
        confidence_level = get_confidence_level(result.confidence)
        
        return ExplanationResponse(
            classification=result,
            explanation=explanation,
            confidence_level=confidence_level
        )
        
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation generation failed: {str(e)}"
        )


@router.post("/predict-and-explain", response_model=ExplanationResponse, responses={422: {"model": ErrorResponse},500: {"model": ErrorResponse}})
async def predict_and_explain(
    file: UploadFile = File(..., description="Image file (JPG/PNG)")
):
    """
    Combined endpoint - CV + LLM pipeline.
    
    This is the recommended endpoint for complete analysis.
    
    Args:
        file: Uploaded image file
        
    Returns:
        ExplanationResponse with classification and explanation
        
    Raises:
        HTTPException: 422 for invalid input, 500 for processing errors
    """
    try:
        # Step 1: CV Prediction
        logger.info("Starting CV prediction")
        file.file.seek(0)
        classification = await predict_image(file)
        
        # Step 2: LLM Explanation
        logger.info("Generating LLM explanation")
        llm_service_instance = llm_service.LLMService()
        explanation = llm_service_instance.generate_explanation(
            label=classification.label,
            confidence=classification.confidence
        )
        
        # Get confidence level
        confidence_level = get_confidence_level(classification.confidence)
        
        return ExplanationResponse(
            classification=classification,
            explanation=explanation,
            confidence_level=confidence_level
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {str(e)}"
        )