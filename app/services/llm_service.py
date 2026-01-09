import ollama
import logging
from typing import Optional

from app import config
from app.utils.prompt_builder import build_explanation_prompt, build_fallback_explanation

logger = logging.getLogger(__name__)


class LLMService:
    """LLM service for generating disease explanations using Ollama."""
    
    def __init__(self):
        """Initialize LLM service."""
        self.client = ollama.Client(host=config.OLLAMA_HOST)
        self.model_name = config.OLLAMA_MODEL
        logger.info(f"LLM Service initialized. Model: {self.model_name}")
    
    def is_available(self) -> bool:
        """
        Check if Ollama service is available.
        
        Returns:
            True if Ollama is reachable, False otherwise
        """
        try:
            # Try to list models as health check
            self.client.list()
            return True
        except Exception as e:
            logger.warning(f"Ollama not available: {str(e)}")
            return False
    
    def generate_explanation(
        self, 
        label: str, 
        confidence: float,
        use_fallback: bool = False
    ) -> str:
        """
        Generate explanation for disease classification.
        
        Args:
            label: Detected disease class
            confidence: Prediction confidence (0-1)
            use_fallback: If True, skip LLM and use fallback
            
        Returns:
            Explanation text (from LLM or fallback)
        """
        # Check if we should use fallback
        if use_fallback or not self.is_available():
            logger.info("Using fallback explanation")
            return build_fallback_explanation(label, confidence)
        
        try:
            # Build prompt
            prompt = build_explanation_prompt(label, confidence)
            
            logger.info(f"Generating explanation for: {label}")
            
            # Call Ollama
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": 500,  # Max tokens
                }
            )
            
            explanation = response['response'].strip()
            
            # Validate response
            if not explanation or len(explanation) < 50:
                if len(explanation) > 5000:  # reasonable limit
                    explanation = explanation[:5000]
                logger.warning("LLM response too short, using fallback")
                return build_fallback_explanation(label, confidence)
            
            logger.info("Explanation generated successfully")
            return explanation
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return build_fallback_explanation(label, confidence)
    
    def generate_text(self, prompt: str) -> str:
        """
        Generic text generation method.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": config.LLM_TEMPERATURE,
                }
            )
            return response['response'].strip()
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            return f"[LLM Error: {str(e)}]"