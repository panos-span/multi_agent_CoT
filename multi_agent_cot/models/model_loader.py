"""
Model loading utilities for small language models (1-3B parameters)
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Utility class for loading and managing small language models"""
    
    SUPPORTED_MODELS = {
        "phi-2": "microsoft/phi-2",
        "phi-1.5": "microsoft/phi-1_5", 
        "stablelm-3b": "stabilityai/stablelm-3b-4e1t",
        "pythia-2.8b": "EleutherAI/pythia-2.8b",
        "gpt2-medium": "gpt2-medium",
        "gpt2-large": "gpt2-large",
        "llama2-chat-7b": "meta-llama/Llama-2-7b-chat-hf",  # Slightly larger but still efficient
    }
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the model loader
        
        Args:
            device: Device to load models on ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"ModelLoader initialized with device: {self.device}")
    
    def load_model(
        self, 
        model_name: str,
        load_in_8bit: bool = False,
        trust_remote_code: bool = True,
        **kwargs
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load a model and tokenizer
        
        Args:
            model_name: Name of the model to load (from SUPPORTED_MODELS or HF model path)
            load_in_8bit: Whether to load model in 8-bit precision for memory efficiency
            trust_remote_code: Whether to trust remote code for custom models
            **kwargs: Additional arguments passed to model loading
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model_path = self.SUPPORTED_MODELS.get(model_name, model_name)
        
        logger.info(f"Loading model: {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Model loading arguments
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            **kwargs
        }
        
        if load_in_8bit and self.device == "cuda":
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = self.device
            
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        logger.info(f"Model loaded successfully. Parameters: {self.count_parameters(model):,}")
        
        return model, tokenizer
    
    def count_parameters(self, model: torch.nn.Module) -> int:
        """Count the number of parameters in a model"""
        return sum(p.numel() for p in model.parameters())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a supported model"""
        model_path = self.SUPPORTED_MODELS.get(model_name, model_name)
        
        return {
            "name": model_name,
            "path": model_path,
            "supported": model_name in self.SUPPORTED_MODELS,
            "device": self.device
        }
    
    def list_supported_models(self) -> list[str]:
        """List all supported model names"""
        return list(self.SUPPORTED_MODELS.keys())