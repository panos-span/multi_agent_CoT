"""
Multi-Agent Chain-of-Thought System for Small Language Models

This package implements a multi-agent adversarial CoT system designed to help 
small language models (1-3B parameters) match or surpass larger models through
collaborative reasoning and adversarial interactions.
"""

__version__ = "0.1.0"
__author__ = "Panos Span"

from multi_agent_cot.agents.base_agent import BaseAgent
from multi_agent_cot.agents.cot_agent import CoTAgent
from multi_agent_cot.orchestrator.coordinator import MultiAgentCoordinator
from multi_agent_cot.models.model_loader import ModelLoader

__all__ = [
    "BaseAgent",
    "CoTAgent", 
    "MultiAgentCoordinator",
    "ModelLoader",
]