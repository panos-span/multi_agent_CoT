"""
Base agent class for the multi-agent CoT system
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a message in the multi-agent conversation"""
    content: str
    sender: str
    message_type: str = "reasoning"  # reasoning, critique, question, answer
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class AgentResponse:
    """Response from an agent including reasoning and metadata"""
    message: Message
    reasoning_steps: List[str]
    confidence: float
    processing_time: float
    token_count: int = 0


class BaseAgent(ABC):
    """Abstract base class for all agents in the multi-agent CoT system"""
    
    def __init__(
        self, 
        agent_id: str,
        model_name: str,
        role: str = "general",
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ):
        """Initialize the base agent
        
        Args:
            agent_id: Unique identifier for this agent
            model_name: Name of the language model to use
            role: Role/specialty of the agent (e.g., "critic", "synthesizer", "specialist")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional agent-specific parameters
        """
        self.agent_id = agent_id
        self.model_name = model_name
        self.role = role
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Conversation history
        self.conversation_history: List[Message] = []
        
        # Agent state
        self.is_active = True
        self.reasoning_depth = kwargs.get("reasoning_depth", 3)
        
        logger.info(f"Initialized {self.__class__.__name__} with ID: {agent_id}, Role: {role}")
    
    @abstractmethod
    def generate_response(
        self, 
        prompt: str, 
        context: List[Message] = None,
        **kwargs
    ) -> AgentResponse:
        """Generate a response to the given prompt with context
        
        Args:
            prompt: The input prompt/question
            context: Previous messages in the conversation
            **kwargs: Additional generation parameters
            
        Returns:
            AgentResponse containing the generated response and metadata
        """
        pass
    
    @abstractmethod
    def critique_response(
        self, 
        response: AgentResponse, 
        original_prompt: str,
        **kwargs
    ) -> AgentResponse:
        """Critique another agent's response
        
        Args:
            response: The response to critique
            original_prompt: The original prompt that generated the response
            **kwargs: Additional critique parameters
            
        Returns:
            AgentResponse containing the critique
        """
        pass
    
    def add_to_history(self, message: Message):
        """Add a message to the conversation history"""
        self.conversation_history.append(message)
        logger.debug(f"Agent {self.agent_id} added message to history from {message.sender}")
    
    def get_conversation_context(self, max_messages: int = 10) -> List[Message]:
        """Get recent conversation context
        
        Args:
            max_messages: Maximum number of recent messages to return
            
        Returns:
            List of recent messages
        """
        return self.conversation_history[-max_messages:]
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history.clear()
        logger.info(f"Agent {self.agent_id} cleared conversation history")
    
    def set_active(self, active: bool):
        """Set the agent's active status"""
        self.is_active = active
        logger.info(f"Agent {self.agent_id} active status set to: {active}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "agent_id": self.agent_id,
            "model_name": self.model_name,
            "role": self.role,
            "is_active": self.is_active,
            "conversation_length": len(self.conversation_history),
            "reasoning_depth": self.reasoning_depth
        }
    
    def format_conversation_history(self, max_messages: int = 5) -> str:
        """Format conversation history as a string for model input"""
        recent_messages = self.get_conversation_context(max_messages)
        
        formatted = "Previous conversation:\n"
        for msg in recent_messages:
            formatted += f"{msg.sender}: {msg.content}\n"
        
        return formatted
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, role={self.role}, model={self.model_name})"