"""
Chain-of-Thought agent implementation
"""
import time
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from multi_agent_cot.agents.base_agent import BaseAgent, Message, AgentResponse
from multi_agent_cot.models.model_loader import ModelLoader
import logging

logger = logging.getLogger(__name__)


class CoTAgent(BaseAgent):
    """Agent with Chain-of-Thought reasoning capabilities"""
    
    def __init__(
        self,
        agent_id: str,
        model_name: str,
        role: str = "reasoner",
        max_tokens: int = 512,
        temperature: float = 0.7,
        reasoning_depth: int = 3,
        use_self_critique: bool = True,
        **kwargs
    ):
        """Initialize CoT Agent
        
        Args:
            agent_id: Unique identifier for this agent
            model_name: Name of the language model to use
            role: Role/specialty of the agent
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            reasoning_depth: Number of reasoning steps to perform
            use_self_critique: Whether to self-critique responses
            **kwargs: Additional parameters
        """
        super().__init__(
            agent_id=agent_id,
            model_name=model_name,
            role=role,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_depth=reasoning_depth,
            **kwargs
        )
        
        self.use_self_critique = use_self_critique
        self.model_loader = ModelLoader()
        self.model = None
        self.tokenizer = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the language model and tokenizer"""
        try:
            self.model, self.tokenizer = self.model_loader.load_model(
                self.model_name,
                load_in_8bit=True  # For memory efficiency with small models
            )
            logger.info(f"Agent {self.agent_id} loaded model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def _create_cot_prompt(
        self, 
        prompt: str, 
        context: List[Message] = None,
        task_type: str = "reasoning"
    ) -> str:
        """Create a Chain-of-Thought prompt template
        
        Args:
            prompt: The original prompt
            context: Previous conversation context
            task_type: Type of task (reasoning, critique, synthesis)
            
        Returns:
            Formatted CoT prompt
        """
        cot_template = {
            "reasoning": """You are an expert reasoning agent. Think step by step to solve problems.

{context}

Question: {prompt}

Let me think through this step by step:

Step 1: Understanding the problem
""",
            "critique": """You are a critical thinking expert. Analyze the given response carefully.

{context}

Original Question: {original_prompt}
Response to Critique: {response_content}

Let me analyze this response step by step:

Step 1: Understanding what was claimed
""",
            "synthesis": """You are a synthesis expert. Combine different perspectives into a coherent answer.

{context}

Question: {prompt}

Let me synthesize the available information step by step:

Step 1: Identifying key points
"""
        }
        
        template = cot_template.get(task_type, cot_template["reasoning"])
        
        # Format context
        context_str = ""
        if context:
            context_str = self.format_conversation_history()
        
        return template.format(
            context=context_str,
            prompt=prompt,
            original_prompt=getattr(self, '_current_prompt', prompt),
            response_content=getattr(self, '_current_response', "")
        )
    
    def _generate_text(self, prompt: str, **kwargs) -> tuple[str, int]:
        """Generate text using the loaded model
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Tuple of (generated_text, token_count)
        """
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        
        if self.model.device != inputs.device:
            inputs = inputs.to(self.model.device)
        
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "attention_mask": torch.ones_like(inputs)
        }
        
        with torch.no_grad():
            outputs = self.model.generate(inputs, **generation_kwargs)
        
        # Decode only the new tokens
        new_tokens = outputs[0][len(inputs[0]):]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return generated_text, len(new_tokens)
    
    def _extract_reasoning_steps(self, generated_text: str) -> List[str]:
        """Extract reasoning steps from generated text
        
        Args:
            generated_text: The generated response text
            
        Returns:
            List of reasoning steps
        """
        steps = []
        lines = generated_text.split('\n')
        
        current_step = ""
        for line in lines:
            line = line.strip()
            if line.startswith("Step ") or line.startswith("step "):
                if current_step:
                    steps.append(current_step.strip())
                current_step = line
            elif current_step and line:
                current_step += " " + line
        
        if current_step:
            steps.append(current_step.strip())
        
        return steps if steps else [generated_text.strip()]
    
    def generate_response(
        self, 
        prompt: str, 
        context: List[Message] = None,
        **kwargs
    ) -> AgentResponse:
        """Generate a Chain-of-Thought response
        
        Args:
            prompt: The input prompt/question
            context: Previous messages in the conversation
            **kwargs: Additional generation parameters
            
        Returns:
            AgentResponse with CoT reasoning
        """
        start_time = time.time()
        
        # Create CoT prompt
        cot_prompt = self._create_cot_prompt(prompt, context, "reasoning")
        
        # Generate response
        generated_text, token_count = self._generate_text(cot_prompt, **kwargs)
        
        # Extract reasoning steps
        reasoning_steps = self._extract_reasoning_steps(generated_text)
        
        # Calculate confidence based on reasoning consistency
        confidence = self._calculate_confidence(reasoning_steps)
        
        # Create message
        message = Message(
            content=generated_text,
            sender=self.agent_id,
            message_type="reasoning",
            confidence=confidence,
            metadata={"reasoning_steps": len(reasoning_steps)}
        )
        
        # Create response
        response = AgentResponse(
            message=message,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            processing_time=time.time() - start_time,
            token_count=token_count
        )
        
        # Add to history
        self.add_to_history(message)
        
        # Self-critique if enabled
        if self.use_self_critique:
            critique = self._self_critique(response, prompt)
            if critique:
                response.metadata = response.message.metadata.copy()
                response.metadata["self_critique"] = critique
        
        logger.info(f"Agent {self.agent_id} generated response with {len(reasoning_steps)} reasoning steps")
        
        return response
    
    def critique_response(
        self, 
        response: AgentResponse, 
        original_prompt: str,
        **kwargs
    ) -> AgentResponse:
        """Critique another agent's response
        
        Args:
            response: The response to critique
            original_prompt: The original prompt
            **kwargs: Additional parameters
            
        Returns:
            AgentResponse containing the critique
        """
        start_time = time.time()
        
        # Store for template formatting
        self._current_prompt = original_prompt
        self._current_response = response.message.content
        
        # Create critique prompt
        critique_prompt = self._create_cot_prompt(
            original_prompt, 
            context=None, 
            task_type="critique"
        )
        
        # Generate critique
        generated_text, token_count = self._generate_text(critique_prompt, **kwargs)
        
        # Extract reasoning steps
        reasoning_steps = self._extract_reasoning_steps(generated_text)
        
        # Calculate confidence
        confidence = self._calculate_confidence(reasoning_steps)
        
        # Create critique message
        message = Message(
            content=generated_text,
            sender=self.agent_id,
            message_type="critique",
            confidence=confidence,
            metadata={
                "target_agent": response.message.sender,
                "critique_type": "adversarial"
            }
        )
        
        critique_response = AgentResponse(
            message=message,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            processing_time=time.time() - start_time,
            token_count=token_count
        )
        
        self.add_to_history(message)
        
        logger.info(f"Agent {self.agent_id} critiqued response from {response.message.sender}")
        
        return critique_response
    
    def _calculate_confidence(self, reasoning_steps: List[str]) -> float:
        """Calculate confidence based on reasoning quality
        
        Args:
            reasoning_steps: List of reasoning steps
            
        Returns:
            Confidence score between 0 and 1
        """
        if not reasoning_steps:
            return 0.5
        
        # Simple heuristic: more steps and specific keywords increase confidence
        confidence_factors = []
        
        # Number of steps factor
        step_factor = min(len(reasoning_steps) / self.reasoning_depth, 1.0)
        confidence_factors.append(step_factor)
        
        # Content quality factor (look for reasoning keywords)
        reasoning_keywords = ["because", "therefore", "thus", "since", "given", "assuming", "if", "then"]
        total_keywords = 0
        total_words = 0
        
        for step in reasoning_steps:
            words = step.lower().split()
            total_words += len(words)
            total_keywords += sum(1 for word in words if word in reasoning_keywords)
        
        if total_words > 0:
            keyword_density = total_keywords / total_words
            confidence_factors.append(min(keyword_density * 10, 1.0))  # Scale appropriately
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _self_critique(self, response: AgentResponse, original_prompt: str) -> Optional[str]:
        """Perform self-critique on generated response
        
        Args:
            response: The response to self-critique
            original_prompt: Original prompt
            
        Returns:
            Self-critique text or None
        """
        try:
            # Simple self-critique: check if response addresses the prompt
            if len(response.reasoning_steps) < 2:
                return "Response lacks sufficient reasoning depth"
            
            if response.confidence < 0.3:
                return "Low confidence in reasoning quality"
            
            return None  # No critique needed
            
        except Exception as e:
            logger.warning(f"Self-critique failed: {e}")
            return None