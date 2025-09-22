"""
Basic tests for the multi-agent CoT system
"""
import pytest
import torch
from unittest.mock import Mock, patch

from multi_agent_cot.agents.base_agent import BaseAgent, Message, AgentResponse
from multi_agent_cot.agents.cot_agent import CoTAgent
from multi_agent_cot.orchestrator.coordinator import MultiAgentCoordinator, InteractionMode
from multi_agent_cot.models.model_loader import ModelLoader
from multi_agent_cot.evaluation.evaluator import MultiAgentEvaluator, BenchmarkTask
from multi_agent_cot.utils.config import ConfigManager


class TestBaseAgent:
    """Test the base agent functionality"""
    
    def test_message_creation(self):
        """Test message creation"""
        message = Message(
            content="Test content",
            sender="test_agent",
            message_type="reasoning",
            confidence=0.8
        )
        
        assert message.content == "Test content"
        assert message.sender == "test_agent"
        assert message.confidence == 0.8
        assert message.metadata == {}
    
    def test_agent_response_creation(self):
        """Test agent response creation"""
        message = Message("Test", "agent1")
        response = AgentResponse(
            message=message,
            reasoning_steps=["Step 1", "Step 2"],
            confidence=0.9,
            processing_time=1.5,
            token_count=100
        )
        
        assert response.message.content == "Test"
        assert len(response.reasoning_steps) == 2
        assert response.confidence == 0.9


class TestModelLoader:
    """Test the model loader"""
    
    def test_model_loader_init(self):
        """Test model loader initialization"""
        loader = ModelLoader()
        assert loader.device in ["cuda", "cpu"]
    
    def test_supported_models(self):
        """Test supported models list"""
        loader = ModelLoader()
        models = loader.list_supported_models()
        
        assert isinstance(models, list)
        assert "phi-2" in models
        assert "phi-1.5" in models
    
    def test_model_info(self):
        """Test getting model info"""
        loader = ModelLoader()
        info = loader.get_model_info("phi-2")
        
        assert info["name"] == "phi-2"
        assert info["supported"] == True
        assert "path" in info


class TestCoTAgent:
    """Test CoT agent functionality"""
    
    @patch('multi_agent_cot.agents.cot_agent.ModelLoader')
    def test_cot_agent_init(self, mock_loader):
        """Test CoT agent initialization"""
        # Mock the model loader to avoid actually loading models
        mock_loader.return_value.load_model.return_value = (Mock(), Mock())
        
        agent = CoTAgent(
            agent_id="test_agent",
            model_name="gpt2-medium",
            role="reasoner"
        )
        
        assert agent.agent_id == "test_agent"
        assert agent.model_name == "gpt2-medium"
        assert agent.role == "reasoner"
    
    def test_reasoning_steps_extraction(self):
        """Test reasoning steps extraction"""
        agent = CoTAgent.__new__(CoTAgent)  # Create without __init__
        
        text = """
        Step 1: Understanding the problem
        This is about analyzing the situation.
        
        Step 2: Finding the solution
        We need to calculate the result.
        
        Step 3: Verification
        Let's check our answer.
        """
        
        steps = agent._extract_reasoning_steps(text)
        assert len(steps) == 3
        assert "Step 1" in steps[0]
        assert "Step 2" in steps[1]
        assert "Step 3" in steps[2]
    
    def test_confidence_calculation(self):
        """Test confidence calculation"""
        agent = CoTAgent.__new__(CoTAgent)
        agent.reasoning_depth = 3
        
        reasoning_steps = [
            "Step 1: Therefore, we can conclude",
            "Step 2: Given that assumption, then",
            "Step 3: Since this is true, thus"
        ]
        
        confidence = agent._calculate_confidence(reasoning_steps)
        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Should be reasonably high due to reasoning keywords


class TestMultiAgentCoordinator:
    """Test the multi-agent coordinator"""
    
    def test_coordinator_init(self):
        """Test coordinator initialization"""
        # Create mock agents
        agents = [
            Mock(spec=BaseAgent, agent_id="agent1", is_active=True),
            Mock(spec=BaseAgent, agent_id="agent2", is_active=True)
        ]
        
        coordinator = MultiAgentCoordinator(
            agents=agents,
            max_rounds=3,
            consensus_threshold=0.7
        )
        
        assert len(coordinator.agents) == 2
        assert coordinator.max_rounds == 3
        assert coordinator.consensus_threshold == 0.7
    
    def test_consensus_calculation(self):
        """Test consensus score calculation"""
        agents = [Mock(spec=BaseAgent, agent_id="agent1", is_active=True)]
        coordinator = MultiAgentCoordinator(agents)
        
        # Create mock responses
        responses = [
            Mock(confidence=0.8, message=Mock(content="Answer 1")),
            Mock(confidence=0.9, message=Mock(content="Answer 2"))
        ]
        
        consensus = coordinator._calculate_consensus(responses)
        assert 0 <= consensus <= 1
    
    def test_add_remove_agents(self):
        """Test adding and removing agents"""
        coordinator = MultiAgentCoordinator(agents=[])
        
        agent = Mock(spec=BaseAgent, agent_id="test_agent")
        coordinator.add_agent(agent)
        
        assert "test_agent" in coordinator.agents
        
        coordinator.remove_agent("test_agent")
        assert "test_agent" not in coordinator.agents


class TestEvaluator:
    """Test the evaluation framework"""
    
    def test_evaluator_init(self):
        """Test evaluator initialization"""
        evaluator = MultiAgentEvaluator()
        assert evaluator.evaluation_history == []
    
    def test_benchmark_task_creation(self):
        """Test benchmark task creation"""
        evaluator = MultiAgentEvaluator()
        tasks = evaluator.create_benchmark_tasks()
        
        assert len(tasks) > 0
        assert all(isinstance(task, BenchmarkTask) for task in tasks)
        assert all(hasattr(task, 'question') for task in tasks)
        assert all(hasattr(task, 'expected_answer') for task in tasks)
    
    def test_accuracy_calculation(self):
        """Test accuracy calculation"""
        evaluator = MultiAgentEvaluator()
        
        # Exact match
        accuracy = evaluator._calculate_accuracy("36", "36")
        assert accuracy == 1.0
        
        # Partial match
        accuracy = evaluator._calculate_accuracy("The answer is 36", "36")
        assert accuracy == 0.8
        
        # No match
        accuracy = evaluator._calculate_accuracy("42", "36")
        assert accuracy < 0.5


class TestConfigManager:
    """Test configuration management"""
    
    def test_config_manager_init(self):
        """Test config manager initialization"""
        config_manager = ConfigManager()
        assert config_manager.config is not None
    
    def test_default_config_creation(self):
        """Test default configuration creation"""
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        assert len(config.models) > 0
        assert len(config.agents) > 0
        assert config.coordinator.max_rounds > 0
        assert 0 <= config.coordinator.consensus_threshold <= 1
    
    def test_config_validation(self):
        """Test configuration validation"""
        config_manager = ConfigManager()
        issues = config_manager.validate_config()
        
        # Default config should be valid
        assert len(issues) == 0


class TestInteractionModes:
    """Test different interaction modes"""
    
    def test_interaction_mode_enum(self):
        """Test interaction mode enumeration"""
        assert InteractionMode.ADVERSARIAL.value == "adversarial"
        assert InteractionMode.COLLABORATIVE.value == "collaborative"
        assert InteractionMode.DEBATE.value == "debate"
        assert InteractionMode.CONSENSUS.value == "consensus"
        assert InteractionMode.CRITIQUE_IMPROVE.value == "critique_improve"


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])