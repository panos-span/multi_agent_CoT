"""
Configuration management for multi-agent CoT system
"""
import os
import json
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a language model"""
    name: str
    model_path: str
    max_tokens: int = 512
    temperature: float = 0.7
    load_in_8bit: bool = True
    trust_remote_code: bool = True


@dataclass
class AgentConfig:
    """Configuration for an agent"""
    agent_id: str
    model_name: str
    role: str = "general"
    reasoning_depth: int = 3
    use_self_critique: bool = True
    max_tokens: int = 512
    temperature: float = 0.7


@dataclass
class CoordinatorConfig:
    """Configuration for the multi-agent coordinator"""
    max_rounds: int = 5
    consensus_threshold: float = 0.8
    enable_adversarial: bool = True
    default_interaction_mode: str = "adversarial"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    benchmark_tasks_file: Optional[str] = None
    output_dir: str = "./results"
    save_detailed_results: bool = True
    compare_with_baseline: bool = False
    baseline_results_file: Optional[str] = None


@dataclass
class SystemConfig:
    """Complete system configuration"""
    models: List[ModelConfig]
    agents: List[AgentConfig]
    coordinator: CoordinatorConfig
    evaluation: EvaluationConfig
    logging_level: str = "INFO"
    device: Optional[str] = None
    random_seed: Optional[int] = 42


class ConfigManager:
    """Manages configuration for the multi-agent CoT system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        self.config_path = config_path
        self.config: Optional[SystemConfig] = None
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.config = self._create_default_config()
    
    def load_config(self, config_path: str) -> SystemConfig:
        """Load configuration from file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded system configuration
        """
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
            
            self.config = self._dict_to_config(config_dict)
            logger.info(f"Configuration loaded from {config_path}")
            
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            logger.info("Using default configuration")
            self.config = self._create_default_config()
            return self.config
    
    def save_config(self, config_path: str, config: Optional[SystemConfig] = None):
        """Save configuration to file
        
        Args:
            config_path: Path to save configuration
            config: Configuration to save (uses current config if None)
        """
        if config is None:
            config = self.config
        
        if config is None:
            logger.error("No configuration to save")
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            config_dict = self._config_to_dict(config)
            
            with open(config_path, 'w') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
    
    def get_config(self) -> SystemConfig:
        """Get current configuration"""
        if self.config is None:
            self.config = self._create_default_config()
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration with new values"""
        if self.config is None:
            self.config = self._create_default_config()
        
        # Update fields that exist
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config.{key} = {value}")
    
    def _create_default_config(self) -> SystemConfig:
        """Create default system configuration"""
        
        # Default models
        models = [
            ModelConfig(
                name="phi-2",
                model_path="microsoft/phi-2",
                max_tokens=512,
                temperature=0.7,
                load_in_8bit=True
            ),
            ModelConfig(
                name="phi-1.5", 
                model_path="microsoft/phi-1_5",
                max_tokens=512,
                temperature=0.8,
                load_in_8bit=True
            )
        ]
        
        # Default agents
        agents = [
            AgentConfig(
                agent_id="reasoner_1",
                model_name="phi-2",
                role="primary_reasoner",
                reasoning_depth=3,
                temperature=0.7
            ),
            AgentConfig(
                agent_id="critic_1",
                model_name="phi-1.5",
                role="critic",
                reasoning_depth=2,
                temperature=0.8,
                use_self_critique=False
            ),
            AgentConfig(
                agent_id="synthesizer_1",
                model_name="phi-2",
                role="synthesizer", 
                reasoning_depth=4,
                temperature=0.6
            )
        ]
        
        # Default coordinator
        coordinator = CoordinatorConfig(
            max_rounds=5,
            consensus_threshold=0.8,
            enable_adversarial=True,
            default_interaction_mode="adversarial"
        )
        
        # Default evaluation
        evaluation = EvaluationConfig(
            output_dir="./results",
            save_detailed_results=True,
            compare_with_baseline=False
        )
        
        return SystemConfig(
            models=models,
            agents=agents,
            coordinator=coordinator,
            evaluation=evaluation,
            logging_level="INFO",
            device=None,
            random_seed=42
        )
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to SystemConfig"""
        
        # Convert models
        models = []
        for model_dict in config_dict.get("models", []):
            models.append(ModelConfig(**model_dict))
        
        # Convert agents
        agents = []
        for agent_dict in config_dict.get("agents", []):
            agents.append(AgentConfig(**agent_dict))
        
        # Convert coordinator
        coordinator_dict = config_dict.get("coordinator", {})
        coordinator = CoordinatorConfig(**coordinator_dict)
        
        # Convert evaluation
        evaluation_dict = config_dict.get("evaluation", {})
        evaluation = EvaluationConfig(**evaluation_dict)
        
        return SystemConfig(
            models=models,
            agents=agents,
            coordinator=coordinator,
            evaluation=evaluation,
            logging_level=config_dict.get("logging_level", "INFO"),
            device=config_dict.get("device"),
            random_seed=config_dict.get("random_seed", 42)
        )
    
    def _config_to_dict(self, config: SystemConfig) -> Dict[str, Any]:
        """Convert SystemConfig to dictionary"""
        return {
            "models": [asdict(model) for model in config.models],
            "agents": [asdict(agent) for agent in config.agents],
            "coordinator": asdict(config.coordinator),
            "evaluation": asdict(config.evaluation),
            "logging_level": config.logging_level,
            "device": config.device,
            "random_seed": config.random_seed
        }
    
    def create_example_config(self, output_path: str):
        """Create an example configuration file"""
        example_config = self._create_default_config()
        self.save_config(output_path, example_config)
        logger.info(f"Example configuration created at {output_path}")
    
    def validate_config(self, config: Optional[SystemConfig] = None) -> List[str]:
        """Validate configuration and return list of issues
        
        Args:
            config: Configuration to validate (uses current if None)
            
        Returns:
            List of validation issues (empty if valid)
        """
        if config is None:
            config = self.config
        
        if config is None:
            return ["No configuration available"]
        
        issues = []
        
        # Validate models
        if not config.models:
            issues.append("No models configured")
        
        # Validate agents
        if not config.agents:
            issues.append("No agents configured")
        else:
            model_names = {model.name for model in config.models}
            for agent in config.agents:
                if agent.model_name not in model_names:
                    issues.append(f"Agent {agent.agent_id} references unknown model {agent.model_name}")
        
        # Validate coordinator settings
        if config.coordinator.max_rounds <= 0:
            issues.append("max_rounds must be positive")
        
        if not 0 <= config.coordinator.consensus_threshold <= 1:
            issues.append("consensus_threshold must be between 0 and 1")
        
        # Validate evaluation settings
        if config.evaluation.output_dir and not os.path.exists(os.path.dirname(config.evaluation.output_dir)):
            try:
                os.makedirs(os.path.dirname(config.evaluation.output_dir), exist_ok=True)
            except:
                issues.append(f"Cannot create output directory: {config.evaluation.output_dir}")
        
        return issues


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('multi_agent_cot.log')
        ]
    )
    
    # Reduce transformers logging
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)