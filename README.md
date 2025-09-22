# Multi-Agent Adversarial Chain-of-Thought (SLMs)

A research framework for multi-agent Chain-of-Thought systems using small language models (1-3B parameters) that can match or surpass larger models through collaborative and adversarial reasoning.

## 🎯 Research Goal

**Thesis Project for MSc in ML**: Design and justify a multi-agent Chain-of-Thought system of small LMs (≈1–3B) that matches/surpasses a single larger model under similar or lower compute/cost.

## 🏗️ System Architecture

```
Multi-Agent CoT System
├── Agents (Small LMs: 1-3B params)
│   ├── CoT Agent (Chain-of-Thought reasoning)
│   ├── Critic Agent (Adversarial analysis)
│   └── Synthesizer Agent (Consensus building)
├── Orchestrator
│   ├── Interaction Modes (Adversarial, Collaborative, Debate)
│   ├── Consensus Management
│   └── Round-based Coordination
├── Evaluation Framework
│   ├── Benchmark Tasks
│   ├── Performance Metrics
│   └── Baseline Comparisons
└── Model Management
    ├── Small Model Loading (Phi-2, Phi-1.5, etc.)
    ├── Memory Optimization
    └── Efficient Inference
```

## 🚀 Key Features

- **Multi-Agent Reasoning**: Multiple small LMs working together with different roles
- **Adversarial Interactions**: Agents critique and challenge each other's responses
- **Chain-of-Thought**: Step-by-step reasoning with explicit thought processes
- **Flexible Orchestration**: Different interaction modes (adversarial, collaborative, debate, consensus)
- **Memory Efficient**: Optimized for 1-3B parameter models with 8-bit quantization
- **Comprehensive Evaluation**: Built-in benchmarking and comparison framework
- **Research-Ready**: Designed for academic research and experimentation

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/panos-span/multi_agent_CoT.git
cd multi_agent_CoT

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## 🎮 Quick Start

### Basic Usage

```python
from multi_agent_cot.agents.cot_agent import CoTAgent
from multi_agent_cot.orchestrator.coordinator import MultiAgentCoordinator, InteractionMode

# Create agents with different small models
agents = [
    CoTAgent(
        agent_id="reasoner",
        model_name="phi-2",  # ~2.7B parameters
        role="primary_reasoner",
        reasoning_depth=3
    ),
    CoTAgent(
        agent_id="critic", 
        model_name="phi-1.5",  # ~1.3B parameters
        role="adversarial_critic",
        reasoning_depth=2
    ),
    CoTAgent(
        agent_id="synthesizer",
        model_name="phi-2",
        role="synthesizer",
        reasoning_depth=4
    )
]

# Create coordinator
coordinator = MultiAgentCoordinator(
    agents=agents,
    max_rounds=5,
    consensus_threshold=0.8,
    enable_adversarial=True
)

# Solve a problem
problem = "What is the optimal strategy for a startup with limited resources?"

result = coordinator.solve_problem(
    problem=problem,
    interaction_mode=InteractionMode.ADVERSARIAL
)

print(f"Final Answer: {result.final_answer}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning Steps: {len(result.reasoning_chain)}")
```

### Research Experiments

```python
from multi_agent_cot.examples.research_experiments import (
    research_experiment_1, 
    research_experiment_2,
    research_experiment_3
)

# Compare multi-agent vs single large model
results1 = research_experiment_1()

# Compare interaction modes
results2 = research_experiment_2()

# Analyze scalability
results3 = research_experiment_3()
```

## 🔬 Supported Models

The system is optimized for small language models (1-3B parameters):

- **Microsoft Phi-2** (~2.7B) - Primary reasoning model
- **Microsoft Phi-1.5** (~1.3B) - Efficient critic model  
- **StableLM-3B** (~3B) - Alternative reasoning model
- **Pythia-2.8B** (~2.8B) - Research baseline model
- **GPT2-Medium/Large** (355M/774M) - Lightweight options

### Adding Custom Models

```python
from multi_agent_cot.models.model_loader import ModelLoader

loader = ModelLoader()

# Add your custom small model
custom_agent = CoTAgent(
    agent_id="custom_agent",
    model_name="your-org/your-small-model",  # Any HF model
    role="specialist"
)
```

## 🧪 Interaction Modes

### 1. Adversarial Mode
- Agents challenge each other's responses
- High-confidence agents attack low-confidence responses
- Promotes robust reasoning through criticism

### 2. Collaborative Mode  
- Agents build upon each other's ideas
- Focuses on synthesis and improvement
- Maximizes collective intelligence

### 3. Debate Mode
- Structured argumentation between agent pairs
- Alternating critique and defense rounds
- Deepens analysis through opposition

### 4. Consensus Mode
- Agents work toward agreement
- Iterative refinement of responses
- Balances diverse perspectives

### 5. Critique-Improve Mode
- Explicit critique followed by improvement
- Self-reflection and peer review
- Quality enhancement focus

## 📊 Evaluation Framework

### Built-in Benchmarks

```python
from multi_agent_cot.evaluation.evaluator import MultiAgentEvaluator

evaluator = MultiAgentEvaluator()

# Create benchmark tasks
tasks = evaluator.create_benchmark_tasks()

# Evaluate your system
results = evaluator.evaluate_coordinator(
    coordinator=coordinator,
    benchmark_tasks=tasks
)

# Generate report
report = evaluator.generate_report(results)
print(report)
```

### Custom Evaluation

```python
from multi_agent_cot.evaluation.evaluator import BenchmarkTask

custom_task = BenchmarkTask(
    task_id="custom_001",
    question="Your research question here",
    expected_answer="Expected answer",
    category="your_category",
    difficulty="medium",
    reasoning_type="logical"
)

results = evaluator.evaluate_coordinator(
    coordinator=coordinator,
    benchmark_tasks=[custom_task]
)
```

## ⚙️ Configuration

### YAML Configuration

```yaml
# config.yaml
models:
  - name: "phi-2"
    model_path: "microsoft/phi-2"
    max_tokens: 512
    temperature: 0.7
    load_in_8bit: true

agents:
  - agent_id: "reasoner_1"
    model_name: "phi-2"
    role: "primary_reasoner"
    reasoning_depth: 3
    temperature: 0.7

coordinator:
  max_rounds: 5
  consensus_threshold: 0.8
  enable_adversarial: true
  default_interaction_mode: "adversarial"

evaluation:
  output_dir: "./results"
  save_detailed_results: true
```

### Loading Configuration

```python
from multi_agent_cot.utils.config import ConfigManager

config_manager = ConfigManager("config.yaml")
config = config_manager.get_config()

# Validate configuration
issues = config_manager.validate_config()
if not issues:
    print("Configuration is valid!")
```

## 🔍 Research Applications

### 1. Model Efficiency Studies
- Compare small multi-agent vs large single model
- Analyze compute/performance trade-offs
- Study scaling laws for collaborative systems

### 2. Reasoning Quality Analysis
- Evaluate different interaction strategies
- Measure reasoning depth and coherence
- Compare adversarial vs collaborative approaches

### 3. Consensus and Agreement
- Study how agents reach consensus
- Analyze disagreement patterns
- Optimize consensus mechanisms

### 4. Specialized Agent Roles
- Mathematical reasoning specialists
- Logical analysis experts
- Creative thinking agents
- Quality control critics

## 📈 Performance Metrics

- **Accuracy**: Correctness of final answers
- **Reasoning Quality**: Depth and coherence of thought processes
- **Efficiency**: Performance per computational cost
- **Consensus Score**: Agreement between agents
- **Token Usage**: Computational resource consumption
- **Response Time**: Speed of problem solving

## 🛠️ Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=multi_agent_cot tests/
```

### Code Quality

```bash
# Format code
black multi_agent_cot/

# Lint code
flake8 multi_agent_cot/

# Type checking
mypy multi_agent_cot/
```

## 📚 Research Paper Contributions

This framework enables research into:

1. **Collaborative Intelligence**: How small models can collectively outperform larger ones
2. **Adversarial Reasoning**: Benefits of competitive interactions in AI systems
3. **Efficient Scaling**: Scaling reasoning capabilities without proportional parameter increase
4. **Specialized Agents**: Role-based specialization in multi-agent systems
5. **Consensus Mechanisms**: How AI agents can reach agreement on complex problems

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{span2024multiagentcot,
  title={Multi-Agent Adversarial Chain-of-Thought for Small Language Models},
  author={Panos Span},
  year={2024},
  note={MSc Thesis Project in Machine Learning}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft for the Phi model series
- Hugging Face for the transformers library
- The research community working on multi-agent systems
- Academic supervisors and collaborators

## 📞 Contact

For questions about the research or implementation:
- GitHub Issues: [Create an issue](https://github.com/panos-span/multi_agent_CoT/issues)
- Research inquiries: Contact through your institution

---

**Thesis Goal**: Demonstrate that carefully orchestrated small language models can achieve reasoning performance comparable to much larger models while using significantly less computational resources.