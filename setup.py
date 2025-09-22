from setuptools import setup, find_packages

setup(
    name="multi_agent_cot",
    version="0.1.0",
    description="Multi-Agent Adversarial Chain-of-Thought system for small language models",
    author="Panos Span",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.20.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.5.0",
        "datasets>=2.14.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ]
    },
)