"""
Evaluation framework for comparing multi-agent CoT performance
"""
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

from multi_agent_cot.orchestrator.coordinator import ConversationResult, MultiAgentCoordinator

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating agent performance"""
    accuracy: float
    reasoning_quality: float
    efficiency: float
    token_usage: int
    response_time: float
    consensus_rounds: int
    confidence_score: float


@dataclass
class BenchmarkTask:
    """A benchmark task for evaluation"""
    task_id: str
    question: str
    expected_answer: str
    category: str
    difficulty: str  # easy, medium, hard
    reasoning_type: str  # mathematical, logical, commonsense, etc.


class MultiAgentEvaluator:
    """Evaluator for multi-agent CoT systems"""
    
    def __init__(self):
        """Initialize the evaluator"""
        self.evaluation_history: List[Dict[str, Any]] = []
        
    def evaluate_coordinator(
        self,
        coordinator: MultiAgentCoordinator,
        benchmark_tasks: List[BenchmarkTask],
        baseline_results: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate a multi-agent coordinator on benchmark tasks
        
        Args:
            coordinator: The coordinator to evaluate
            benchmark_tasks: List of benchmark tasks
            baseline_results: Optional baseline results for comparison
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Starting evaluation on {len(benchmark_tasks)} tasks")
        
        results = []
        total_start_time = time.time()
        
        for task in benchmark_tasks:
            logger.info(f"Evaluating task {task.task_id}: {task.category}")
            
            try:
                # Run the coordinator on this task
                conversation_result = coordinator.solve_problem(
                    problem=task.question,
                    **kwargs
                )
                
                # Calculate metrics
                metrics = self._calculate_metrics(
                    conversation_result, 
                    task,
                    **kwargs
                )
                
                task_result = {
                    "task_id": task.task_id,
                    "task": asdict(task),
                    "result": asdict(conversation_result),
                    "metrics": asdict(metrics),
                    "timestamp": time.time()
                }
                
                results.append(task_result)
                
            except Exception as e:
                logger.error(f"Failed to evaluate task {task.task_id}: {e}")
                continue
        
        total_time = time.time() - total_start_time
        
        # Aggregate results
        evaluation_summary = self._aggregate_results(results, total_time)
        
        # Compare with baseline if provided
        if baseline_results:
            evaluation_summary["baseline_comparison"] = self._compare_with_baseline(
                evaluation_summary, baseline_results
            )
        
        # Store in history
        self.evaluation_history.append(evaluation_summary)
        
        logger.info(f"Evaluation completed in {total_time:.2f}s")
        
        return evaluation_summary
    
    def _calculate_metrics(
        self,
        result: ConversationResult,
        task: BenchmarkTask,
        **kwargs
    ) -> EvaluationMetrics:
        """Calculate metrics for a single task result"""
        
        # Accuracy (simplified - could use more sophisticated matching)
        accuracy = self._calculate_accuracy(result.final_answer, task.expected_answer)
        
        # Reasoning quality (based on reasoning steps and confidence)
        reasoning_quality = self._calculate_reasoning_quality(result)
        
        # Efficiency (tokens per second)
        efficiency = result.token_usage / result.total_duration if result.total_duration > 0 else 0
        
        return EvaluationMetrics(
            accuracy=accuracy,
            reasoning_quality=reasoning_quality,
            efficiency=efficiency,
            token_usage=result.token_usage,
            response_time=result.total_duration,
            consensus_rounds=len(result.rounds),
            confidence_score=result.confidence
        )
    
    def _calculate_accuracy(self, predicted: str, expected: str) -> float:
        """Calculate accuracy score between predicted and expected answers"""
        # Simple exact match for now - could be enhanced with semantic similarity
        predicted_clean = predicted.strip().lower()
        expected_clean = expected.strip().lower()
        
        if predicted_clean == expected_clean:
            return 1.0
        
        # Check if expected answer is contained in predicted
        if expected_clean in predicted_clean:
            return 0.8
        
        # Check for partial matches (simplified)
        predicted_words = set(predicted_clean.split())
        expected_words = set(expected_clean.split())
        
        if expected_words and predicted_words:
            overlap = len(predicted_words & expected_words)
            return overlap / len(expected_words | predicted_words)
        
        return 0.0
    
    def _calculate_reasoning_quality(self, result: ConversationResult) -> float:
        """Calculate reasoning quality score"""
        factors = []
        
        # Number of reasoning steps
        if result.reasoning_chain:
            step_quality = min(len(result.reasoning_chain) / 5, 1.0)  # Normalize to max 5 steps
            factors.append(step_quality)
        
        # Confidence score
        factors.append(result.confidence)
        
        # Consensus through rounds (more rounds might indicate thorough reasoning)
        if result.rounds:
            round_quality = min(len(result.rounds) / 3, 1.0)  # Normalize to max 3 rounds
            factors.append(round_quality)
        
        return sum(factors) / len(factors) if factors else 0.0
    
    def _aggregate_results(self, results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """Aggregate results across all tasks"""
        if not results:
            return {"error": "No successful evaluations"}
        
        # Extract metrics
        all_metrics = [result["metrics"] for result in results]
        
        # Calculate averages
        avg_metrics = {}
        for metric_name in all_metrics[0].keys():
            values = [metrics[metric_name] for metrics in all_metrics]
            avg_metrics[f"avg_{metric_name}"] = sum(values) / len(values)
        
        # Category-wise performance
        category_performance = {}
        for result in results:
            category = result["task"]["category"]
            if category not in category_performance:
                category_performance[category] = []
            category_performance[category].append(result["metrics"]["accuracy"])
        
        for category, accuracies in category_performance.items():
            category_performance[category] = {
                "avg_accuracy": sum(accuracies) / len(accuracies),
                "num_tasks": len(accuracies)
            }
        
        return {
            "summary": avg_metrics,
            "category_performance": category_performance,
            "total_tasks": len(results),
            "total_evaluation_time": total_time,
            "detailed_results": results,
            "timestamp": time.time()
        }
    
    def _compare_with_baseline(
        self, 
        current_results: Dict[str, Any], 
        baseline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare current results with baseline"""
        comparison = {}
        
        current_summary = current_results.get("summary", {})
        baseline_summary = baseline_results.get("summary", {})
        
        for metric, current_value in current_summary.items():
            if metric in baseline_summary:
                baseline_value = baseline_summary[metric]
                if baseline_value != 0:
                    improvement = (current_value - baseline_value) / baseline_value
                    comparison[metric] = {
                        "current": current_value,
                        "baseline": baseline_value,
                        "improvement": improvement,
                        "better": improvement > 0
                    }
        
        return comparison
    
    def create_benchmark_tasks(self) -> List[BenchmarkTask]:
        """Create a set of benchmark tasks for evaluation"""
        tasks = [
            BenchmarkTask(
                task_id="math_001",
                question="What is 15% of 240?",
                expected_answer="36",
                category="mathematics",
                difficulty="easy",
                reasoning_type="mathematical"
            ),
            BenchmarkTask(
                task_id="logic_001", 
                question="If all roses are flowers, and some flowers are red, can we conclude that some roses are red?",
                expected_answer="No, we cannot conclude that some roses are red from the given statements.",
                category="logic",
                difficulty="medium",
                reasoning_type="logical"
            ),
            BenchmarkTask(
                task_id="reasoning_001",
                question="A farmer has 17 sheep, and all but 9 die. How many sheep does the farmer have left?",
                expected_answer="9",
                category="reasoning",
                difficulty="medium", 
                reasoning_type="logical"
            ),
            BenchmarkTask(
                task_id="commonsense_001",
                question="What happens to ice when it's heated above 0°C (32°F)?",
                expected_answer="It melts and becomes water.",
                category="commonsense",
                difficulty="easy",
                reasoning_type="commonsense"
            ),
            BenchmarkTask(
                task_id="complex_001",
                question="A company's revenue increased by 20% in year 1, decreased by 10% in year 2, and increased by 15% in year 3. If the initial revenue was $100,000, what is the final revenue?",
                expected_answer="$124,200",
                category="mathematics",
                difficulty="hard",
                reasoning_type="mathematical"
            )
        ]
        
        return tasks
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save evaluation results to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load evaluation results from file"""
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            logger.info(f"Results loaded from {filepath}")
            return results
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return {}
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report"""
        report = "Multi-Agent CoT Evaluation Report\n"
        report += "=" * 40 + "\n\n"
        
        summary = results.get("summary", {})
        
        report += "Overall Performance:\n"
        report += f"  Average Accuracy: {summary.get('avg_accuracy', 0):.3f}\n"
        report += f"  Average Reasoning Quality: {summary.get('avg_reasoning_quality', 0):.3f}\n"
        report += f"  Average Efficiency: {summary.get('avg_efficiency', 0):.1f} tokens/sec\n"
        report += f"  Average Response Time: {summary.get('avg_response_time', 0):.2f}s\n"
        report += f"  Average Token Usage: {summary.get('avg_token_usage', 0):.0f}\n\n"
        
        # Category performance
        category_perf = results.get("category_performance", {})
        if category_perf:
            report += "Performance by Category:\n"
            for category, perf in category_perf.items():
                report += f"  {category.capitalize()}: {perf['avg_accuracy']:.3f} ({perf['num_tasks']} tasks)\n"
            report += "\n"
        
        # Baseline comparison
        baseline_comp = results.get("baseline_comparison", {})
        if baseline_comp:
            report += "Baseline Comparison:\n"
            for metric, comp in baseline_comp.items():
                improvement = comp.get('improvement', 0)
                sign = "+" if improvement >= 0 else ""
                report += f"  {metric}: {sign}{improvement:.1%} {'✓' if comp.get('better', False) else '✗'}\n"
            report += "\n"
        
        report += f"Total Tasks: {results.get('total_tasks', 0)}\n"
        report += f"Evaluation Time: {results.get('total_evaluation_time', 0):.2f}s\n"
        
        return report