"""
Multi-Agent Coordinator for orchestrating adversarial CoT interactions
"""
import time
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from multi_agent_cot.agents.base_agent import BaseAgent, Message, AgentResponse

logger = logging.getLogger(__name__)


class InteractionMode(Enum):
    """Different modes of agent interaction"""
    DEBATE = "debate"
    CONSENSUS = "consensus"
    CRITIQUE_IMPROVE = "critique_improve"
    ADVERSARIAL = "adversarial"
    COLLABORATIVE = "collaborative"


@dataclass
class ConversationRound:
    """Represents a round of conversation between agents"""
    round_number: int
    responses: List[AgentResponse]
    mode: InteractionMode
    duration: float
    consensus_score: float = 0.0


@dataclass
class ConversationResult:
    """Final result of a multi-agent conversation"""
    final_answer: str
    confidence: float
    reasoning_chain: List[str]
    rounds: List[ConversationRound]
    participating_agents: List[str]
    total_duration: float
    token_usage: int


class MultiAgentCoordinator:
    """Coordinates interactions between multiple CoT agents"""
    
    def __init__(
        self,
        agents: List[BaseAgent],
        max_rounds: int = 5,
        consensus_threshold: float = 0.8,
        enable_adversarial: bool = True,
        **kwargs
    ):
        """Initialize the multi-agent coordinator
        
        Args:
            agents: List of agents to coordinate
            max_rounds: Maximum number of conversation rounds
            consensus_threshold: Threshold for reaching consensus
            enable_adversarial: Whether to enable adversarial interactions
            **kwargs: Additional coordination parameters
        """
        self.agents = {agent.agent_id: agent for agent in agents}
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.enable_adversarial = enable_adversarial
        
        self.conversation_history: List[ConversationRound] = []
        self.current_round = 0
        
        # Coordination strategies
        self.interaction_strategies = {
            InteractionMode.DEBATE: self._run_debate_round,
            InteractionMode.CONSENSUS: self._run_consensus_round,
            InteractionMode.CRITIQUE_IMPROVE: self._run_critique_improve_round,
            InteractionMode.ADVERSARIAL: self._run_adversarial_round,
            InteractionMode.COLLABORATIVE: self._run_collaborative_round,
        }
        
        logger.info(f"Initialized MultiAgentCoordinator with {len(agents)} agents")
    
    def solve_problem(
        self,
        problem: str,
        interaction_mode: InteractionMode = InteractionMode.ADVERSARIAL,
        custom_roles: Dict[str, str] = None,
        **kwargs
    ) -> ConversationResult:
        """Solve a problem using multi-agent interaction
        
        Args:
            problem: The problem statement to solve
            interaction_mode: Mode of interaction between agents
            custom_roles: Custom roles for specific agents
            **kwargs: Additional solving parameters
            
        Returns:
            ConversationResult with the final solution
        """
        start_time = time.time()
        
        logger.info(f"Starting problem solving with {len(self.agents)} agents in {interaction_mode.value} mode")
        
        # Reset state
        self.conversation_history.clear()
        self.current_round = 0
        
        # Assign roles if provided
        if custom_roles:
            self._assign_roles(custom_roles)
        
        # Initial responses from all agents
        initial_responses = self._get_initial_responses(problem)
        
        # Run interaction rounds
        rounds = [ConversationRound(
            round_number=0,
            responses=initial_responses,
            mode=interaction_mode,
            duration=0.0,
            consensus_score=self._calculate_consensus(initial_responses)
        )]
        
        # Continue with additional rounds based on interaction mode
        for round_num in range(1, self.max_rounds + 1):
            self.current_round = round_num
            
            round_start = time.time()
            
            # Run interaction round
            round_responses = self.interaction_strategies[interaction_mode](
                problem, rounds[-1].responses
            )
            
            round_duration = time.time() - round_start
            consensus_score = self._calculate_consensus(round_responses)
            
            current_round = ConversationRound(
                round_number=round_num,
                responses=round_responses,
                mode=interaction_mode,
                duration=round_duration,
                consensus_score=consensus_score
            )
            
            rounds.append(current_round)
            
            # Check for convergence
            if consensus_score >= self.consensus_threshold:
                logger.info(f"Consensus reached at round {round_num} (score: {consensus_score:.3f})")
                break
        
        # Generate final result
        final_result = self._synthesize_final_answer(rounds, problem)
        final_result.rounds = rounds
        final_result.total_duration = time.time() - start_time
        
        logger.info(f"Problem solving completed in {len(rounds)} rounds ({final_result.total_duration:.2f}s)")
        
        return final_result
    
    def _get_initial_responses(self, problem: str) -> List[AgentResponse]:
        """Get initial responses from all agents"""
        responses = []
        
        for agent in self.agents.values():
            if agent.is_active:
                try:
                    response = agent.generate_response(problem)
                    responses.append(response)
                except Exception as e:
                    logger.error(f"Agent {agent.agent_id} failed to generate initial response: {e}")
        
        return responses
    
    def _run_debate_round(
        self, 
        problem: str, 
        previous_responses: List[AgentResponse]
    ) -> List[AgentResponse]:
        """Run a debate round where agents argue different positions"""
        responses = []
        
        # Select pairs of agents to debate
        active_agents = [agent for agent in self.agents.values() if agent.is_active]
        
        for i in range(0, len(active_agents), 2):
            if i + 1 < len(active_agents):
                agent1, agent2 = active_agents[i], active_agents[i + 1]
                
                # Agent1 responds to agent2's previous response
                prev_response = self._find_agent_response(previous_responses, agent2.agent_id)
                if prev_response:
                    response1 = agent1.critique_response(prev_response, problem)
                    responses.append(response1)
                
                # Agent2 responds to agent1's critique
                response2 = agent2.critique_response(response1, problem)
                responses.append(response2)
        
        return responses
    
    def _run_consensus_round(
        self, 
        problem: str, 
        previous_responses: List[AgentResponse]
    ) -> List[AgentResponse]:
        """Run a consensus-building round"""
        responses = []
        
        # Create synthesis prompt including all previous responses
        context_messages = [resp.message for resp in previous_responses]
        
        for agent in self.agents.values():
            if agent.is_active:
                synthesis_prompt = f"""
                Based on the following responses from other agents, provide a synthesized answer:
                
                Original Problem: {problem}
                
                Previous Responses:
                {self._format_responses_for_context(previous_responses)}
                
                Please provide a consensus view that incorporates the best insights:
                """
                
                response = agent.generate_response(synthesis_prompt, context_messages)
                responses.append(response)
        
        return responses
    
    def _run_critique_improve_round(
        self, 
        problem: str, 
        previous_responses: List[AgentResponse]
    ) -> List[AgentResponse]:
        """Run a critique-and-improve round"""
        responses = []
        
        # Each agent critiques others and improves their own response
        for agent in self.agents.values():
            if not agent.is_active:
                continue
            
            # Find agent's previous response
            agent_prev_response = self._find_agent_response(previous_responses, agent.agent_id)
            
            # Collect critiques from other agents
            critiques = []
            for other_response in previous_responses:
                if other_response.message.sender != agent.agent_id:
                    critique = agent.critique_response(other_response, problem)
                    critiques.append(critique)
            
            # Generate improved response based on critiques
            if critiques:
                improvement_prompt = f"""
                Original Problem: {problem}
                
                Your previous response: {agent_prev_response.message.content if agent_prev_response else "None"}
                
                Critiques received:
                {self._format_responses_for_context(critiques)}
                
                Please provide an improved response that addresses these critiques:
                """
                
                improved_response = agent.generate_response(improvement_prompt)
                responses.append(improved_response)
        
        return responses
    
    def _run_adversarial_round(
        self, 
        problem: str, 
        previous_responses: List[AgentResponse]
    ) -> List[AgentResponse]:
        """Run an adversarial round with competitive dynamics"""
        responses = []
        
        # Sort agents by confidence to create adversarial pairs
        agent_confidences = []
        for response in previous_responses:
            agent = self.agents[response.message.sender]
            agent_confidences.append((agent, response.confidence))
        
        agent_confidences.sort(key=lambda x: x[1], reverse=True)
        
        # High confidence agents attack low confidence responses
        for i, (high_conf_agent, _) in enumerate(agent_confidences[:len(agent_confidences)//2]):
            if i < len(agent_confidences) - i - 1:
                low_conf_response = previous_responses[len(agent_confidences) - i - 1]
                
                adversarial_prompt = f"""
                You are challenging this response. Find flaws and provide a better alternative:
                
                Original Problem: {problem}
                Response to Challenge: {low_conf_response.message.content}
                
                Your adversarial response:
                """
                
                attack_response = high_conf_agent.generate_response(adversarial_prompt)
                responses.append(attack_response)
        
        # Low confidence agents defend and improve
        for i, response in enumerate(previous_responses[len(agent_confidences)//2:]):
            agent = self.agents[response.message.sender]
            
            defense_prompt = f"""
            Your response is being challenged. Defend and improve your position:
            
            Original Problem: {problem}
            Your previous response: {response.message.content}
            
            Provide a stronger, defended response:
            """
            
            defense_response = agent.generate_response(defense_prompt)
            responses.append(defense_response)
        
        return responses
    
    def _run_collaborative_round(
        self, 
        problem: str, 
        previous_responses: List[AgentResponse]
    ) -> List[AgentResponse]:
        """Run a collaborative round focusing on building upon each other's ideas"""
        responses = []
        
        # Agents build upon the best ideas from previous round
        best_response = max(previous_responses, key=lambda r: r.confidence)
        
        for agent in self.agents.values():
            if not agent.is_active or agent.agent_id == best_response.message.sender:
                continue
            
            collaboration_prompt = f"""
            Build upon this strong response from a colleague:
            
            Original Problem: {problem}
            Strong Response: {best_response.message.content}
            
            How can you extend and improve this response:
            """
            
            collaborative_response = agent.generate_response(collaboration_prompt)
            responses.append(collaborative_response)
        
        return responses
    
    def _calculate_consensus(self, responses: List[AgentResponse]) -> float:
        """Calculate consensus score among responses"""
        if len(responses) < 2:
            return 1.0
        
        # Simple heuristic: average confidence and similarity
        avg_confidence = sum(r.confidence for r in responses) / len(responses)
        
        # Similarity based on reasoning steps overlap (simplified)
        similarity_scores = []
        for i, resp1 in enumerate(responses):
            for resp2 in responses[i+1:]:
                # Count common reasoning keywords
                keywords1 = set(resp1.message.content.lower().split())
                keywords2 = set(resp2.message.content.lower().split())
                similarity = len(keywords1 & keywords2) / len(keywords1 | keywords2) if keywords1 | keywords2 else 0
                similarity_scores.append(similarity)
        
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        
        return (avg_confidence + avg_similarity) / 2
    
    def _synthesize_final_answer(
        self, 
        rounds: List[ConversationRound], 
        problem: str
    ) -> ConversationResult:
        """Synthesize the final answer from all rounds"""
        
        # Get the best response from the last round
        last_round = rounds[-1]
        best_response = max(last_round.responses, key=lambda r: r.confidence)
        
        # Collect all reasoning steps
        all_reasoning = []
        for round_data in rounds:
            for response in round_data.responses:
                all_reasoning.extend(response.reasoning_steps)
        
        # Calculate total token usage
        total_tokens = sum(
            response.token_count 
            for round_data in rounds 
            for response in round_data.responses
        )
        
        # Get participating agents
        participating_agents = list(set(
            response.message.sender 
            for round_data in rounds 
            for response in round_data.responses
        ))
        
        return ConversationResult(
            final_answer=best_response.message.content,
            confidence=best_response.confidence,
            reasoning_chain=all_reasoning,
            rounds=[],  # Will be set by caller
            participating_agents=participating_agents,
            total_duration=0.0,  # Will be set by caller
            token_usage=total_tokens
        )
    
    def _find_agent_response(
        self, 
        responses: List[AgentResponse], 
        agent_id: str
    ) -> Optional[AgentResponse]:
        """Find response from specific agent"""
        for response in responses:
            if response.message.sender == agent_id:
                return response
        return None
    
    def _format_responses_for_context(self, responses: List[AgentResponse]) -> str:
        """Format responses for use as context in prompts"""
        formatted = ""
        for i, response in enumerate(responses):
            formatted += f"Agent {response.message.sender}: {response.message.content}\n\n"
        return formatted
    
    def _assign_roles(self, custom_roles: Dict[str, str]):
        """Assign custom roles to agents"""
        for agent_id, role in custom_roles.items():
            if agent_id in self.agents:
                self.agents[agent_id].role = role
                logger.info(f"Assigned role '{role}' to agent {agent_id}")
    
    def add_agent(self, agent: BaseAgent):
        """Add a new agent to the coordinator"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Added agent {agent.agent_id} to coordinator")
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from the coordinator"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Removed agent {agent_id} from coordinator")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        return {
            "num_agents": len(self.agents),
            "active_agents": sum(1 for agent in self.agents.values() if agent.is_active),
            "rounds_completed": len(self.conversation_history),
            "max_rounds": self.max_rounds,
            "consensus_threshold": self.consensus_threshold,
            "adversarial_enabled": self.enable_adversarial
        }