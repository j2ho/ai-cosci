"""Enhanced LangGraph nodes with consensus mechanism."""

import os
from src.virtuallab_workflow.state import ResearchState
from src.virtuallab_workflow.consensus import run_consensus_meeting, DEFAULT_CONSENSUS_MODELS


def virtual_lab_consensus_node(state: ResearchState) -> dict:
    """Execute Virtual Lab with consensus across multiple models.
    
    This node runs the question through multiple LLM models and
    synthesizes a consensus answer.
    """
    question = state["question"]
    team_size = state.get("team_size", 3)
    num_rounds = state.get("num_rounds", 2)
    
    # Models for consensus (can be configured via state)
    consensus_models = state.get("consensus_models", DEFAULT_CONSENSUS_MODELS)
    
    try:
        result = run_consensus_meeting(
            question=question,
            models=consensus_models,
            team_size=team_size,
            num_rounds=num_rounds,
            verbose=True
        )
        
        # Format meeting transcript from individual answers
        transcript_parts = []
        for i, answer in enumerate(result['individual_answers'], 1):
            if answer['success']:
                transcript_parts.append(f"=== MODEL {i}: {answer['model']} ===\n{answer['answer']}\n")
        
        meeting_transcript = "\n".join(transcript_parts)
        
        return {
            "meeting_transcript": meeting_transcript,
            "final_answer": result['consensus_answer'],
            "confidence_score": result['agreement_score'],
            "team_composition": [f"Consensus ({result['successful_models']} models)"],
            "consensus_metadata": {
                "agreement_score": result['agreement_score'],
                "key_agreements": result.get('key_agreements', []),
                "key_disagreements": result.get('key_disagreements', []),
                "successful_models": result['successful_models'],
                "total_models": result['total_models']
            },
            "execution_path": ["virtual_lab_consensus"]
        }
    except Exception as e:
        return {
            "meeting_transcript": "",
            "final_answer": f"Error executing consensus meeting: {str(e)}",
            "confidence_score": 0.0,
            "team_composition": [],
            "execution_path": ["virtual_lab_consensus"],
            "errors": [str(e)]
        }


def should_use_consensus(state: ResearchState) -> str:
    """Conditional edge: decide whether to use consensus mechanism.
    
    Routes to consensus if:
    - User explicitly requested it (requires_consensus=True)
    - Question complexity is "complex"
    - Confidence threshold requires extra validation
    
    Args:
        state: Current research state
        
    Returns:
        "consensus" or "single_model"
    """
    # Check if consensus explicitly requested
    if state.get("requires_consensus", False):
        return "consensus"
    
    # Use consensus for complex questions
    if state.get("question_complexity") == "complex":
        return "consensus"
    
    # Default to single model
    return "single_model"
