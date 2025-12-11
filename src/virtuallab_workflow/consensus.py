"""Consensus mechanism for running multiple Virtual Lab meetings with different models."""

import os
from typing import List, Dict, Any, Optional
from src.agent.meeting import VirtualLabMeeting
from src.agent.openrouter_client import OpenRouterClient
from src.agent.anthropic_client import AnthropicClient


# Model configurations for consensus
DEFAULT_CONSENSUS_MODELS = [
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4",
]


def run_consensus_meeting(
    question: str,
    models: Optional[List[str]] = None,
    provider: str = "openrouter",
    team_size: int = 3,
    num_rounds: int = 2,
    max_iterations: int = 10,
    data_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run multiple Virtual Lab meetings with different models and synthesize consensus.
    
    Configuration Priority:
        1. Explicit parameters (highest)
        2. Environment variables (.env)
        3. Function defaults (lowest)
    
    Environment Variables Used:
        - OPENROUTER_API_KEY: Required for authentication
        - DATABASE_DIR: Data directory (if data_dir not provided)
        - OPENROUTER_MODEL: NOT used (consensus needs multiple models)
    
    Args:
        question: Research question
        models: List of model names to use. If None, uses DEFAULT_CONSENSUS_MODELS.
                Note: Ignores OPENROUTER_MODEL from .env (needs multiple models)
        provider: LLM provider (openrouter or anthropic). Default: "openrouter"
        team_size: Number of specialists per meeting. Default: 3
        num_rounds: Discussion rounds per meeting. Default: 2
        max_iterations: Maximum iterations per specialist. Default: 10 (reduce to save credits)
        data_dir: Database directory. If None, uses DATABASE_DIR from .env
        verbose: Print progress. Default: True
        
    Returns:
        Dictionary with:
        - individual_answers: List of answers from each model
        - consensus_answer: Synthesized consensus
        - agreement_score: How much models agree (0.0-1.0)
        - key_agreements: List of agreed-upon points
        - key_disagreements: List of disagreement areas
        - successful_models: Number of successful executions
        - total_models: Total number of models attempted
        
    Example:
        >>> result = run_consensus_meeting(
        ...     question="What is T cell exhaustion?",
        ...     models=["google/gemini-2.0-flash-exp:free", 
        ...             "meta-llama/llama-3.3-70b-instruct:free"],
        ...     team_size=2,
        ...     num_rounds=1
        ... )
        >>> print(f"Agreement: {result['agreement_score']:.0%}")
        >>> print(f"Consensus: {result['consensus_answer']}")
    """
    if models is None:
        models = DEFAULT_CONSENSUS_MODELS
    
    if data_dir is None:
        data_dir = os.getenv("DATABASE_DIR", "/home.galaxy4/sumin/project/aisci/Competition_Data")
    
    api_key = os.getenv("OPENROUTER_API_KEY") if provider == "openrouter" else os.getenv("ANTHROPIC_API_KEY")
    
    if verbose:
        print("\n" + "="*80)
        print(f"CONSENSUS MECHANISM: Running {len(models)} meetings")
        print("="*80)
        print(f"Question: {question}")
        print(f"Models: {', '.join(models)}")
        print("="*80 + "\n")
    
    # Run meetings with each model
    individual_results = []
    
    for i, model in enumerate(models, 1):
        if verbose:
            print(f"\n{'='*80}")
            print(f"MEETING {i}/{len(models)}: {model}")
            print(f"{'='*80}\n")
        
        try:
            meeting = VirtualLabMeeting(
                user_question=question,
                model=model,
                provider=provider,
                api_key=api_key,
                max_team_size=team_size,
                verbose=verbose,
                data_dir=data_dir
            )
            
            answer = meeting.run_meeting(num_rounds=num_rounds)
            
            individual_results.append({
                "model": model,
                "answer": answer,
                "success": True,
                "error": None
            })
            
            if verbose:
                print(f"\n✓ Meeting {i} complete ({model})")
                print(f"Answer length: {len(answer)} characters\n")
                
        except Exception as e:
            error_msg = str(e)
            individual_results.append({
                "model": model,
                "answer": None,
                "success": False,
                "error": error_msg
            })
            
            if verbose:
                print(f"\n❌ Meeting {i} failed ({model})")
                print(f"Error: {error_msg}\n")
    
    # Filter successful results
    successful_results = [r for r in individual_results if r["success"]]
    
    if not successful_results:
        return {
            "individual_answers": individual_results,
            "consensus_answer": "All meetings failed - no consensus available",
            "agreement_score": 0.0,
            "model_votes": {},
            "successful_models": 0,
            "total_models": len(models)
        }
    
    # Synthesize consensus
    if verbose:
        print(f"\n{'='*80}")
        print(f"SYNTHESIZING CONSENSUS from {len(successful_results)} successful meetings")
        print(f"{'='*80}\n")
    
    consensus = _synthesize_consensus(
        question=question,
        individual_results=successful_results,
        provider=provider,
        api_key=api_key,
        verbose=verbose
    )
    
    return {
        "individual_answers": individual_results,
        "consensus_answer": consensus["answer"],
        "consensus_reasoning": consensus["reasoning"],
        "agreement_score": consensus["agreement_score"],
        "key_agreements": consensus["key_agreements"],
        "key_disagreements": consensus["key_disagreements"],
        "successful_models": len(successful_results),
        "total_models": len(models)
    }


def _synthesize_consensus(
    question: str,
    individual_results: List[Dict[str, Any]],
    provider: str,
    api_key: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """Synthesize consensus from multiple model outputs.
    
    Uses a meta-model to analyze agreements and disagreements.
    """
    # Use a strong model for synthesis (Gemini 2.0 or fallback)
    synthesis_model = "google/gemini-2.0-flash-exp:free"
    
    # Create synthesis prompt
    answers_text = "\n\n".join([
        f"MODEL {i+1} ({r['model']}):\n{r['answer']}"
        for i, r in enumerate(individual_results)
    ])
    
    synthesis_prompt = f"""You are synthesizing consensus from multiple AI research teams that independently analyzed the same question.

ORIGINAL QUESTION:
{question}

INDIVIDUAL TEAM ANSWERS:
{answers_text}

Your task is to:
1. Identify key points where ALL or MOST teams AGREE
2. Identify points where teams DISAGREE
3. Synthesize a CONSENSUS answer that:
   - Incorporates agreed-upon facts and conclusions
   - Notes areas of uncertainty or disagreement
   - Provides the most accurate and comprehensive answer
   
Format your response as:

CONSENSUS ANSWER:
[Your synthesized answer incorporating agreements]

KEY AGREEMENTS:
- [Point 1 where most/all teams agree]
- [Point 2 where most/all teams agree]
...

KEY DISAGREEMENTS:
- [Point 1 where teams differ]
- [Point 2 where teams differ]
...

AGREEMENT SCORE: [0-100, where 100 = complete agreement, 0 = no agreement]

REASONING:
[Brief explanation of how you synthesized the consensus]
"""

    if verbose:
        print("Running meta-synthesis with consensus model...")
    
    # Call synthesis model
    if provider == "openrouter":
        client = OpenRouterClient(api_key=api_key, model=synthesis_model)
    else:
        client = AnthropicClient(api_key=api_key, model=synthesis_model)
    
    try:
        response = client.create_message(
            messages=[{"role": "user", "content": synthesis_prompt}],
            max_tokens=4000,
            temperature=0.3  # Lower temperature for consistent synthesis
        )
        
        synthesis_text = client.get_response_text(response)
        
        # Parse response
        consensus_answer = ""
        key_agreements = []
        key_disagreements = []
        agreement_score = 0.5
        reasoning = ""
        
        current_section = None
        for line in synthesis_text.split("\n"):
            line = line.strip()
            
            if line.startswith("CONSENSUS ANSWER:"):
                current_section = "consensus"
                continue
            elif line.startswith("KEY AGREEMENTS:"):
                current_section = "agreements"
                continue
            elif line.startswith("KEY DISAGREEMENTS:"):
                current_section = "disagreements"
                continue
            elif line.startswith("AGREEMENT SCORE:"):
                current_section = "score"
                try:
                    score_text = line.split("AGREEMENT SCORE:")[1].strip()
                    # Extract number (handles formats like "85" or "85/100" or "85%")
                    score_num = ''.join(c for c in score_text if c.isdigit())
                    if score_num:
                        agreement_score = int(score_num) / 100.0
                except:
                    agreement_score = 0.5
                continue
            elif line.startswith("REASONING:"):
                current_section = "reasoning"
                continue
            
            # Append to current section
            if current_section == "consensus" and line:
                consensus_answer += line + "\n"
            elif current_section == "agreements" and line.startswith("-"):
                key_agreements.append(line[1:].strip())
            elif current_section == "disagreements" and line.startswith("-"):
                key_disagreements.append(line[1:].strip())
            elif current_section == "reasoning" and line:
                reasoning += line + "\n"
        
        return {
            "answer": consensus_answer.strip() or synthesis_text,  # Fallback to full text
            "key_agreements": key_agreements,
            "key_disagreements": key_disagreements,
            "agreement_score": min(1.0, max(0.0, agreement_score)),  # Clamp to [0, 1]
            "reasoning": reasoning.strip()
        }
        
    except Exception as e:
        if verbose:
            print(f"⚠ Consensus synthesis failed: {e}")
            print("Falling back to first successful answer")
        
        # Fallback: return first successful answer
        return {
            "answer": individual_results[0]["answer"],
            "key_agreements": ["Could not synthesize - using first model's answer"],
            "key_disagreements": [],
            "agreement_score": 0.0,
            "reasoning": f"Synthesis failed: {str(e)}"
        }


def compare_model_answers(
    question: str,
    models: Optional[List[str]] = None,
    team_size: int = 3,
    num_rounds: int = 2,
    output_file: Optional[str] = None
) -> str:
    """Compare answers from different models side-by-side.
    
    Args:
        question: Research question
        models: List of models to compare
        team_size: Team size per meeting
        num_rounds: Rounds per meeting
        output_file: Optional file to save comparison
        
    Returns:
        Formatted comparison text
    """
    result = run_consensus_meeting(
        question=question,
        models=models,
        team_size=team_size,
        num_rounds=num_rounds,
        verbose=True
    )
    
    # Format comparison
    comparison = []
    comparison.append("="*80)
    comparison.append("MODEL COMPARISON REPORT")
    comparison.append("="*80)
    comparison.append(f"\nQuestion: {question}\n")
    comparison.append(f"Models tested: {result['total_models']}")
    comparison.append(f"Successful: {result['successful_models']}")
    comparison.append(f"Agreement score: {result['agreement_score']:.2%}\n")
    
    comparison.append("-"*80)
    comparison.append("INDIVIDUAL ANSWERS")
    comparison.append("-"*80)
    
    for i, answer in enumerate(result['individual_answers'], 1):
        comparison.append(f"\nMODEL {i}: {answer['model']}")
        if answer['success']:
            comparison.append(f"Status: ✓ Success")
            comparison.append(f"Answer ({len(answer['answer'])} chars):")
            comparison.append(answer['answer'][:500] + "..." if len(answer['answer']) > 500 else answer['answer'])
        else:
            comparison.append(f"Status: ✗ Failed")
            comparison.append(f"Error: {answer['error']}")
    
    comparison.append("\n" + "="*80)
    comparison.append("CONSENSUS SYNTHESIS")
    comparison.append("="*80)
    comparison.append(f"\n{result['consensus_answer']}\n")
    
    if result['key_agreements']:
        comparison.append("-"*80)
        comparison.append("KEY AGREEMENTS:")
        for agreement in result['key_agreements']:
            comparison.append(f"  • {agreement}")
    
    if result['key_disagreements']:
        comparison.append("\n" + "-"*80)
        comparison.append("KEY DISAGREEMENTS:")
        for disagreement in result['key_disagreements']:
            comparison.append(f"  • {disagreement}")
    
    comparison.append("\n" + "="*80)
    
    comparison_text = "\n".join(comparison)
    
    # Save to file if requested
    if output_file:
        from pathlib import Path
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(comparison_text)
        print(f"\n✓ Comparison saved to: {output_file}")
    
    return comparison_text
