"""State definition for LangGraph workflow."""

from typing import TypedDict, Literal, Optional
from typing_extensions import Annotated
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    """State shared across all nodes in the research workflow.
    
    This state flows through the graph and gets updated by each node.
    """
    # Input
    question: str
    """The research question to answer"""
    
    # Question analysis
    question_type: Optional[Literal["wet_lab", "computational", "literature", "general"]]
    """Classification of question type for routing"""
    
    question_complexity: Optional[Literal["simple", "moderate", "complex"]]
    """Complexity assessment (affects team size, rounds)"""
    
    # Team composition
    team_composition: list[dict]
    """List of specialist agent specifications"""
    
    team_size: int
    """Number of specialists to use (default: 3)"""
    
    # Meeting execution
    meeting_transcript: list[dict]
    """Full transcript of Virtual Lab meeting"""
    
    num_rounds: int
    """Number of discussion rounds (default: 2)"""
    
    # Human-in-the-loop
    requires_human_approval: bool
    """Whether to pause for human review (default: False)"""
    
    human_feedback: Optional[str]
    """Feedback provided by human reviewer"""
    
    approval_status: Optional[Literal["approved", "rejected", "revision_requested"]]
    """Human approval decision"""
    
    # Output
    final_answer: str
    """Synthesized final answer from PI"""
    
    confidence_score: Optional[float]
    """Confidence in the answer (0-1)"""
    
    references: list[str]
    """List of references (PMIDs, data sources)"""
    
    # Consensus mechanism (optional)
    requires_consensus: Optional[bool]
    """Whether to use multi-model consensus (default: False)"""
    
    consensus_models: Optional[list[str]]
    """List of models to use for consensus"""
    
    consensus_metadata: Optional[dict]
    """Metadata from consensus: agreement_score, key_agreements, key_disagreements"""
    
    # Metadata
    execution_path: list[str]
    """Track which nodes were executed"""
    
    errors: list[str]
    """Any errors encountered during execution"""


def update_execution_path(current: list[str], update: str) -> list[str]:
    """Reducer function to append to execution path."""
    return current + [update]


# Alternative using Annotated for automatic state updates
class ResearchStateAnnotated(TypedDict):
    """Research state with annotated reducers for automatic merging."""
    
    question: str
    question_type: Optional[str]
    question_complexity: Optional[str]
    team_composition: list[dict]
    team_size: int
    meeting_transcript: list[dict]
    num_rounds: int
    requires_human_approval: bool
    human_feedback: Optional[str]
    approval_status: Optional[str]
    final_answer: str
    confidence_score: Optional[float]
    references: list[str]
    
    # Annotated fields with custom reducers
    execution_path: Annotated[list[str], update_execution_path]
    """Automatically appends to path instead of replacing"""
    
    errors: Annotated[list[str], lambda x, y: x + y]
    """Automatically concatenates errors"""
