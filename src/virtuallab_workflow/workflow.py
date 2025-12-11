"""LangGraph workflow orchestration for Virtual Lab."""

from typing import Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.virtuallab_workflow.state import ResearchState
from src.virtuallab_workflow.classifier import classify_question_node, route_by_question_type
from src.virtuallab_workflow.nodes import (
    virtual_lab_wetlab_node,
    virtual_lab_computational_node,
    virtual_lab_literature_node,
    virtual_lab_general_node,
    human_review_node,
    should_continue_after_review
)
from src.virtuallab_workflow.nodes_consensus import virtual_lab_consensus_node


def create_research_workflow(enable_human_review: bool = False) -> StateGraph:
    """Create the complete LangGraph workflow for research questions.
    
    Workflow structure:
    1. Classifier: Analyze question type and complexity
    2. Conditional routing to appropriate Virtual Lab
    3. Optional human review checkpoint
    4. Final answer
    
    Args:
        enable_human_review: If True, adds human-in-the-loop checkpoint
        
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the graph
    workflow = StateGraph(ResearchState)
    
    # Add classifier node
    workflow.add_node("classifier", classify_question_node)
    
    # Add Virtual Lab nodes for each question type
    workflow.add_node("virtual_lab_wetlab", virtual_lab_wetlab_node)
    workflow.add_node("virtual_lab_computational", virtual_lab_computational_node)
    workflow.add_node("virtual_lab_literature", virtual_lab_literature_node)
    workflow.add_node("virtual_lab_general", virtual_lab_general_node)
    
    # Add human review node if enabled
    if enable_human_review:
        workflow.add_node("human_review", human_review_node)
    
    # Set entry point
    workflow.set_entry_point("classifier")
    
    # Add conditional routing from classifier to Virtual Lab nodes
    workflow.add_conditional_edges(
        "classifier",
        route_by_question_type,
        {
            "virtual_lab_wetlab": "virtual_lab_wetlab",
            "virtual_lab_computational": "virtual_lab_computational",
            "virtual_lab_literature": "virtual_lab_literature",
            "virtual_lab_general": "virtual_lab_general"
        }
    )
    
    # Connect Virtual Lab nodes to either human review or end
    if enable_human_review:
        # All Virtual Lab paths go to human review
        workflow.add_edge("virtual_lab_wetlab", "human_review")
        workflow.add_edge("virtual_lab_computational", "human_review")
        workflow.add_edge("virtual_lab_literature", "human_review")
        workflow.add_edge("virtual_lab_general", "human_review")
        
        # Conditional edge from human review
        workflow.add_conditional_edges(
            "human_review",
            should_continue_after_review,
            {
                "end": END,
                "revise": "classifier"  # Could also route to specific nodes
            }
        )
    else:
        # Direct to end
        workflow.add_edge("virtual_lab_wetlab", END)
        workflow.add_edge("virtual_lab_computational", END)
        workflow.add_edge("virtual_lab_literature", END)
        workflow.add_edge("virtual_lab_general", END)
    
    # Add memory checkpointing for state persistence
    memory = MemorySaver()
    
    # Compile the graph
    if enable_human_review:
        # Interrupt before human review to allow input
        app = workflow.compile(
            checkpointer=memory,
            interrupt_before=["human_review"]
        )
    else:
        app = workflow.compile(checkpointer=memory)
    
    return app


def run_research_workflow(
    question: str,
    enable_human_review: bool = False,
    thread_id: str = "default",
    verbose: bool = True
) -> dict:
    """Run a research question through the LangGraph workflow.
    
    Args:
        question: Research question to answer
        enable_human_review: Enable human-in-the-loop checkpoint
        thread_id: Unique thread ID for state persistence
        verbose: Print execution details
        
    Returns:
        Final state dictionary with answer and metadata
    """
    # Create workflow
    app = create_research_workflow(enable_human_review)
    
    # Initial state
    initial_state: ResearchState = {
        "question": question,
        "question_type": "",
        "question_complexity": "",
        "team_composition": [],
        "meeting_transcript": "",
        "requires_human_approval": enable_human_review,
        "human_feedback": None,
        "approval_status": "pending" if enable_human_review else "approved",
        "final_answer": "",
        "confidence_score": 0.0,
        "references": [],
        "execution_path": [],
        "errors": []
    }
    
    # Configure thread for state persistence
    config = {"configurable": {"thread_id": thread_id}}
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Research Question: {question}")
        print(f"Human Review: {'Enabled' if enable_human_review else 'Disabled'}")
        print(f"Thread ID: {thread_id}")
        print(f"{'='*80}\n")
    
    # Execute workflow
    try:
        # Stream execution
        for event in app.stream(initial_state, config):
            if verbose:
                for node_name, node_output in event.items():
                    print(f"\n--- Node: {node_name} ---")
                    if "execution_path" in node_output:
                        print(f"Execution Path: {' -> '.join(node_output['execution_path'])}")
                    if "question_type" in node_output:
                        print(f"Question Type: {node_output['question_type']}")
                    if "question_complexity" in node_output:
                        print(f"Complexity: {node_output['question_complexity']}")
                    if "team_composition" in node_output:
                        print(f"Team: {', '.join(node_output['team_composition'])}")
                    if "errors" in node_output and node_output["errors"]:
                        print(f"Errors: {node_output['errors']}")
        
        # Get final state
        final_state = app.get_state(config)
        
        if verbose:
            print(f"\n{'='*80}")
            print("WORKFLOW COMPLETE")
            print(f"{'='*80}")
            print(f"Execution Path: {' -> '.join(final_state.values.get('execution_path', []))}")
            print(f"\nFinal Answer:\n{final_state.values.get('final_answer', 'No answer generated')}")
            print(f"\nConfidence: {final_state.values.get('confidence_score', 0.0):.2f}")
            print(f"{'='*80}\n")
        
        return final_state.values
        
    except Exception as e:
        if verbose:
            print(f"\n❌ Workflow execution error: {str(e)}")
        return {
            "question": question,
            "final_answer": f"Error executing workflow: {str(e)}",
            "confidence_score": 0.0,
            "errors": [str(e)],
            "execution_path": ["error"]
        }


def continue_after_human_review(
    thread_id: str,
    approval_status: str = "approved",
    human_feedback: Optional[str] = None,
    verbose: bool = True
) -> dict:
    """Continue workflow execution after human review.
    
    Call this function after the workflow pauses at human_review node.
    
    Args:
        thread_id: Thread ID from initial execution
        approval_status: "approved", "rejected", or "revise"
        human_feedback: Optional feedback text from human reviewer
        verbose: Print execution details
        
    Returns:
        Final state after continuing execution
    """
    # Recreate workflow with human review enabled
    app = create_research_workflow(enable_human_review=True)
    
    # Configure thread
    config = {"configurable": {"thread_id": thread_id}}
    
    # Update state with human decision
    current_state = app.get_state(config)
    
    # Add human feedback to state
    updated_state = {
        "approval_status": approval_status,
        "human_feedback": human_feedback
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Resuming workflow with human decision: {approval_status}")
        if human_feedback:
            print(f"Feedback: {human_feedback}")
        print(f"{'='*80}\n")
    
    # Continue execution with updated state
    try:
        # Update state and continue
        app.update_state(config, updated_state)
        
        for event in app.stream(None, config):
            if verbose:
                for node_name, node_output in event.items():
                    print(f"\n--- Node: {node_name} ---")
                    if "final_answer" in node_output:
                        print(f"Answer: {node_output['final_answer'][:200]}...")
        
        final_state = app.get_state(config)
        
        if verbose:
            print(f"\n{'='*80}")
            print("WORKFLOW COMPLETE (AFTER REVIEW)")
            print(f"{'='*80}\n")
        
        return final_state.values
        
    except Exception as e:
        if verbose:
            print(f"\n❌ Error continuing workflow: {str(e)}")
        return {
            "final_answer": f"Error continuing workflow: {str(e)}",
            "errors": [str(e)]
        }


def create_consensus_workflow() -> StateGraph:
    """Create a workflow that uses the consensus node."""
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("classifier", classify_question_node)
    workflow.add_node("consensus_lab", virtual_lab_consensus_node)
    
    # Set entry point
    workflow.set_entry_point("classifier")
    
    # Simple linear flow for this test: Classifier -> Consensus -> End
    # (In a full version, we might route to different consensus configurations)
    workflow.add_edge("classifier", "consensus_lab")
    workflow.add_edge("consensus_lab", END)
    
    return workflow.compile(checkpointer=MemorySaver())


def run_consensus_workflow(
    question: str,
    team_size: int = 3,
    num_rounds: int = 2,
    thread_id: str = "default",
    verbose: bool = True
) -> dict:
    """Run a research question through the Consensus LangGraph workflow.
    
    Args:
        question: Research question to answer
        team_size: Number of specialists per meeting
        num_rounds: Number of discussion rounds
        thread_id: Unique thread ID for state persistence
        verbose: Print execution details
        
    Returns:
        Final state dictionary with answer and metadata
    """
    # Create workflow
    app = create_consensus_workflow()
    
    # Initial state
    initial_state: ResearchState = {
        "question": question,
        "team_size": team_size,
        "num_rounds": num_rounds,
        "question_type": "",
        "question_complexity": "",
        "team_composition": [],
        "meeting_transcript": "",
        "requires_human_approval": False,
        "human_feedback": None,
        "approval_status": "approved",
        "final_answer": "",
        "confidence_score": 0.0,
        "references": [],
        "execution_path": [],
        "errors": []
    }
    
    # Configure thread for state persistence
    config = {"configurable": {"thread_id": thread_id}}
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Research Question: {question}")
        print(f"Mode: Consensus (Combined)")
        print(f"Thread ID: {thread_id}")
        print(f"{'='*80}\n")
    
    # Execute workflow
    try:
        # Stream execution
        for event in app.stream(initial_state, config):
            if verbose:
                for node_name, node_output in event.items():
                    print(f"\n--- Node: {node_name} ---")
                    if "execution_path" in node_output:
                        print(f"Execution Path: {' -> '.join(node_output['execution_path'])}")
                    if "question_type" in node_output:
                        print(f"Question Type: {node_output['question_type']}")
                    if "meeting_transcript" in node_output:
                        print("Consensus Meeting Complete")
                        print(f"Agreement Score: {node_output.get('confidence_score', 'N/A')}")
        
        # Get final state
        final_state = app.get_state(config)
        
        if verbose:
            print(f"\n{'='*80}")
            print("WORKFLOW COMPLETE")
            print(f"{'='*80}")
            print(f"Execution Path: {' -> '.join(final_state.values.get('execution_path', []))}")
            print(f"\nFinal Answer:\n{final_state.values.get('final_answer', 'No answer generated')}")
            print(f"\nConfidence: {final_state.values.get('confidence_score', 0.0):.2f}")
            print(f"{'='*80}\n")
        
        return final_state.values
        
    except Exception as e:
        if verbose:
            print(f"\n❌ Workflow execution error: {str(e)}")
        return {
            "question": question,
            "final_answer": f"Error executing workflow: {str(e)}",
            "confidence_score": 0.0,
            "errors": [str(e)],
            "execution_path": ["error"]
        }
