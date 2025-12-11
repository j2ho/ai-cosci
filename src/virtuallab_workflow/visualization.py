"""Workflow visualization and debugging utilities."""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from src.virtuallab_workflow.workflow import create_research_workflow


def visualize_workflow(
    enable_human_review: bool = False,
    output_format: str = "mermaid",
    output_file: Optional[str] = None
) -> str:
    """Generate workflow visualization diagram.
    
    Args:
        enable_human_review: Include human review node in diagram
        output_format: "mermaid" or "ascii" (only mermaid supported currently)
        output_file: Optional file path to save diagram
        
    Returns:
        Diagram string in requested format
    """
    # Create workflow
    app = create_research_workflow(enable_human_review)
    
    # Get graph object
    graph = app.get_graph()
    
    # Generate Mermaid diagram
    if output_format == "mermaid":
        diagram = graph.draw_mermaid()
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(diagram)
        print(f"Workflow diagram saved to: {output_file}")
    
    return diagram


def export_execution_trace(
    state: Dict[str, Any],
    output_file: str,
    format: str = "json"
) -> None:
    """Export execution trace to file for debugging.
    
    Args:
        state: Final workflow state dictionary
        output_file: Path to save trace
        format: "json" or "txt"
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        # Save full state as JSON
        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        print(f"Execution trace saved to: {output_file}")
        
    elif format == "txt":
        # Save human-readable trace
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("WORKFLOW EXECUTION TRACE\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Question: {state.get('question', 'N/A')}\n\n")
            
            f.write(f"Classification:\n")
            f.write(f"  Type: {state.get('question_type', 'N/A')}\n")
            f.write(f"  Complexity: {state.get('question_complexity', 'N/A')}\n\n")
            
            f.write(f"Execution Path:\n")
            for step in state.get('execution_path', []):
                f.write(f"  -> {step}\n")
            f.write("\n")
            
            f.write(f"Team Composition:\n")
            for member in state.get('team_composition', []):
                f.write(f"  - {member}\n")
            f.write("\n")
            
            if state.get('meeting_transcript'):
                f.write(f"Meeting Transcript:\n")
                f.write("-"*80 + "\n")
                f.write(state['meeting_transcript'])
                f.write("\n" + "-"*80 + "\n\n")
            
            if state.get('human_feedback'):
                f.write(f"Human Feedback:\n")
                f.write(f"  Status: {state.get('approval_status', 'N/A')}\n")
                f.write(f"  Feedback: {state['human_feedback']}\n\n")
            
            f.write(f"Final Answer:\n")
            f.write("-"*80 + "\n")
            f.write(state.get('final_answer', 'No answer generated'))
            f.write("\n" + "-"*80 + "\n\n")
            
            f.write(f"Confidence Score: {state.get('confidence_score', 0.0):.2f}\n\n")
            
            if state.get('errors'):
                f.write(f"Errors:\n")
                for error in state['errors']:
                    f.write(f"  - {error}\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
        
        print(f"Execution trace saved to: {output_file}")
    else:
        raise ValueError(f"Unsupported format: {format}")


def print_workflow_summary() -> None:
    """Print a summary of the workflow structure."""
    print("\n" + "="*80)
    print("LANGGRAPH WORKFLOW SUMMARY")
    print("="*80 + "\n")
    
    print("NODES:")
    print("  1. classifier - Analyze question type and complexity")
    print("  2. virtual_lab_wetlab - Execute with experimental specialists")
    print("  3. virtual_lab_computational - Execute with bioinformatics specialists")
    print("  4. virtual_lab_literature - Execute with literature review specialists")
    print("  5. virtual_lab_general - Execute with generalist researchers")
    print("  6. human_review - Human-in-the-loop checkpoint (optional)\n")
    
    print("ROUTING:")
    print("  classifier -> [conditional]")
    print("    - wet_lab questions -> virtual_lab_wetlab")
    print("    - computational questions -> virtual_lab_computational")
    print("    - literature questions -> virtual_lab_literature")
    print("    - general questions -> virtual_lab_general\n")
    
    print("  virtual_lab_* -> [if human review enabled]")
    print("    -> human_review")
    print("    -> [conditional] approved -> END")
    print("    -> [conditional] revise -> classifier\n")
    
    print("  virtual_lab_* -> [if human review disabled]")
    print("    -> END\n")
    
    print("STATE FIELDS:")
    print("  - question: Research question text")
    print("  - question_type: wet_lab | computational | literature | general")
    print("  - question_complexity: simple | moderate | complex")
    print("  - team_composition: List of specialist roles")
    print("  - meeting_transcript: Full conversation transcript")
    print("  - final_answer: Synthesized answer from PI")
    print("  - confidence_score: Answer confidence (0.0-1.0)")
    print("  - execution_path: List of nodes executed")
    print("  - human_feedback: Optional reviewer feedback")
    print("  - approval_status: approved | rejected | revise\n")
    
    print("="*80 + "\n")


def compare_workflows(
    question: str,
    output_dir: str = "workflow_comparison"
) -> None:
    """Generate diagrams comparing workflow with/without human review.
    
    Args:
        question: Sample question for documentation
        output_dir: Directory to save comparison diagrams
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate diagram without human review
    diagram_basic = visualize_workflow(
        enable_human_review=False,
        output_file=str(output_path / "workflow_basic.mmd")
    )
    
    # Generate diagram with human review
    diagram_human = visualize_workflow(
        enable_human_review=True,
        output_file=str(output_path / "workflow_human_review.mmd")
    )
    
    # Create README
    readme_content = f"""# Workflow Diagrams

Generated from question: "{question}"

## Basic Workflow (No Human Review)

```mermaid
{diagram_basic}
```

See [workflow_basic.mmd](workflow_basic.mmd) for editable diagram.

## Workflow with Human Review

```mermaid
{diagram_human}
```

See [workflow_human_review.mmd](workflow_human_review.mmd) for editable diagram.

## Key Differences

1. **Basic Workflow**: Direct execution from classifier -> Virtual Lab -> END
2. **Human Review**: Adds checkpoint: classifier -> Virtual Lab -> human_review -> [approved] END or [revise] classifier

## Usage

To view these diagrams:
1. Copy the Mermaid code to https://mermaid.live
2. Or use a Mermaid-compatible Markdown viewer
3. Or install Mermaid CLI: `npm install -g @mermaid-js/mermaid-cli`
"""
    
    readme_path = output_path / "README.md"
    readme_path.write_text(readme_content)
    
    print(f"\n✓ Workflow comparison saved to: {output_dir}/")
    print(f"  - workflow_basic.mmd")
    print(f"  - workflow_human_review.mmd")
    print(f"  - README.md\n")
