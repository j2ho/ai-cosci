"""Virtual Lab execution nodes for different question types."""

import os
from src.virtuallab_workflow.state import ResearchState
from src.agent.meeting import VirtualLabMeeting


# Specialist profiles for different question types
SPECIALIST_PROFILES = {
    "wet_lab": [
        {
            "role": "Experimental Design Specialist",
            "expertise": "Design rigorous experiments using appropriate controls, statistical power, and reproducible methodologies"
        },
        {
            "role": "Molecular Biology Specialist",
            "expertise": "Expert in molecular techniques including CRISPR, cloning, protein expression, and biochemical assays"
        },
        {
            "role": "Cell Biology Specialist",
            "expertise": "Expertise in cell culture, microscopy, flow cytometry, and cellular assays"
        },
        {
            "role": "Model Organism Specialist",
            "expertise": "Experience with animal models, tissue handling, and in vivo experimentation"
        }
    ],
    "computational": [
        {
            "role": "Bioinformatics Specialist",
            "expertise": "Expert in genomics, transcriptomics, sequence analysis, and biological databases"
        },
        {
            "role": "Biostatistics Specialist",
            "expertise": "Statistical modeling, experimental design, differential expression, and data quality control"
        },
        {
            "role": "Machine Learning Specialist",
            "expertise": "Predictive modeling, feature selection, deep learning, and algorithm optimization"
        },
        {
            "role": "Data Science Specialist",
            "expertise": "Data integration, visualization, pipeline development, and reproducible workflows"
        }
    ],
    "literature": [
        {
            "role": "Literature Review Specialist",
            "expertise": "Systematic literature search, evidence synthesis, and meta-analysis"
        },
        {
            "role": "Mechanism Expert",
            "expertise": "Deep understanding of biological mechanisms, pathways, and molecular interactions"
        },
        {
            "role": "Clinical Translation Specialist",
            "expertise": "Connecting basic research to clinical applications and therapeutic strategies"
        }
    ],
    "general": [
        {
            "role": "Generalist Researcher",
            "expertise": "Broad biomedical knowledge spanning wet lab, computational, and literature review"
        },
        {
            "role": "Methodology Expert",
            "expertise": "Cross-domain expertise in experimental design, data analysis, and literature synthesis"
        },
        {
            "role": "Integration Specialist",
            "expertise": "Connecting diverse approaches and synthesizing multi-modal evidence"
        }
    ]
}


def _create_virtual_lab(question_type: str, question: str, team_size: int, num_rounds: int) -> dict:
    """Execute Virtual Lab meeting with appropriate configuration.
    
    Args:
        question_type: Type of question (wet_lab, computational, literature, general)
        question: The research question
        team_size: Number of specialists to include
        num_rounds: Number of discussion rounds
        
    Returns:
        Result dictionary from meeting
    """
    # Get data directory from environment
    data_dir = os.getenv("DATABASE_DIR", "/home.galaxy4/sumin/project/aisci/Competition_Data")
    
    # Add context hint to question to guide PI's team design
    hints = {
        "wet_lab": "Focus on experimental design and laboratory techniques.",
        "computational": "Focus on bioinformatics, data analysis, and computational methods.",
        "literature": "Focus on literature review and mechanism synthesis.",
        "general": "Provide broad, cross-domain expertise."
    }
    
    hint = hints.get(question_type, "")
    enhanced_question = f"{question}\n\n[Context: {hint}]" if hint else question
    
    # Create meeting
    meeting = VirtualLabMeeting(
        user_question=enhanced_question,
        model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"),
        provider=os.getenv("PROVIDER", "openrouter"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_team_size=team_size,
        verbose=True,
        data_dir=data_dir
    )
    
    # Run meeting
    result = meeting.run_meeting(num_rounds=num_rounds)
    
    return result


def virtual_lab_wetlab_node(state: ResearchState) -> dict:
    """Execute Virtual Lab for wet lab questions.
    
    Uses experimental specialists focused on laboratory techniques.
    """
    question = state["question"]
    team_size = state.get("team_size", 3)
    num_rounds = state.get("num_rounds", 2)
    
    try:
        result = _create_virtual_lab("wet_lab", question, team_size, num_rounds)
        
        return {
            "meeting_transcript": result,
            "final_answer": result,  # The meeting result is the answer
            "confidence_score": 0.7,
            "team_composition": ["Experimental Design", "Molecular Biology", "Cell Biology"][:team_size],
            "execution_path": ["virtual_lab_wetlab"]
        }
    except Exception as e:
        return {
            "meeting_transcript": "",
            "final_answer": f"Error executing wet lab meeting: {str(e)}",
            "confidence_score": 0.0,
            "team_composition": [],
            "execution_path": ["virtual_lab_wetlab"],
            "errors": [str(e)]
        }


def virtual_lab_computational_node(state: ResearchState) -> dict:
    """Execute Virtual Lab for computational questions.
    
    Uses bioinformatics and data science specialists.
    """
    question = state["question"]
    team_size = state.get("team_size", 3)
    num_rounds = state.get("num_rounds", 2)
    
    try:
        result = _create_virtual_lab("computational", question, team_size, num_rounds)
        
        return {
            "meeting_transcript": result,
            "final_answer": result,
            "confidence_score": 0.7,
            "team_composition": ["Bioinformatics", "Biostatistics", "Data Science"][:team_size],
            "execution_path": ["virtual_lab_computational"]
        }
    except Exception as e:
        return {
            "meeting_transcript": "",
            "final_answer": f"Error executing computational meeting: {str(e)}",
            "confidence_score": 0.0,
            "team_composition": [],
            "execution_path": ["virtual_lab_computational"],
            "errors": [str(e)]
        }


def virtual_lab_literature_node(state: ResearchState) -> dict:
    """Execute Virtual Lab for literature review questions.
    
    Uses literature and mechanism specialists.
    """
    question = state["question"]
    team_size = state.get("team_size", 3)
    num_rounds = state.get("num_rounds", 2)
    
    try:
        result = _create_virtual_lab("literature", question, team_size, num_rounds)
        
        return {
            "meeting_transcript": result,
            "final_answer": result,
            "confidence_score": 0.7,
            "team_composition": ["Literature Review", "Mechanism Expert", "Clinical Translation"][:team_size],
            "execution_path": ["virtual_lab_literature"]
        }
    except Exception as e:
        return {
            "meeting_transcript": "",
            "final_answer": f"Error executing literature meeting: {str(e)}",
            "confidence_score": 0.0,
            "team_composition": [],
            "execution_path": ["virtual_lab_literature"],
            "errors": [str(e)]
        }


def virtual_lab_general_node(state: ResearchState) -> dict:
    """Execute Virtual Lab for general questions.
    
    Uses generalist researchers with broad expertise.
    """
    question = state["question"]
    team_size = state.get("team_size", 3)
    num_rounds = state.get("num_rounds", 2)
    
    try:
        result = _create_virtual_lab("general", question, team_size, num_rounds)
        
        return {
            "meeting_transcript": result,
            "final_answer": result,
            "confidence_score": 0.7,
            "team_composition": ["Generalist Researcher", "Methodology Expert", "Integration Specialist"][:team_size],
            "execution_path": ["virtual_lab_general"]
        }
    except Exception as e:
        return {
            "meeting_transcript": "",
            "final_answer": f"Error executing general meeting: {str(e)}",
            "confidence_score": 0.0,
            "team_composition": [],
            "execution_path": ["virtual_lab_general"],
            "errors": [str(e)]
        }


def human_review_node(state: ResearchState) -> dict:
    """Human-in-the-loop review checkpoint.
    
    This node pauses execution and waits for human approval.
    Controlled by interrupt_before in workflow configuration.
    
    Args:
        state: Current research state
        
    Returns:
        Updated state with human feedback and approval status
    """
    # This is a pass-through node - actual human interaction happens
    # via LangGraph's interrupt mechanism
    
    # Check if human has provided feedback
    human_feedback = state.get("human_feedback")
    approval_status = state.get("approval_status", "approved")
    
    return {
        "execution_path": ["human_review"],
        "approval_status": approval_status,
        "human_feedback": human_feedback
    }


def should_continue_after_review(state: ResearchState) -> str:
    """Conditional edge after human review.
    
    Routes based on human approval decision.
    
    Args:
        state: Current research state
        
    Returns:
        Next node name ("end" or "revise")
    """
    approval_status = state.get("approval_status", "approved")
    
    if approval_status == "approved":
        return "end"
    elif approval_status == "revise":
        # Could route back to question classifier or specific Virtual Lab
        return "revise"
    else:
        # Default to end
        return "end"
