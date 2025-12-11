"""Question classification for intelligent routing."""

import os
from src.virtuallab_workflow.state import ResearchState
from src.agent.openrouter_client import OpenRouterClient
from src.agent.anthropic_client import AnthropicClient


def classify_question_node(state: ResearchState) -> dict:
    """Classify research question to route to appropriate specialist team.
    
    Analyzes the question to determine:
    1. Type: wet_lab, computational, literature, or general
    2. Complexity: simple, moderate, or complex
    
    Args:
        state: Current research state
        
    Returns:
        Updated state with question_type and question_complexity
    """
    question = state["question"]
    
    # Create a client for classification
    provider = os.getenv("LANGGRAPH_PROVIDER", "openrouter")
    model = os.getenv("LANGGRAPH_MODEL", os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"))
    api_key = os.getenv("OPENROUTER_API_KEY") if provider == "openrouter" else os.getenv("ANTHROPIC_API_KEY")
    
    if provider == "openrouter":
        client = OpenRouterClient(api_key=api_key, model=model)
    else:
        client = AnthropicClient(api_key=api_key, model=model)
    
    # Classification prompt
    classification_prompt = f"""Analyze this biomedical research question and classify it:

Question: "{question}"

Provide TWO classifications:

1. QUESTION TYPE (choose ONE):
   - wet_lab: Requires experimental/laboratory techniques (e.g., CRISPR gene editing, cell culture, animal models, protein purification, Western blots, flow cytometry, immunohistochemistry)
   - computational: Requires data analysis, bioinformatics, machine learning, or computational modeling (e.g., RNA-seq analysis, drug discovery screening, protein structure prediction, pathway analysis, statistical modeling)
   - literature: Primarily requires literature review, synthesis of existing research, or theoretical analysis (e.g., "What is known about...", "Review mechanisms of...", "Compare approaches for...")
   - general: Broad question requiring diverse expertise across multiple domains

2. COMPLEXITY (choose ONE):
   - simple: Single-step analysis or straightforward question with clear methodology
   - moderate: Multi-step analysis requiring 2-3 different approaches
   - complex: Multi-faceted problem requiring extensive analysis, multiple data sources, and cross-domain expertise

Respond ONLY in this exact format (no explanation):
TYPE: <wet_lab|computational|literature|general>
COMPLEXITY: <simple|moderate|complex>"""

    try:
        # Call LLM
        response = client.create_message(
            messages=[{"role": "user", "content": classification_prompt}],
            max_tokens=100,
            temperature=0.3  # Low temperature for consistent classification
        )
        
        # Extract text from response
        text = client.get_response_text(response)
        
        # Parse response
        question_type = "general"
        complexity = "moderate"
        
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("TYPE:"):
                type_value = line.split("TYPE:")[1].strip().lower()
                if type_value in ["wet_lab", "computational", "literature", "general"]:
                    question_type = type_value
            elif line.startswith("COMPLEXITY:"):
                complexity_value = line.split("COMPLEXITY:")[1].strip().lower()
                if complexity_value in ["simple", "moderate", "complex"]:
                    complexity = complexity_value
        
        # Determine team size and rounds based on complexity
        if complexity == "simple":
            team_size = 2
            num_rounds = 1
        elif complexity == "moderate":
            team_size = 3
            num_rounds = 2
        else:  # complex
            team_size = 4
            num_rounds = 3
        
        return {
            "question_type": question_type,
            "question_complexity": complexity,
            "team_size": team_size,
            "num_rounds": num_rounds,
            "execution_path": ["classifier"]
        }
        
    except Exception as e:
        # Fallback to safe defaults
        return {
            "question_type": "general",
            "question_complexity": "moderate",
            "team_size": 3,
            "num_rounds": 2,
            "execution_path": ["classifier"],
            "errors": [f"Classification error: {str(e)}"]
        }


def route_by_question_type(state: ResearchState) -> str:
    """Conditional edge routing function.
    
    Routes to different Virtual Lab configurations based on question type.
    
    Args:
        state: Current research state
        
    Returns:
        Name of the next node to execute
    """
    question_type = state.get("question_type", "general")
    
    # Map question types to node names
    routing_map = {
        "wet_lab": "virtual_lab_wetlab",
        "computational": "virtual_lab_computational",
        "literature": "virtual_lab_literature",
        "general": "virtual_lab_general"
    }
    
    return routing_map.get(question_type, "virtual_lab_general")
