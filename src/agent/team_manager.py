"""Team management and dynamic team composition for Virtual Lab."""

import json
from typing import Any, Optional
from src.agent.agent import AgentPersona


def create_research_team(
    user_question: str,
    client: Any,
    max_team_size: int = 3
) -> list[dict[str, str]]:
    """Use PI persona to dynamically design a research team for the question.

    The PI agent analyzes the research question and determines which
    specialist roles are needed to solve it effectively.

    Args:
        user_question: The research question to solve
        client: The LLM client (AnthropicClient or OpenRouterClient)
        max_team_size: Maximum number of specialist agents (default: 3)

    Returns:
        List of agent specification dictionaries with keys:
        - title: Agent's role title (e.g., "Computational Biologist")
        - expertise: Domain expertise description
        - goal: What this agent should accomplish
        - role: Specific responsibilities in the meeting
    """
    # Construct the PI prompt for team design
    pi_prompt = f"""You are a Principal Investigator (PI) designing a research team.

**Research Question:**
"{user_question}"

**Your Task:**
Analyze this question and design a team of {max_team_size} specialized scientific agents to solve it.
Each agent should bring unique, complementary expertise.

For each agent, provide:
1. **title**: Role title (e.g., "Immunologist", "Data Scientist", "Structural Biologist")
2. **expertise**: Specific domain knowledge they bring
3. **goal**: What they should accomplish in this research
4. **role**: Their specific responsibilities in the team meeting

**Important Guidelines:**
- Choose specialists whose expertise DIRECTLY addresses the research question
- Avoid redundant roles - each agent should contribute something unique
- Consider: Does this need computational analysis? Literature review? Domain expertise? Data analysis?
- Prefer specialists over generalists
- Common useful roles: Computational Biologist, Bioinformatician, Immunologist, Oncologist,
  Data Scientist, Structural Biologist, Systems Biologist, Drug Discovery Specialist, etc.

**Output Format:**
Return ONLY a valid JSON array of {max_team_size} agent specifications. No explanation.

Example format:
[
    {{
        "title": "Computational Biologist",
        "expertise": "Sequence analysis, protein structure prediction, bioinformatics algorithms",
        "goal": "Design and validate the binder sequence using computational methods",
        "role": "Generate candidate sequences and predict their biophysical properties"
    }},
    {{
        "title": "Immunologist",
        "expertise": "T-cell biology, immune checkpoint mechanisms, cancer immunology",
        "goal": "Validate biological relevance and mechanism of action",
        "role": "Assess immunological plausibility and potential therapeutic implications"
    }},
    {{
        "title": "Data Scientist",
        "expertise": "Statistical analysis, machine learning, biomedical databases",
        "goal": "Mine databases for relevant precedents and validate predictions",
        "role": "Query databases and analyze patterns in existing data"
    }}
]

Now design the team for the research question above. Output JSON only:"""

    # Call LLM to get team design
    try:
        from src.agent.anthropic_client import AnthropicClient

        # Build the call parameters
        messages = [{"role": "user", "content": pi_prompt}]

        call_params = {
            "messages": messages,
            "tools": [],  # PI doesn't use tools for team design
            "temperature": 0.3,  # Lower temperature for consistent team design
            "max_tokens": 1500,
        }

        # Add system prompt if using Anthropic
        if isinstance(client, AnthropicClient):
            call_params["system"] = "You are a Principal Investigator expert at assembling research teams. Always output valid JSON."

        response = client.create_message(**call_params)
        text = client.get_response_text(response)

        # Parse JSON response
        # Extract JSON from text (in case there's extra text)
        json_start = text.find('[')
        json_end = text.rfind(']') + 1

        if json_start == -1 or json_end == 0:
            # Fallback: No JSON found, return default team
            return _get_default_team()

        json_text = text[json_start:json_end]
        team_specs = json.loads(json_text)

        # Validate that we got the right structure
        if not isinstance(team_specs, list) or len(team_specs) == 0:
            return _get_default_team()

        # Ensure each spec has required fields
        valid_specs = []
        for spec in team_specs[:max_team_size]:
            if all(key in spec for key in ["title", "expertise", "goal", "role"]):
                valid_specs.append(spec)

        if len(valid_specs) == 0:
            return _get_default_team()

        return valid_specs

    except Exception as e:
        # If anything goes wrong, return a sensible default team
        print(f"Warning: Team design failed ({e}), using default team")
        return _get_default_team()


def _get_default_team() -> list[dict[str, str]]:
    """Get a default general-purpose research team.

    Returns:
        Default team specification
    """
    return [
        {
            "title": "Bioinformatician",
            "expertise": "Computational biology, sequence analysis, database queries, Python programming",
            "goal": "Perform computational analyses and query biomedical databases",
            "role": "Execute data analysis and generate insights from databases"
        },
        {
            "title": "Biomedical Scientist",
            "expertise": "Molecular biology, disease mechanisms, drug discovery, literature review",
            "goal": "Provide biological context and validate findings against literature",
            "role": "Interpret results in biological context and search relevant literature"
        },
        {
            "title": "Data Scientist",
            "expertise": "Statistical analysis, data visualization, pattern recognition",
            "goal": "Analyze patterns in data and validate statistical significance",
            "role": "Perform statistical analyses and create visualizations"
        }
    ]


def create_pi_persona() -> AgentPersona:
    """Create the Principal Investigator persona.

    Returns:
        AgentPersona for the PI role
    """
    return AgentPersona(
        title="Principal Investigator",
        expertise="Research leadership, experimental design, scientific synthesis, grant writing",
        goal="Lead the research team, synthesize findings, and provide strategic direction",
        role="Meeting leader - frame questions, synthesize contributions, and guide the team toward conclusions"
    )


def create_critic_persona() -> AgentPersona:
    """Create the Scientific Critic persona.

    Returns:
        AgentPersona for the Critic role
    """
    return AgentPersona(
        title="Scientific Critic",
        expertise="Peer review, experimental validation, error detection, methodological rigor",
        goal="Ensure scientific accuracy and identify flaws in reasoning or analysis",
        role="Critique proposals and findings - point out errors, gaps, and unsupported claims. Do not generate solutions."
    )
