# Comparison: Original Virtual Lab vs. CoScientist Virtual Lab

## Overview

This document compares the **original Virtual Lab** implementation (from the Nature paper) with our **CoScientist Virtual Lab** implementation.

## Side-by-Side Comparison

| Feature | Original Virtual Lab | CoScientist Virtual Lab |
|---------|---------------------|------------------------|
| **Paper** | [Nature 2025](https://doi.org/10.1038/s41586-025-09442-9) | Based on Virtual Lab methodology |
| **Primary Use Case** | Nanobody design for SARS-CoV-2 | General biomedical research questions |
| **LLM Provider** | OpenAI (GPT-4o) | Anthropic (Claude) or OpenRouter |
| **API Framework** | OpenAI Assistants API (threads) | Anthropic Messages API / OpenRouter |
| **Team Design** | Static (predefined in code) | **Dynamic (PI creates team per question)** |
| **Default Model** | GPT-4o | Claude Sonnet 4 |

## Detailed Architectural Differences

### 1. Agent Architecture

**Original Virtual Lab:**
```python
class Agent:
    def __init__(self, title, expertise, goal, role, model):
        self.title = title
        self.expertise = expertise
        self.goal = goal
        self.role = role
        self.model = model

    @property
    def prompt(self):
        return f"You are a {self.title}. Your expertise is in {self.expertise}..."
```

- Simple data class
- No built-in tool capabilities
- Tools added via OpenAI Assistants API

**CoScientist Virtual Lab:**
```python
class ScientificAgent(BioinformaticsAgent):
    def __init__(self, persona: AgentPersona, api_key, model, provider):
        super().__init__(api_key, model, provider)
        self.persona = persona
        self.tools = {execute_python, search_pubmed, query_database, read_file}

    def get_system_prompt(self) -> str:
        return f"You are {self.persona.title}..."
```

- Extends existing BioinformaticsAgent
- Built-in tool execution capabilities
- Inherits full agent loop (run method, conversation history, etc.)
- **More sophisticated tool usage** (Python execution, databases)

### 2. Team Composition

**Original Virtual Lab:**
```python
# Teams are STATIC - predefined in code
PRINCIPAL_INVESTIGATOR = Agent(
    title="Principal Investigator",
    expertise="running a science research lab",
    goal="perform research...",
    role="lead a team...",
    model=DEFAULT_MODEL,
)

# Usage: manually create team members
team_members = (immunologist, structural_biologist, computational_biologist)
```

- Teams are hardcoded before the meeting
- Researcher manually selects appropriate specialists
- Same team structure for similar questions

**CoScientist Virtual Lab:**
```python
# Teams are DYNAMIC - created per question
def create_research_team(user_question, client, max_team_size=3):
    """PI agent analyzes question and designs the team."""
    # PI receives the question and decides who's needed
    team_specs = pi_agent.run("Design a team for: {question}")
    # Returns specialists tailored to THIS specific question
    return team_specs

# Example output for "Design PD-L1 binder":
# -> Computational Biologist, Immunologist, Data Scientist
```

- **Teams designed dynamically** for each question
- PI analyzes research question and selects appropriate expertise
- Different questions → different team compositions
- More flexible and adaptive

### 3. Meeting Structure

**Original Virtual Lab:**

Two meeting types:

**A. Team Meeting:**
```
User sets agenda → Team Lead opens → Each member speaks (Round 1) →
Team Lead synthesizes → Each member speaks (Round 2) → ... →
Team Lead final summary
```

**B. Individual Meeting:**
```
User sets agenda → Single Agent responds →
Critic reviews → Agent revises →
Critic reviews → Agent revises (repeat)
```

**CoScientist Virtual Lab:**

One meeting type (team meeting):
```
User asks question → PI designs team → PI opens meeting →
Round 1: [Specialist 1, Specialist 2, Specialist 3, Critic, PI synthesis] →
Round 2: [Specialist 1, Specialist 2, Specialist 3, Critic] →
PI final synthesis
```

- Simpler: Only one meeting type
- Critic integrated into every round (not separate meeting)
- No individual meetings (could be added)

### 4. Conversation Management

**Original Virtual Lab:**
```python
# Uses OpenAI Threads (stateful, persistent)
thread = client.beta.threads.create()
client.beta.threads.messages.create(thread_id=thread.id, ...)

# Full conversation history maintained by OpenAI
# All agents see entire thread
```

- Conversation persists in OpenAI's infrastructure
- Stateful across API calls
- Full history always visible
- Can reference any previous message

**CoScientist Virtual Lab:**
```python
# In-memory conversation history (stateless)
self.conversation_history = []
self.meeting_transcript = []

# Context management: last N messages
def _build_context(self, last_n=5):
    return recent_transcript[-last_n:]
```

- Conversation stored locally
- Each agent's run() method is independent
- Context passed explicitly (last N messages)
- More control over context window
- Easier to debug and inspect

### 5. Tools Available

**Original Virtual Lab:**
```python
# Only PubMed search
PUBMED_TOOL_DESCRIPTION = {
    "type": "function",
    "function": {
        "name": "pubmed_search",
        "description": "Get abstracts from PubMed...",
        ...
    }
}
```

- **Single tool**: PubMed literature search
- Limited computational capabilities
- Designed for literature-heavy tasks

**CoScientist Virtual Lab:**
```python
# Four comprehensive tools
tools = {
    "execute_python": execute_python,      # Run Python code (pandas, numpy, biopython)
    "search_pubmed": search_pubmed,        # Search scientific literature
    "query_database": query_database,      # Query BindingDB, DrugBank, Pharos, etc.
    "read_file": read_file,                # Read local data files
}
```

- **Multiple tools** for different needs
- **Python execution**: Data analysis, visualization, sequence design
- **Database access**: Drug databases, protein databases, GWAS
- **File I/O**: Work with local datasets
- Better suited for computational + experimental design

### 6. Prompting & Output Structure

**Original Virtual Lab:**

Highly structured with explicit requirements:
```python
team_meeting_start_prompt(
    agenda="Design nanobodies...",
    agenda_questions=(
        "What computational pipeline should be used?",
        "What filters should be applied?",
    ),
    agenda_rules=(
        "Code must be self-contained",
        "No pseudocode allowed",
    ),
)

# Expected output format:
# ### Agenda
# ### Team Member Input
# ### Recommendation
# ### Answers (to agenda questions)
# ### Next Steps
```

- Very structured agenda system
- Explicit questions to answer
- Strict formatting requirements
- Designed for reproducible computational pipelines

**CoScientist Virtual Lab:**

Flexible research-question driven:
```python
run_virtual_lab(
    question="Design a PD-L1 binder for TNBC",
    num_rounds=2,
    max_team_size=3
)

# More flexible output:
# - PI frames the problem
# - Specialists contribute freely
# - PI synthesizes naturally
```

- Natural research questions (not formal agendas)
- Flexible contribution format
- Less rigid structure
- Better for exploratory research

### 7. API Calls & Implementation

**Original Virtual Lab:**
```python
# Uses OpenAI Assistants API
assistant = client.beta.assistants.create(
    name=agent.title,
    instructions=agent.prompt,
    model=agent.model,
    tools=[PUBMED_TOOL_DESCRIPTION],
)

run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

# Tool execution handled by OpenAI
if run.status == "requires_action":
    tool_outputs = run_tools(run)
    run = client.beta.threads.runs.submit_tool_outputs_and_poll(...)
```

- Uses Assistants API (higher-level abstraction)
- Stateful threads managed by OpenAI
- Tool execution orchestrated by API
- Simpler for OpenAI ecosystem

**CoScientist Virtual Lab:**
```python
# Uses Anthropic/OpenRouter Messages API directly
response = client.create_message(
    messages=conversation_history,
    tools=get_tool_definitions(),
    system=agent.get_system_prompt(),
)

# Tool execution handled locally
tool_calls = extract_tool_calls(response)
for tool_call in tool_calls:
    result = execute_tool(tool_call)
    conversation_history.append(result)
```

- Uses Messages API (lower-level, more control)
- Stateless conversation management
- Tool execution handled in application code
- More flexible across LLM providers
- Easier to customize and extend

### 8. Cost & Token Management

**Original Virtual Lab:**
```python
# Tracks token usage and calculates costs
token_counts = count_discussion_tokens(discussion)
print_cost_and_time(
    token_counts=token_counts,
    model=team_lead.model,
    elapsed_time=elapsed_time,
)

# Full thread history → higher token counts
```

- Detailed token tracking
- Cost estimates printed
- Full conversation history (expensive)
- Optimized for GPT-4o pricing

**CoScientist Virtual Lab:**
```python
# Context window management to control costs
context = self._build_context(last_n=5)  # Only recent messages

# Configurable rounds and team size
run_virtual_lab(
    question="...",
    num_rounds=2,      # Tune to control cost
    max_team_size=3,   # Tune to control cost
)
```

- Context truncation to reduce tokens
- User controls cost via rounds/team-size
- No built-in cost tracking (could be added)
- Optimized for Claude pricing

## Key Innovations in CoScientist Version

### 1. **Dynamic Team Design** ⭐
Our key innovation! The PI agent **creates the team on-the-fly** based on the research question.

**Why this matters:**
- Original VL: Researcher manually decides "I need an immunologist, structural biologist, and computational chemist"
- CoScientist VL: PI automatically determines "For this PD-L1 question, I need computational bio, immunology, and database expertise"
- More flexible, less manual work, adapts to question complexity

### 2. **Rich Tool Ecosystem** ⭐
Python execution + multiple databases + literature search

**Why this matters:**
- Can design molecules computationally AND validate with databases
- End-to-end workflows (literature → design → validation)
- Better for biomedical research vs. just literature review

### 3. **Multi-Provider Support**
Works with Anthropic, OpenRouter, any OpenAI-compatible API

**Why this matters:**
- Not locked into one provider
- Can use Claude (better for scientific reasoning) or GPT-4o
- Future-proof as new models emerge

### 4. **Integration with Existing Agent**
ScientificAgent extends BioinformaticsAgent

**Why this matters:**
- Leverages existing mature codebase
- All VL agents have full capabilities of existing agent
- Easy migration path (backward compatible)

## What Original Virtual Lab Does Better

### 1. **Structured Outputs**
Enforces strict format (Agenda → Team Input → Recommendation → Answers → Next Steps)

**Better for:**
- Reproducible computational pipelines
- Formal experimental protocols
- When you need specific questions answered

### 2. **Individual Meetings**
One-on-one agent sessions with iterative critic feedback

**Better for:**
- Deep dives into specific subtasks
- Code refinement workflows
- Focused individual expertise

### 3. **OpenAI Assistants API**
Simpler infrastructure (threads managed by OpenAI)

**Better for:**
- Less code to maintain
- Built-in conversation persistence
- Easier initial setup

### 4. **Proven Track Record**
Published in Nature, experimentally validated nanobodies

**Better for:**
- High-stakes research projects
- When you need citation to published methodology
- Reproducing published results

## Use Case Comparison

### Original Virtual Lab is Better For:

✅ **Nanobody/antibody design** (proven use case)
✅ **Formal computational pipeline development**
✅ **Structured experimental protocols**
✅ **When you need specific questions answered explicitly**
✅ **Code generation with strict quality requirements**
✅ **Reproducible science workflows**
✅ **OpenAI ecosystem preference**

### CoScientist Virtual Lab is Better For:

✅ **General biomedical research questions**
✅ **Exploratory research (don't know exact team needed)**
✅ **Data analysis + computational + literature synthesis**
✅ **Drug repurposing, target discovery, pathway analysis**
✅ **When you want database integration (BindingDB, DrugBank, etc.)**
✅ **Multi-provider flexibility (Claude, GPT-4, etc.)**
✅ **Integration with existing agents/tools**
✅ **Rapid prototyping and iteration**

## Example Scenarios

### Scenario 1: Design Nanobodies for New SARS-CoV-2 Variant

**Original VL:** ⭐ Better choice
- Proven methodology (published results)
- Need formal pipeline (ESM → AlphaFold → Rosetta)
- Specific agenda questions about filters, criteria
- Structured output for reproducibility

**CoScientist VL:** Also viable
- Dynamic team would create similar specialists
- Python execution can run computational tools
- More flexible exploration phase

**Winner:** Original VL (proven for this exact use case)

### Scenario 2: "Find drug repurposing candidates for Alzheimer's based on GWAS data"

**Original VL:** Limited
- Only PubMed search available
- No database access
- Would need manual data provision

**CoScientist VL:** ⭐ Better choice
- query_database for GWAS catalog
- query_database for DrugBank/Pharos
- execute_python for statistical analysis
- search_pubmed for literature validation
- Dynamic team: Data Scientist + Neuroscientist + Drug Discovery Specialist

**Winner:** CoScientist VL (better tools for this task)

### Scenario 3: "Design an experimental validation protocol for a new therapeutic target"

**Original VL:** ⭐ Better choice
- Structured agenda with specific questions
- Individual meetings for protocol refinement
- Enforced output format
- Explicit rules for protocols

**CoScientist VL:** Also viable
- Round-robin can develop protocols
- Critic provides validation
- Less formal structure

**Winner:** Original VL (better for structured protocols)

### Scenario 4: "What are the molecular mechanisms linking gut microbiome to depression?"

**Original VL:** Good
- Literature synthesis via PubMed

**CoScientist VL:** ⭐ Better choice
- Dynamic team creates: Microbiologist + Neuroscientist + Systems Biologist
- Can analyze relevant databases
- Python for pathway analysis
- More flexible exploration

**Winner:** CoScientist VL (benefits from dynamic team + tools)

## Hybrid Approach: Best of Both Worlds

You could combine both approaches:

```python
# Use CoScientist VL for exploration phase
exploratory_answer = run_virtual_lab(
    question="What's the best approach for TNBC treatment?",
    num_rounds=2,
    max_team_size=4
)

# Use Original VL for implementation phase with structured agenda
from virtual_lab import run_meeting

run_meeting(
    meeting_type="team",
    agenda="Design PD-L1 binder based on exploration findings",
    agenda_questions=(
        "What computational tools should be used?",
        "What are the binding criteria?",
        "What validation experiments are needed?",
    ),
    agenda_rules=CODING_RULES,
    contexts=(exploratory_answer,),  # Pass CoScientist results as context
    ...
)
```

## Summary Table

| Aspect | Original VL | CoScientist VL | Better For |
|--------|-------------|----------------|------------|
| **Team Design** | Static (manual) | **Dynamic (automated)** | CoScientist ⭐ |
| **Tools** | PubMed only | **Python + DBs + PubMed** | CoScientist ⭐ |
| **Structure** | **Highly structured** | Flexible | Original ⭐ |
| **API** | OpenAI Assistants | Anthropic/OpenRouter | Tie |
| **Output Format** | **Formal (sections)** | Natural synthesis | Original ⭐ |
| **Individual Meetings** | **Yes** | No | Original ⭐ |
| **Provider Lock-in** | OpenAI only | **Multi-provider** | CoScientist ⭐ |
| **Experimental Validation** | **Published (Nature)** | Not yet | Original ⭐ |
| **Code Maturity** | **Production-ready** | New implementation | Original ⭐ |
| **Flexibility** | Lower | **Higher** | CoScientist ⭐ |
| **Computational Analysis** | Limited | **Extensive** | CoScientist ⭐ |

## Recommendations

### Use Original Virtual Lab if:
- You're replicating their nanobody design workflow
- You need formally structured outputs
- You prefer OpenAI ecosystem
- You want a proven, published methodology
- You need individual meetings for deep dives

### Use CoScientist Virtual Lab if:
- You want dynamic team composition
- You need database integration
- You want Python computational capabilities
- You prefer Claude or multi-provider flexibility
- You're doing exploratory research
- You want to integrate with existing CoScientist tools

### Use Both if:
- Large research project with multiple phases
- Exploration (CoScientist) → Implementation (Original VL)
- Want best of both tools and structure

## Future Enhancements

To close the gaps, CoScientist VL could add:

1. **Individual meetings** (like Original VL)
2. **Structured output format** option (agenda/questions/rules)
3. **Token tracking and cost estimation**
4. **Meeting persistence** (save/load meetings)
5. **Markdown export** with formatted sections

Original VL could benefit from:

1. **Dynamic team creation** (from CoScientist)
2. **Broader tool ecosystem** (databases, Python)
3. **Multi-provider support**
4. **Flexible meeting structures**

---

**Conclusion:** Both implementations are valid and excel in different areas. Original Virtual Lab is more mature and proven for structured computational workflows. CoScientist Virtual Lab is more flexible and feature-rich for exploratory biomedical research. Choose based on your specific use case!
