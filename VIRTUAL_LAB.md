# Virtual Lab Architecture

This document describes the Virtual Lab multi-agent architecture implemented in CoScientist.

## Overview

The Virtual Lab architecture transforms CoScientist from a **single-turn generalist** into a **multi-agent collaborative system** based on the methodology described in the Virtual Lab paper. Instead of one agent trying to solve everything, the system creates a team of specialized agents that collaborate in a structured meeting format.

## Key Components

### 1. **AgentPersona** (Dataclass)
Defines a scientific agent's role:
- `title`: Role name (e.g., "Immunologist", "Computational Biologist")
- `expertise`: Domain knowledge and skills
- `goal`: What this agent should accomplish
- `role`: Specific responsibilities in the meeting

### 2. **ScientificAgent** (Class)
Extends `BioinformaticsAgent` with dynamic role-playing:
- Accepts an `AgentPersona` to define its behavior
- Generates persona-specific system prompts
- Maintains all tool-using capabilities from the base agent

### 3. **Team Manager** (Module: `team_manager.py`)
Dynamic team composition:
- **PI Agent**: Analyzes the research question and designs the team
- Selects 2-4 specialists based on the question's requirements
- Creates personas for PI and Scientific Critic roles

### 4. **VirtualLabMeeting** (Class)
Orchestrates the meeting structure:
1. **Initialization**: PI designs the research team
2. **Opening**: PI frames the question and sets agenda
3. **Round-Robin Discussions**: Specialists contribute sequentially
4. **Critic Review**: Identifies flaws and gaps after each round
5. **PI Synthesis**: Integrates findings into final answer

## Architecture Comparison

### Before (Single-Turn Generalist)
```
User Question → BioinformaticsAgent → Answer
                (tries to do everything)
```

### After (Virtual Lab)
```
User Question → PI (designs team)
              ↓
        [Specialist 1, Specialist 2, Specialist 3]
              ↓
      Round-Robin Discussion (2-3 rounds)
              ↓
         Critic Review
              ↓
        PI Final Synthesis → Answer
```

## Usage

### Command Line

#### Basic Virtual Lab Mode
```bash
python -m src.cli --question "How can we design a PD-L1 binder?" --virtual-lab
```

#### Customized Configuration
```bash
# More discussion rounds and larger team
python -m src.cli \
  --question "Design a therapeutic peptide for triple-negative breast cancer" \
  --virtual-lab \
  --rounds 3 \
  --team-size 4 \
  --verbose
```

#### Interactive Virtual Lab Mode
```bash
python -m src.cli --interactive --virtual-lab --rounds 2 --team-size 3
```

### Python API

```python
from src.agent.meeting import run_virtual_lab

# Simple usage
answer = run_virtual_lab(
    question="What are the mechanisms of checkpoint inhibition?",
    num_rounds=2,
    max_team_size=3,
    verbose=True
)

# Advanced: Access full meeting transcript
from src.agent.meeting import VirtualLabMeeting

meeting = VirtualLabMeeting(
    user_question="How do mRNA vaccines work?",
    max_team_size=3,
    verbose=True
)

final_answer = meeting.run_meeting(num_rounds=2)

# Get full transcript
transcript = meeting.get_transcript()
for entry in transcript:
    print(f"{entry['speaker']}: {entry['content'][:100]}...")
```

### Creating Custom Personas

```python
from src.agent.agent import ScientificAgent, AgentPersona

# Define a custom specialist
oncologist_persona = AgentPersona(
    title="Oncologist",
    expertise="Cancer biology, tumor microenvironment, therapeutic resistance",
    goal="Analyze cancer mechanisms and therapeutic strategies",
    role="Provide oncology expertise and validate therapeutic relevance"
)

# Create the agent
oncologist = ScientificAgent(
    persona=oncologist_persona,
    model="claude-sonnet-4-20250514",
    provider="anthropic"
)

# Use it
response = oncologist.run("Analyze the role of PD-L1 in immune evasion")
```

## How It Works

### Phase 1: Team Design (Dynamic)
The PI agent receives the research question and decides which specialists are needed:

**Example Team for "Design a PD-L1 binder for TNBC":**
- Computational Biologist (sequence design, structure prediction)
- Immunologist (PD-L1 biology, checkpoint mechanisms)
- Data Scientist (database mining, validation)

### Phase 2: Discussion Rounds
Each round follows this pattern:

1. **Specialist Contributions**: Each agent applies their expertise
   - Can use tools: `execute_python`, `search_pubmed`, `query_database`
   - Builds on previous contributions
   - Focuses on their domain

2. **Critic Review**: Identifies issues
   - Points out logical flaws
   - Notes unsupported claims
   - Suggests missing analyses
   - Does NOT propose solutions

3. **PI Synthesis** (interim): Summarizes progress between rounds

### Phase 3: Final Synthesis
The PI integrates all findings into a comprehensive answer:
- Addresses the original question directly
- Synthesizes insights from all specialists
- Acknowledges limitations
- Proposes next steps

## Benefits Over Single Agent

### 1. **Specialization**
- Each agent focuses on their domain expertise
- Reduces hallucination by limiting scope
- More rigorous domain-specific reasoning

### 2. **Error Detection**
- Critic agent catches mistakes early
- Cross-validation between specialists
- Multiple perspectives reduce blind spots

### 3. **Complex Problem Solving**
- Multi-step problems broken down naturally
- Different aspects addressed by appropriate experts
- Better integration of computational + biological reasoning

### 4. **Transparency**
- Clear attribution of contributions
- Visible reasoning chain
- Easier to debug and improve

## Example: TNBC Binder Design

**Question**: "Design a peptide binder for PD-L1 to treat triple-negative breast cancer"

**Traditional Approach** (Single Agent):
- Agent tries to do sequence design + immunology + validation all at once
- Risk of hallucinating sequences without proper validation
- May miss key biological constraints

**Virtual Lab Approach**:
1. **Computational Biologist**: Generates candidate sequences using Python tools
2. **Immunologist**: Validates PD-L1 biology and TNBC context from literature
3. **Data Scientist**: Mines BindingDB for precedent compounds and validates predictions
4. **Critic**: Checks for biophysical plausibility, missing validations
5. **PI**: Synthesizes into final binder design with rationale

## Configuration Parameters

### `num_rounds` (default: 2)
- Number of discussion rounds
- More rounds = deeper analysis but higher cost
- Recommended: 2 for most questions, 3 for complex multi-step problems

### `max_team_size` (default: 3)
- Maximum number of specialist agents (excluding PI and Critic)
- More specialists = broader coverage but longer meetings
- Recommended: 2-3 for focused questions, 4 for broad investigations

### `verbose` (default: False)
- Print full meeting transcript with tool calls
- Useful for debugging and understanding reasoning
- Recommended: True for development, False for production

## Comparison to Original Agent

### When to Use Single Agent (`BioinformaticsAgent`)
- Simple, straightforward questions
- Literature review only
- Quick database queries
- Rapid iteration needed

### When to Use Virtual Lab (`VirtualLabMeeting`)
- Complex multi-step problems
- Need computational + biological expertise
- Designing novel molecules or strategies
- High-stakes decisions requiring validation
- Research questions requiring diverse expertise

### When to Use Critic Mode (`--with-critic`)
- Single-agent workflow but want quality validation
- Intermediate complexity
- Budget-conscious (fewer API calls than Virtual Lab)

## File Structure

```
src/agent/
├── agent.py              # Core agents: BioinformaticsAgent, ScientificAgent, AgentPersona
├── team_manager.py       # Dynamic team creation logic
├── meeting.py            # VirtualLabMeeting orchestration
├── anthropic_client.py   # Anthropic API wrapper
└── openrouter_client.py  # OpenRouter API wrapper
```

## Future Enhancements

Potential improvements to the Virtual Lab architecture:

1. **Persistent Experts**: Cache common specialist personas
2. **Adaptive Rounds**: Automatically determine optimal number of rounds
3. **Parallel Tool Calls**: Specialists execute tools simultaneously
4. **Memory System**: Agents remember findings across multiple questions
5. **Hierarchical Teams**: Sub-teams for complex multi-faceted problems
6. **Performance Metrics**: Track specialist contribution quality

## Troubleshooting

### Team Creation Fails
If the PI fails to design a team, the system falls back to a default general-purpose team:
- Bioinformatician
- Biomedical Scientist
- Data Scientist

### Excessive API Costs
Reduce costs by:
- Decreasing `num_rounds` (2 → 1)
- Decreasing `max_team_size` (3 → 2)
- Using the single agent with critic instead
- Setting `verbose=False` to reduce token usage

### Repetitive Contributions
If agents repeat themselves:
- Check that context window isn't being truncated
- Reduce `num_rounds` to avoid diminishing returns
- Ensure specialists have distinct personas

## References

- **Virtual Lab Paper**: "Virtual Lab: AI Agents as Research Assistants"
- **Original CoScientist**: Single-agent bioinformatics assistant
- **Claude API Documentation**: https://docs.anthropic.com/

---

**Questions or Issues?**
See the main README for contribution guidelines and issue reporting.
