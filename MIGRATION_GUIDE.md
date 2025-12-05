# Migration Guide: From Single Agent to Virtual Lab

This guide helps you transition from the single-agent architecture to the Virtual Lab multi-agent system.

## Quick Start

### Before (Single Agent)
```bash
python -m src.cli --question "How do mRNA vaccines work?"
```

### After (Virtual Lab)
```bash
python -m src.cli --question "How do mRNA vaccines work?" --virtual-lab
```

That's it! The `--virtual-lab` flag enables multi-agent collaboration.

## Backward Compatibility

**Important**: All existing code continues to work! The original `BioinformaticsAgent` is unchanged.

```python
# This still works exactly as before
from src.agent.agent import create_agent

agent = create_agent()
answer = agent.run("What is CRISPR?")
```

## Upgrading Your Code

### Option 1: Use the Convenience Function (Recommended)

```python
from src.agent.meeting import run_virtual_lab

# Simplest upgrade path
answer = run_virtual_lab(
    question="Design a PD-L1 binder for TNBC",
    num_rounds=2,
    max_team_size=3,
    verbose=True
)
```

### Option 2: Use the VirtualLabMeeting Class

```python
from src.agent.meeting import VirtualLabMeeting

# More control over the meeting
meeting = VirtualLabMeeting(
    user_question="Your research question here",
    model="claude-sonnet-4-20250514",
    provider="anthropic",
    max_team_size=3,
    verbose=True
)

final_answer = meeting.run_meeting(num_rounds=2)

# Access the full transcript
for entry in meeting.get_transcript():
    print(f"{entry['speaker']}: {entry['content']}")
```

### Option 3: Create Custom Agent Teams

```python
from src.agent.agent import ScientificAgent, AgentPersona

# Define custom specialists
immunologist = ScientificAgent(
    persona=AgentPersona(
        title="Immunologist",
        expertise="T-cell biology, checkpoint inhibitors, CAR-T therapy",
        goal="Analyze immune mechanisms and therapeutic strategies",
        role="Provide immunology expertise"
    )
)

# Use like a regular agent
response = immunologist.run("Explain PD-1/PD-L1 axis in detail")
```

## When Should You Migrate?

### Stick with Single Agent if:
- ✓ Simple Q&A or literature lookup
- ✓ Quick iterations needed
- ✓ Cost is a primary concern
- ✓ Questions don't require multiple domains

### Migrate to Virtual Lab if:
- ✓ Complex multi-step problems (e.g., binder design)
- ✓ Need both computational + biological expertise
- ✓ High-stakes decisions requiring validation
- ✓ Want to reduce hallucination risk
- ✓ Transparency and attribution matter

## Feature Comparison

| Feature | Single Agent | Single + Critic | Virtual Lab |
|---------|-------------|----------------|-------------|
| API Calls | ~3-10 | ~15-25 | ~20-40 |
| Time | Fast (30s-2min) | Medium (1-3min) | Slower (2-5min) |
| Specialization | None | None | High |
| Error Detection | Self-validation | 1 critic review | Continuous |
| Tool Usage | Yes | Yes | Yes (per specialist) |
| Transparency | Medium | High | Very High |
| Cost | $ | $$ | $$$ |

## Code Examples

### Example 1: Database Analysis

**Before**:
```python
agent = create_agent()
answer = agent.run("Find EGFR inhibitors in BindingDB with IC50 < 100nM")
```

**After** (Virtual Lab for complex analysis):
```python
answer = run_virtual_lab(
    question="Find EGFR inhibitors in BindingDB with IC50 < 100nM and analyze their structure-activity relationships",
    num_rounds=2,
    max_team_size=3  # Might create: Bioinformatician, Medicinal Chemist, Data Scientist
)
```

### Example 2: Literature Review

**Before**:
```python
agent = create_agent()
answer = agent.run("What are the latest advances in CAR-T therapy?")
```

**After** (keep single agent - simple lookup):
```python
# Virtual Lab is overkill for simple literature review
agent = create_agent()
answer = agent.run("What are the latest advances in CAR-T therapy?")
```

### Example 3: Molecular Design

**Before** (risky - might hallucinate):
```python
agent = create_agent()
answer = agent.run("Design a peptide that binds to PD-L1")
# Risk: Single agent might generate plausible-sounding but invalid sequences
```

**After** (Virtual Lab - validation built-in):
```python
answer = run_virtual_lab(
    question="Design a peptide that binds to PD-L1 for cancer immunotherapy",
    num_rounds=3,  # More rounds for complex design
    max_team_size=4,  # Comp Bio, Immunologist, Structural Biologist, Critic
    verbose=True
)
# Computational Biologist generates sequences
# Immunologist validates PD-L1 biology
# Structural Biologist checks binding plausibility
# Critic identifies gaps
# PI synthesizes validated design
```

## CLI Comparison

### All CLI Modes

```bash
# 1. Single agent (original)
python -m src.cli --question "..."

# 2. Single agent with critic feedback
python -m src.cli --question "..." --with-critic

# 3. Virtual Lab (multi-agent)
python -m src.cli --question "..." --virtual-lab

# 4. Virtual Lab with customization
python -m src.cli --question "..." --virtual-lab --rounds 3 --team-size 4 --verbose

# 5. Interactive Virtual Lab
python -m src.cli --interactive --virtual-lab --rounds 2
```

## Configuration Recommendations

### For Quick Exploration
```bash
--virtual-lab --rounds 1 --team-size 2
```

### For Standard Research Questions
```bash
--virtual-lab --rounds 2 --team-size 3
```

### For Complex Multi-Step Problems
```bash
--virtual-lab --rounds 3 --team-size 4 --verbose
```

### For Debugging
```bash
--virtual-lab --rounds 1 --team-size 2 --verbose
```

## Cost Estimation

Approximate token usage (Claude Sonnet 4):

| Mode | Input Tokens | Output Tokens | Cost (approx) |
|------|--------------|---------------|---------------|
| Single Agent | 5K-20K | 2K-5K | $0.10-$0.40 |
| Single + Critic | 15K-40K | 5K-10K | $0.30-$0.80 |
| Virtual Lab (2 rounds, 3 agents) | 30K-80K | 10K-20K | $0.60-$1.50 |
| Virtual Lab (3 rounds, 4 agents) | 50K-120K | 15K-30K | $1.00-$2.50 |

*Estimates based on Claude Sonnet 4 pricing ($3/MTok input, $15/MTok output). Actual costs vary by question complexity.*

## Testing Your Migration

Use the provided test script:

```bash
python test_virtual_lab.py
```

This will:
1. Test dynamic team creation for different question types
2. Run a full Virtual Lab meeting
3. Validate that all components work together

## Troubleshooting

### "Module not found" errors
Ensure you're running from the project root:
```bash
cd /data/galaxy4/user/j2ho/projects/coscientist
python -m src.cli --virtual-lab ...
```

### Team creation fails
The system will automatically fall back to a default general-purpose team. Check that:
- Your API key is set correctly
- The model supports JSON output
- Network connection is stable

### Agents repeat themselves
- Reduce `num_rounds` (3 → 2)
- Ensure personas are distinct
- Check context window isn't truncated

### Too expensive
- Start with `--rounds 1 --team-size 2`
- Use single agent with critic (`--with-critic`) instead
- Reserve Virtual Lab for complex problems

## Next Steps

1. **Try it**: Run your most complex question with `--virtual-lab`
2. **Compare**: Run the same question with and without Virtual Lab
3. **Tune**: Adjust `--rounds` and `--team-size` based on results
4. **Customize**: Create custom personas for your specific domain
5. **Contribute**: Share your findings and improvements!

## Support

- See [VIRTUAL_LAB.md](VIRTUAL_LAB.md) for architecture details
- See [README.md](README.md) for general CoScientist documentation
- File issues on GitHub for bugs or feature requests

---

Happy researching with Virtual Lab! 🔬
