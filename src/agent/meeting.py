"""Virtual Lab meeting structure for multi-agent collaboration."""

import os
from typing import Optional
from src.agent.agent import ScientificAgent, AgentPersona
from src.agent.team_manager import (
    create_research_team,
    create_pi_persona,
    create_critic_persona,
)


class VirtualLabMeeting:
    """Manages a Virtual Lab research meeting with multiple specialist agents.

    Implements the Virtual Lab methodology where:
    1. A PI agent dynamically designs the research team based on the question
    2. Specialist agents contribute in round-robin discussions
    3. A Critic agent reviews findings and identifies flaws
    4. The PI synthesizes conclusions

    This mirrors the hierarchical meeting structure from the Virtual Lab paper,
    replacing a single-turn generalist with a multi-agent collaborative system.
    """

    def __init__(
        self,
        user_question: str,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        max_team_size: int = 3,
        verbose: bool = False
    ):
        """Initialize a Virtual Lab meeting.

        Args:
            user_question: The research question to solve
            api_key: API key for LLM provider
            model: Model to use for all agents
            provider: 'anthropic' or 'openrouter'
            max_team_size: Maximum number of specialist agents (default: 3)
            verbose: Print detailed meeting transcript
        """
        self.user_question = user_question
        self.verbose = verbose
        self.api_key = api_key
        self.model = model
        self.provider = provider

        # Initialize the PI first
        if self.verbose:
            print("\n" + "=" * 60)
            print("INITIALIZING VIRTUAL LAB MEETING")
            print("=" * 60)

        self.pi = ScientificAgent(
            persona=create_pi_persona(),
            api_key=api_key,
            model=model,
            provider=provider
        )

        # PI designs the research team
        if self.verbose:
            print("\n[PI is designing the research team...]")

        team_specs = create_research_team(
            user_question,
            self.pi.client,
            max_team_size=max_team_size
        )

        if self.verbose:
            print(f"\n[Team designed: {len(team_specs)} specialists]")
            for i, spec in enumerate(team_specs, 1):
                print(f"  {i}. {spec['title']}")

        # Instantiate specialist agents
        self.specialists = [
            ScientificAgent(
                persona=AgentPersona(**spec),
                api_key=api_key,
                model=model,
                provider=provider
            )
            for spec in team_specs
        ]

        # Add the Scientific Critic (static role)
        self.critic = ScientificAgent(
            persona=create_critic_persona(),
            api_key=api_key,
            model=model,
            provider=provider
        )

        self.meeting_transcript = []

    def run_meeting(self, num_rounds: int = 2) -> str:
        """Run the Virtual Lab meeting.

        Args:
            num_rounds: Number of discussion rounds (default: 2)

        Returns:
            Final synthesized answer from the PI
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("STARTING MEETING")
            print("=" * 60)

        # Phase 1: PI opens the meeting and sets the agenda
        if self.verbose:
            print("\n[PHASE 1: PI Opening Remarks]")

        pi_intro_prompt = f"""Open the research meeting and set the agenda.

Research Question: "{self.user_question}"

Your team consists of:
{self._format_team_list()}

Provide a brief opening (2-3 sentences) that:
1. Frames the research question
2. Identifies key challenges or sub-problems
3. Sets expectations for the team

Keep it concise - this is just the opening."""

        pi_intro = self.pi.run(pi_intro_prompt, verbose=self.verbose)
        self.meeting_transcript.append({
            "speaker": "PI",
            "role": "Opening Remarks",
            "content": pi_intro
        })

        # Phase 2: Round-robin specialist discussions
        for round_num in range(num_rounds):
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"[PHASE 2: DISCUSSION ROUND {round_num + 1}/{num_rounds}]")
                print(f"{'=' * 60}")

            # Each specialist contributes
            for agent in self.specialists:
                if self.verbose:
                    print(f"\n--- {agent.persona.title} speaking ---")

                # Build context from recent transcript
                context = self._build_context(last_n=5)

                specialist_prompt = f"""Research Question: "{self.user_question}"

Meeting Context (recent discussion):
{context}

Contribute your specialized analysis. You may:
- Use tools (execute_python, search_pubmed, query_database) as needed
- Build on others' findings
- Propose specific analyses or experiments
- Point out issues you see

Be concise (3-5 sentences or a specific analysis). Focus on YOUR expertise."""

                response = agent.run(specialist_prompt, verbose=self.verbose)
                self.meeting_transcript.append({
                    "speaker": agent.persona.title,
                    "role": agent.persona.role,
                    "content": response
                })

            # Phase 3: Critic reviews the round
            if self.verbose:
                print(f"\n--- Scientific Critic Review ---")

            recent_discussion = self._build_context(last_n=len(self.specialists))
            critique_prompt = f"""Review the most recent contributions from the team.

Recent Discussion:
{recent_discussion}

Your task:
- Identify errors, logical flaws, or unsupported claims
- Point out missing analyses or gaps
- Highlight strong points worth pursuing
- DO NOT provide solutions - only critique

Be specific and constructive (2-4 sentences)."""

            critique = self.critic.run(critique_prompt, verbose=False)  # Critic doesn't need verbose
            self.meeting_transcript.append({
                "speaker": "Critic",
                "role": "Quality Review",
                "content": critique
            })

            if self.verbose:
                print(f"Critic: {critique[:200]}..." if len(critique) > 200 else f"Critic: {critique}")

            # PI synthesizes the round (except on last round)
            if round_num < num_rounds - 1:
                if self.verbose:
                    print(f"\n--- PI Round Synthesis ---")

                round_context = self._build_context(last_n=len(self.specialists) + 1)
                synthesis_prompt = f"""Synthesize the current round of discussion.

Round {round_num + 1} Discussion:
{round_context}

Provide a brief synthesis (2-3 sentences):
- What progress was made?
- What should the team focus on in the next round?

Be concise - this is an interim summary."""

                round_summary = self.pi.run(synthesis_prompt, verbose=False)
                self.meeting_transcript.append({
                    "speaker": "PI",
                    "role": f"Round {round_num + 1} Synthesis",
                    "content": round_summary
                })

                if self.verbose:
                    print(f"PI Summary: {round_summary}")

        # Phase 4: PI final synthesis
        if self.verbose:
            print("\n" + "=" * 60)
            print("[PHASE 3: FINAL SYNTHESIS]")
            print("=" * 60)

        final_prompt = f"""Synthesize the team's findings into a final answer.

Original Question: "{self.user_question}"

Full Meeting Transcript:
{self._build_full_transcript()}

Provide a comprehensive final answer that:
1. Directly answers the research question
2. Integrates insights from all specialists
3. Acknowledges limitations and uncertainties
4. Proposes next steps if appropriate

Structure your answer clearly with sections if needed."""

        final_answer = self.pi.run(final_prompt, verbose=self.verbose)
        self.meeting_transcript.append({
            "speaker": "PI",
            "role": "Final Answer",
            "content": final_answer
        })

        return final_answer

    def _format_team_list(self) -> str:
        """Format the team member list for prompts."""
        team_list = []
        for agent in self.specialists:
            team_list.append(f"- {agent.persona.title}: {agent.persona.expertise}")
        return "\n".join(team_list)

    def _build_context(self, last_n: int = 5) -> str:
        """Build context string from last N transcript entries."""
        recent = self.meeting_transcript[-last_n:] if len(self.meeting_transcript) >= last_n else self.meeting_transcript
        context_parts = []
        for entry in recent:
            context_parts.append(f"[{entry['speaker']}]: {entry['content']}")
        return "\n\n".join(context_parts)

    def _build_full_transcript(self) -> str:
        """Build the full meeting transcript."""
        transcript_parts = []
        for entry in self.meeting_transcript:
            transcript_parts.append(f"=== {entry['speaker']} ({entry['role']}) ===\n{entry['content']}")
        return "\n\n".join(transcript_parts)

    def get_transcript(self) -> list[dict]:
        """Get the full meeting transcript.

        Returns:
            List of transcript entries with speaker, role, and content
        """
        return self.meeting_transcript


def run_virtual_lab(
    question: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    num_rounds: int = 2,
    max_team_size: int = 3,
    verbose: bool = False
) -> str:
    """Convenience function to run a Virtual Lab meeting.

    Args:
        question: Research question to solve
        api_key: API key (defaults to env var)
        model: Model to use (defaults based on provider)
        provider: 'anthropic' or 'openrouter' (defaults to env var)
        num_rounds: Number of discussion rounds (default: 2)
        max_team_size: Maximum number of specialists (default: 3)
        verbose: Print detailed transcript

    Returns:
        Final synthesized answer
    """
    # Set defaults
    if provider is None:
        provider = os.getenv("API_PROVIDER", "anthropic")

    if model is None:
        if provider == "anthropic":
            model = "claude-sonnet-4-20250514"
        else:
            model = "anthropic/claude-sonnet-4"

    # Create and run meeting
    meeting = VirtualLabMeeting(
        user_question=question,
        api_key=api_key,
        model=model,
        provider=provider,
        max_team_size=max_team_size,
        verbose=verbose
    )

    final_answer = meeting.run_meeting(num_rounds=num_rounds)
    return final_answer
