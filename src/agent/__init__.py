"""Agent module for CoScientist."""

from src.agent.agent import BioinformaticsAgent, ScientificAgent, AgentPersona, create_agent
from src.agent.meeting import VirtualLabMeeting, run_virtual_lab
from src.agent.team_manager import (
    create_research_team,
    create_pi_persona,
    create_critic_persona,
)

__all__ = [
    "BioinformaticsAgent",
    "ScientificAgent",
    "AgentPersona",
    "create_agent",
    "VirtualLabMeeting",
    "run_virtual_lab",
    "create_research_team",
    "create_pi_persona",
    "create_critic_persona",
]
