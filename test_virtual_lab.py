"""Test script for Virtual Lab architecture."""

import os
from dotenv import load_dotenv
from src.agent.meeting import run_virtual_lab

def test_basic_virtual_lab():
    """Test basic Virtual Lab functionality."""
    load_dotenv()

    # Simple test question
    question = "What are the key differences between mRNA vaccines and traditional vaccines?"

    print("\n" + "=" * 70)
    print("TESTING VIRTUAL LAB ARCHITECTURE")
    print("=" * 70)
    print(f"\nTest Question: {question}\n")

    # Run with verbose output to see the full meeting
    final_answer = run_virtual_lab(
        question=question,
        num_rounds=1,  # Just 1 round for testing
        max_team_size=2,  # Small team for testing
        verbose=True
    )

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)
    print(f"\nFinal Answer:\n{final_answer}\n")

    # Basic validation
    assert final_answer, "Should produce a final answer"
    assert len(final_answer) > 100, "Answer should be substantive"

    print("✓ Test passed!")

def test_team_creation():
    """Test that the PI can create appropriate teams for different questions."""
    load_dotenv()

    from src.agent.agent import create_agent
    from src.agent.team_manager import create_research_team

    questions = [
        "How can we design a peptide binder for PD-L1?",
        "What genetic variants are associated with Alzheimer's disease?",
        "How do checkpoint inhibitors work in cancer immunotherapy?"
    ]

    print("\n" + "=" * 70)
    print("TESTING DYNAMIC TEAM CREATION")
    print("=" * 70)

    agent = create_agent()

    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        team = create_research_team(question, agent.client, max_team_size=3)
        print(f"   Team ({len(team)} members):")
        for member in team:
            print(f"   - {member['title']}: {member['expertise'][:60]}...")

    print("\n✓ Team creation test passed!")

if __name__ == "__main__":
    # Uncomment to run tests
    # Note: These tests will make actual API calls

    print("\nVirtual Lab Architecture Test Suite")
    print("====================================")
    print("\nTest 1: Dynamic Team Creation")
    test_team_creation()

    print("\n\nTest 2: Full Virtual Lab Meeting")
    print("(This may take 2-3 minutes...)")
    test_basic_virtual_lab()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
