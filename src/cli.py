"""Command-line interface for the bioinformatics agent."""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
from src.agent.agent import create_agent
from src.agent.meeting import run_virtual_lab


def main():
    """Main CLI entry point."""
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="CoScientist: AI Research Assistant for Biomedical Questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask a single question
  python -m src.cli --question "How can gene signatures guide drug repositioning?"

  # Virtual Lab mode (multi-agent collaboration)
  python -m src.cli --question "..." --virtual-lab

  # Virtual Lab with more rounds and specialists
  python -m src.cli --question "..." --virtual-lab --rounds 3 --team-size 4

  # With critic feedback loop for quality validation (single agent)
  python -m src.cli --question "..." --with-critic

  # Interactive mode
  python -m src.cli --interactive

  # Use a different model
  python -m src.cli --question "..." --model "openai/gpt-4"

  # Verbose output to see tool calls
  python -m src.cli --question "..." --verbose
        """,
    )

    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Question to ask the agent",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="anthropic/claude-sonnet-4",
        help="Model to use (default: claude-sonnet-4)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output with tool calls",
    )
    parser.add_argument(
        "--with-critic",
        "-c",
        action="store_true",
        help="Enable critic feedback loop for quality validation (single agent mode)",
    )
    parser.add_argument(
        "--virtual-lab",
        "-vl",
        action="store_true",
        help="Enable Virtual Lab mode (multi-agent collaboration)",
    )
    parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=2,
        help="Number of discussion rounds in Virtual Lab mode (default: 2)",
    )
    parser.add_argument(
        "--team-size",
        "-t",
        type=int,
        default=3,
        help="Maximum number of specialist agents in Virtual Lab mode (default: 3)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (Anthropic or OpenRouter, or set env var)",
    )

    args = parser.parse_args()

    # Check for conflicting modes
    if args.virtual_lab and args.with_critic:
        print("Error: Cannot use both --virtual-lab and --with-critic at the same time.", file=sys.stderr)
        print("Choose one mode: Virtual Lab (multi-agent) OR single agent with critic.", file=sys.stderr)
        sys.exit(1)

    # Determine provider from model name or environment
    provider = None
    if args.model and "/" in args.model:
        # Model format like "anthropic/..." or "openai/..." indicates OpenRouter
        provider = "openrouter"
    else:
        provider = "anthropic"

    # Only create agent for non-virtual-lab modes
    if not args.virtual_lab:
        try:
            agent = create_agent(api_key=args.api_key, model=args.model, provider=provider)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    if args.question:
        # Single question mode
        if args.virtual_lab:
            # Virtual Lab mode - multi-agent collaboration
            print("\n" + "=" * 60)
            print("VIRTUAL LAB MODE")
            print("=" * 60)
            print(f"Question: {args.question}")
            print(f"Configuration: {args.rounds} rounds, max {args.team_size} specialists")
            print("=" * 60)

            final_answer = run_virtual_lab(
                question=args.question,
                api_key=args.api_key,
                model=args.model,
                provider=provider,
                num_rounds=args.rounds,
                max_team_size=args.team_size,
                verbose=args.verbose
            )

            print("\n" + "=" * 60)
            print("FINAL ANSWER (PI Synthesis):")
            print("=" * 60)
            print(final_answer)

        elif args.with_critic:
            initial, critique, final = agent.run_with_critic(args.question, verbose=args.verbose)
            print("\n" + "=" * 60)
            print("INITIAL ANSWER:")
            print("=" * 60)
            print(initial)
            print("\n" + "=" * 60)
            print("CRITIC FEEDBACK:")
            print("=" * 60)
            print(critique)
            print("\n" + "=" * 60)
            print("FINAL REFINED ANSWER:")
            print("=" * 60)
            print(final)
        else:
            response = agent.run(args.question, verbose=args.verbose)
            print("\n" + "=" * 60)
            print("Final Answer:")
            print("=" * 60)
            print(response)

    elif args.interactive:
        # Interactive mode
        print("=" * 60)
        print("CoScientist: Interactive Mode")
        if args.virtual_lab:
            print("(Virtual Lab - Multi-Agent Collaboration)")
        print("=" * 60)
        print("Ask biomedical research questions. Type 'exit' or 'quit' to exit.\n")

        while True:
            try:
                question = input("Question: ").strip()
            except EOFError:
                break

            if question.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not question:
                continue

            if args.virtual_lab:
                # Virtual Lab mode in interactive
                print("\n" + "=" * 60)
                print("VIRTUAL LAB MEETING")
                print("=" * 60)

                final_answer = run_virtual_lab(
                    question=question,
                    api_key=args.api_key,
                    model=args.model,
                    provider=provider,
                    num_rounds=args.rounds,
                    max_team_size=args.team_size,
                    verbose=args.verbose
                )

                print("\n" + "=" * 60)
                print("FINAL ANSWER:")
                print("=" * 60)
                print(final_answer)
                print()

            elif args.with_critic:
                initial, critique, final = agent.run_with_critic(question, verbose=args.verbose)
                print("\n" + "=" * 60)
                print("INITIAL ANSWER:")
                print("=" * 60)
                print(initial)
                print("\n" + "=" * 60)
                print("CRITIC FEEDBACK:")
                print("=" * 60)
                print(critique)
                print("\n" + "=" * 60)
                print("FINAL REFINED ANSWER:")
                print("=" * 60)
                print(final)
                print()
            else:
                response = agent.run(question, verbose=args.verbose)
                print("\n" + "=" * 60)
                print("Answer:")
                print("=" * 60)
                print(response)
                print()

    else:
        # No input provided
        parser.print_help()


if __name__ == "__main__":
    main()
