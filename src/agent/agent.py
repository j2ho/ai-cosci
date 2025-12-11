"""Core agent loop for the bioinformatics AI system."""

import json
import os
import asyncio
from typing import Any, Optional
from dataclasses import dataclass
from src.agent.openrouter_client import OpenRouterClient
from src.agent.anthropic_client import AnthropicClient
from src.tools.implementations import (
    execute_python,
    search_pubmed,
    search_literature,
    query_database,
    read_file,
    find_files,
    get_tool_definitions,
)


def get_max_tokens_for_model(model_name: str) -> int:
    """Determine appropriate max_tokens based on model name.

    Free models on OpenRouter have limited credits, so we use fewer tokens.

    Args:
        model_name: The model identifier (e.g., "openai/gpt-oss-120b:free")

    Returns:
        Maximum tokens to request (500 for free models, 4096 for paid)
    """
    if ":free" in model_name.lower():
        return 500  # Conservative limit for free tier
    return 4096  # Higher limit for paid models to avoid truncation


@dataclass
class AgentPersona:
    """Represents a scientific agent's persona for role-playing in Virtual Lab."""
    title: str
    expertise: str
    goal: str
    role: str


class BioinformaticsAgent:
    """Agent for answering complex bioinformatics questions."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514", provider: str = "anthropic", data_dir: str = "/home.galaxy4/sumin/project/aisci/Competition_Data", input_dir: Optional[str] = None):
        """Initialize the agent.

        Args:
            api_key: API key (Anthropic or OpenRouter)
            model: Model to use
            provider: 'anthropic' or 'openrouter'
            data_dir: Path to database directory (Drug databases, PPI, GWAS, etc.)
            input_dir: Path to question-specific input data (defaults to data_dir)
        """
        if provider == "anthropic":
            self.client = AnthropicClient(api_key=api_key, model=model)
        else:
            self.client = OpenRouterClient(api_key=api_key, model=model)
        self.data_dir = data_dir
        self.input_dir = input_dir if input_dir is not None else data_dir
        self.tools = {
            "execute_python": execute_python,
            "search_pubmed": search_pubmed,
            "search_literature": search_literature,
            "query_database": query_database,
            "read_file": read_file,
            "find_files": find_files,
        }
        self.conversation_history = []
        self.max_iterations = 30  # Increased from 10 to allow complex multi-step analyses

    def get_system_prompt(self) -> str:
        """Get the system prompt for scientific reasoning.

        Returns:
            System prompt string
        """
        return """You are CoScientist, an expert AI research assistant for biomedical research. Your role is to help scientists answer complex, multi-step research questions through rigorous analysis, data exploration, and literature review.

## Core Capabilities
- Analyze complex biological datasets and databases
- Search peer-reviewed literature to ground answers in evidence
- Execute Python code for data analysis and visualization
- Reason through multi-step research problems systematically
- Propose novel experimental strategies and validate their feasibility

## Response Guidelines

1. **Scientific Rigor**: Always distinguish between established facts, well-supported hypotheses, and speculative ideas. Use phrases like "evidence suggests", "it is plausible that", "needs validation", etc.

2. **Multi-step Problem Solving**: For complex questions:
   - Break the problem into logical steps
   - Explain your reasoning for each step
   - Use tools strategically to gather data
   - Synthesize findings into coherent conclusions

3. **Data Exploration**: When analyzing data:
   - Query relevant databases to get context
   - Examine distributions, patterns, and outliers
   - Document assumptions clearly
   - Consider alternative explanations

4. **Literature Integration**:
   - Use `search_literature` for deep, evidence-based answers from full-text papers (recommended for complex mechanistic questions)
   - Use `search_pubmed` for quick abstract-level searches
   - ALWAYS cite papers in this format: "Title" (PMID: 12345678)
   - Include the full paper title and PMID for every claim backed by literature
   - Example: "According to 'CRISPR-Cas9 genome editing in human cells' (PMID: 23287718), ..."
   - Note consensus vs. contrasting findings
   - Acknowledge knowledge gaps where relevant

5. **Practical Feasibility**:
   - When proposing strategies, consider experimental resources needed
   - Discuss timeline expectations
   - Identify potential technical challenges
   - Suggest validation approaches

## Communication Style
- Be precise and avoid vague statements
- Use appropriate scientific terminology
- Structure complex answers with clear sections
- Highlight key findings and limitations
- Propose next steps when appropriate

Remember: Your goal is to help scientists make informed decisions, not to provide definitive answers. Surface uncertainty honestly and help them understand where more research is needed."""

    def add_message(self, role: str, content: str):
        """Add a message to conversation history.

        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        self.conversation_history.append({"role": role, "content": content})

    def call_tool(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to call
            tool_input: Input arguments for the tool

        Returns:
            Tool result as dict
        """
        if tool_name not in self.tools:
            return {
                "success": False,
                "output": None,
                "error": f"Unknown tool: {tool_name}",
            }

        try:
            tool_func = self.tools[tool_name]
            # Add data_dir to query_database calls
            if tool_name == "query_database":
                tool_input["data_dir"] = self.data_dir
            # Add input_dir to read_file calls
            elif tool_name == "read_file":
                tool_input["input_dir"] = self.input_dir
            # Add data_dir to find_files calls
            elif tool_name == "find_files":
                tool_input["data_dir"] = self.data_dir
            result = tool_func(**tool_input)
            return result.to_dict()
        except TypeError as e:
            return {
                "success": False,
                "output": None,
                "error": f"Invalid arguments for {tool_name}: {str(e)}",
            }
        except Exception as e:
            return {
                "success": False,
                "output": None,
                "error": f"Tool execution error: {str(e)}",
            }

    def process_response(self, response: dict[str, Any]) -> tuple[Optional[str], list[dict[str, Any]]]:
        """Process API response and extract text and tool calls.

        Args:
            response: Response from OpenRouter API

        Returns:
            Tuple of (response_text, tool_calls)
        """
        text = self.client.get_response_text(response)
        tool_calls = self.client.extract_tool_calls(response)
        return text, tool_calls

    def run(self, user_question: str, verbose: bool = False) -> str:
        """Run the agent loop for a user question.

        Args:
            user_question: The question to answer
            verbose: Print intermediate steps

        Returns:
            Final response from the agent
        """
        self.conversation_history = []
        self.add_message("user", user_question)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Question: {user_question}")
            print(f"{'='*60}\n")

        for iteration in range(self.max_iterations):
            if verbose:
                print(f"[Iteration {iteration + 1}/{self.max_iterations}]")

            # Get response from LLM
            # Anthropic API requires system prompt separately
            call_params = {
                "messages": self.conversation_history,
                "tools": get_tool_definitions(),
                "temperature": 0.7,
                "max_tokens": get_max_tokens_for_model(self.client.model),
            }

            # Add system prompt for Anthropic
            if isinstance(self.client, AnthropicClient):
                call_params["system"] = self.get_system_prompt()

            response = self.client.create_message(**call_params)

            text, tool_calls = self.process_response(response)

            # Check if response was truncated (hit token limit)
            finish_reason = None
            if response.get("choices") and len(response["choices"]) > 0:
                finish_reason = response["choices"][0].get("finish_reason")

            if text:
                if verbose:
                    print(f"Assistant: {text[:200]}..." if len(text) > 200 else f"Assistant: {text}")
                    if finish_reason:
                        print(f"[Finish reason: {finish_reason}]")

            # If no tool calls, we're done
            if not tool_calls:
                # Warn if response was truncated due to length
                if finish_reason == "length":
                    if verbose:
                        print("\n[WARNING: Response was truncated due to max_tokens limit]")
                        print("[Agent completed - no more tools needed]")
                elif verbose:
                    print("\n[Agent completed - no more tools needed]")
                # Add the final assistant message
                if text:
                    self.add_message("assistant", text)
                return text

            if verbose:
                print(f"[Tools to call: {[tc['name'] for tc in tool_calls]}]")

            # Add assistant message with tool calls (required for proper conversation flow)
            # Store the raw response message which includes tool_calls
            if response.get("choices") and response["choices"][0].get("message"):
                assistant_message = response["choices"][0]["message"]
                self.conversation_history.append(assistant_message)

            # Process tool calls
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_input = tool_call["input"]
                tool_call_id = tool_call.get("id", "")

                if verbose:
                    print(f"  Calling {tool_name}({json.dumps(tool_input)})...")

                # Execute tool
                result = self.call_tool(tool_name, tool_input)

                if verbose:
                    if result["success"]:
                        result_preview = str(result["output"])[:200]
                        print(f"    → Success: {result_preview}...")
                    else:
                        print(f"    → Error: {result['error']}")

                # Format tool result according to OpenAI spec
                # Truncate large results to avoid context overflow
                result_str = json.dumps(result)
                if len(result_str) > 5000:
                    result_truncated = {
                        "success": result.get("success"),
                        "output": str(result.get("output"))[:4500] + "...[truncated]",
                        "error": result.get("error")
                    }
                    result_str = json.dumps(result_truncated)

                tool_result = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result_str
                }
                tool_results.append(tool_result)

            # Add all tool results to conversation
            self.conversation_history.extend(tool_results)

        if verbose:
            print("\n[Max iterations reached]")

        # Return last assistant message or empty string
        for msg in reversed(self.conversation_history):
            if msg["role"] == "assistant":
                return msg["content"]

        return ""

    async def run_async(self, user_question: str, verbose: bool = False) -> str:
        """Async version of run() for parallel specialist execution.

        Args:
            user_question: The question to answer
            verbose: Print intermediate steps

        Returns:
            Final response from the agent
        """
        # Run the synchronous version in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, user_question, verbose)

    def get_critic_prompt(self) -> str:
        """Get the system prompt for the scientific critic.

        Returns:
            Critic system prompt string
        """
        return """You are a Scientific Critic specializing in biomedical research evaluation. Your role is to rigorously review scientific answers for accuracy, completeness, and methodological soundness.

## Evaluation Criteria

1. **Scientific Rigor**:
   - Are claims supported by evidence?
   - Are appropriate statistical methods used?
   - Are limitations acknowledged?
   - Is the reasoning logically sound?

2. **Completeness**:
   - Does the answer address all parts of the question?
   - Are key details missing?
   - Should additional analyses be performed?

3. **Data Analysis Quality**:
   - Are the data analyses appropriate?
   - Are results interpreted correctly?
   - Are there computational errors?

4. **Literature Support**:
   - Are claims consistent with the literature?
   - Are ALL papers cited with BOTH title and PMID in format: "Title" (PMID: 12345678)?
   - Should additional papers be cited?
   - Are there contradictory findings that should be discussed?

5. **Practical Feasibility**:
   - Are proposed strategies realistic?
   - Are technical challenges acknowledged?
   - Are resource requirements reasonable?

## Your Task

Review the provided answer and provide constructive feedback. Focus on:
- **Strengths**: What is done well
- **Issues**: Errors, gaps, or weaknesses
- **Improvements**: Specific suggestions to enhance the answer

Be direct but constructive. Prioritize accuracy and completeness over minor stylistic issues.
Do NOT provide a rewritten answer - only critique and suggestions.

If the answer is of high quality and scientifically sound, you may approve it with minor suggestions."""

    def run_with_critic(self, user_question: str, verbose: bool = False, max_refinement_rounds: int = 1) -> tuple[str, str, str]:
        """Run the agent with critic feedback loop.

        Args:
            user_question: The question to answer
            verbose: Print intermediate steps
            max_refinement_rounds: Maximum number of refinement iterations (default: 1)

        Returns:
            Tuple of (initial_answer, critique, final_answer)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"RUNNING WITH CRITIC FEEDBACK")
            print(f"{'='*60}\n")

        # Step 1: Get initial answer from main agent
        if verbose:
            print("[STEP 1: Main Agent Analysis]")
        initial_answer = self.run(user_question, verbose=verbose)

        if not initial_answer:
            return "", "No answer produced by agent", ""

        # Step 2: Get critic feedback
        if verbose:
            print(f"\n{'='*60}")
            print("[STEP 2: Scientific Critic Review]")
            print(f"{'='*60}\n")

        # Create a new agent instance for the critic with the same config
        critic_agent = BioinformaticsAgent(
            api_key=None,  # Reuse existing credentials
            model=self.client.model if hasattr(self.client, 'model') else "claude-sonnet-4-20250514",
            provider="anthropic" if isinstance(self.client, AnthropicClient) else "openrouter"
        )

        # Build critic prompt
        critic_question = f"""Please review the following scientific answer for rigor, completeness, and accuracy.

ORIGINAL QUESTION:
{user_question}

ANSWER TO REVIEW:
{initial_answer}

Provide a structured critique with:
1. Strengths (what is done well)
2. Issues (errors, gaps, or weaknesses - be specific)
3. Improvements (specific suggestions to enhance the answer)

If the answer is high quality and scientifically sound, you may approve it."""

        # Override the critic's system prompt
        original_get_system_prompt = critic_agent.get_system_prompt
        critic_agent.get_system_prompt = self.get_critic_prompt

        # Run critic (without tools, just reasoning)
        critique = ""
        critic_messages = [{"role": "user", "content": critic_question}]

        call_params = {
            "messages": critic_messages,
            "tools": [],  # Critic doesn't use tools
            "temperature": 0.3,  # Lower temperature for consistent evaluation
            "max_tokens": get_max_tokens_for_model(self.client.model),
        }

        if isinstance(self.client, AnthropicClient):
            call_params["system"] = self.get_critic_prompt()

        response = self.client.create_message(**call_params)
        critique = self.client.get_response_text(response)

        if verbose:
            print(f"Critic Feedback:\n{critique}\n")

        # Restore original system prompt
        critic_agent.get_system_prompt = original_get_system_prompt

        # Step 3: Check if refinement needed
        # Simple heuristic: if critique mentions serious issues, refine
        needs_refinement = any(keyword in critique.lower() for keyword in [
            "error", "incorrect", "missing", "should", "needs", "improve",
            "gap", "weakness", "concern", "problem"
        ])

        if not needs_refinement or max_refinement_rounds == 0:
            if verbose:
                print("[STEP 3: No refinement needed - answer approved]\n")
            return initial_answer, critique, initial_answer

        # Step 4: Refine answer based on critique
        if verbose:
            print(f"\n{'='*60}")
            print("[STEP 3: Refining Answer Based on Feedback]")
            print(f"{'='*60}\n")

        refinement_question = f"""Based on the scientific critique below, please revise and improve your previous answer.

ORIGINAL QUESTION:
{user_question}

YOUR PREVIOUS ANSWER:
{initial_answer}

CRITIC FEEDBACK:
{critique}

Please provide an improved answer that addresses the critique's suggestions. Focus on fixing errors, filling gaps, and adding missing analyses."""

        # Clear conversation history and run refinement
        final_answer = self.run(refinement_question, verbose=verbose)

        if verbose:
            print(f"\n{'='*60}")
            print("[COMPLETE: Answer refined based on critic feedback]")
            print(f"{'='*60}\n")

        return initial_answer, critique, final_answer


def create_agent(api_key: Optional[str] = None, model: Optional[str] = None, provider: Optional[str] = None, data_dir: str = "/home.galaxy4/sumin/project/aisci/Competition_Data", input_dir: Optional[str] = None) -> BioinformaticsAgent:
    """Factory function to create an agent.

    Args:
        api_key: API key (Anthropic or OpenRouter)
        model: Model identifier (defaults based on provider)
        provider: 'anthropic' or 'openrouter' (defaults to API_PROVIDER env var)
        data_dir: Path to database directory (Drug databases, PPI, GWAS, etc.)
        input_dir: Path to question-specific input data (defaults to data_dir)

    Returns:
        BioinformaticsAgent instance
    """
    # Get provider from env if not specified
    if provider is None:
        provider = os.getenv("API_PROVIDER", "anthropic")

    # Set default models based on provider
    if model is None:
        if provider == "anthropic":
            model = "claude-sonnet-4-20250514"
        else:
            model = "anthropic/claude-sonnet-4"

    return BioinformaticsAgent(api_key=api_key, model=model, provider=provider, data_dir=data_dir, input_dir=input_dir)


class ScientificAgent(BioinformaticsAgent):
    """A persona-based agent for Virtual Lab architecture.

    Extends BioinformaticsAgent with dynamic role-playing capabilities,
    allowing the agent to take on specific scientific roles (e.g., PI,
    Immunologist, Computational Biologist, Critic) in a team meeting setting.
    """

    def __init__(
        self,
        persona: AgentPersona,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        data_dir: str = "/home.galaxy4/sumin/project/aisci/Competition_Data",
        input_dir: Optional[str] = None
    ):
        """Initialize a scientific agent with a specific persona.

        Args:
            persona: AgentPersona defining the agent's role, expertise, and goals
            api_key: API key (Anthropic or OpenRouter)
            model: Model to use
            provider: 'anthropic' or 'openrouter'
            data_dir: Path to database directory (Drug databases, PPI, GWAS, etc.)
            input_dir: Path to question-specific input data (defaults to data_dir)
        """
        super().__init__(api_key, model, provider, data_dir, input_dir)
        self.persona = persona

    def get_system_prompt(self) -> str:
        """Get dynamic system prompt based on the agent's persona.

        Returns:
            System prompt string tailored to this agent's role
        """
        return f"""You are {self.persona.title}.

**Expertise:** {self.persona.expertise}
**Goal:** {self.persona.goal}
**Role:** {self.persona.role}

You are participating in a Virtual Lab research meeting with other scientific experts.
Each participant brings specialized knowledge to solve complex biomedical problems.

## Your Responsibilities
- Apply your specific expertise to analyze the problem from your unique perspective
- Use available tools (Python, PubMed search, database queries) when needed for your analysis
- Contribute concise, scientifically rigorous insights
- Build on and reference other team members' contributions when relevant
- Acknowledge limitations outside your expertise area
- Be precise and avoid speculation beyond your domain

## Communication Style
- Be direct and scientific in your communication
- Structure your contributions clearly
- Use appropriate technical terminology for your field
- Cite evidence and data when making claims
- Propose actionable next steps when appropriate

Remember: You are ONE expert on a team. Your goal is to contribute your specialized
knowledge to the collective scientific effort, not to solve everything alone."""
