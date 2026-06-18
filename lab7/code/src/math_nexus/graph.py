"""LangGraph workflow definition for MathNexus.

The multi-agent workflow:
    1. Orchestrator classifies the problem
    2. Route to Prover (ToT) or Solver (ReAct) based on type
    3. Validator checks the solution
    4. If correct → final answer
    5. If incorrect → Reflector analyzes failure → loop back to step 2
    6. After max attempts → return best attempt

Graph structure:

    ┌─────────────┐
    │ Orchestrator │  ← Entry point: classify problem
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │    Router    │  ← Route based on problem_type
    └──┬───────┬──┘
       │       │
  ┌────▼──┐ ┌──▼──────┐
  │ Prover│ │ Solver  │  ← Solve using ToT or ReAct
  │ (ToT) │ │ (ReAct) │
  └───┬───┘ └───┬─────┘
      │         │
      └────┬────┘
           │
    ┌──────▼──────┐
    │  Validator  │  ← Validate solution correctness
    └──┬───────┬──┘
       │       │
   correct  incorrect
       │       │
       │  ┌────▼──────┐
       │  │ Reflector │  ← Analyze failure, suggest improvements
       │  └────┬──────┘
       │       │
       │  ┌────▼──────┐
       │  │  Retry?   │  ← Check if max attempts reached
       │  └──┬─────┬──┘
       │     │     │
       │   yes    no
       │     │     │
       │  (back    │
       │   to      │
       │  Router)  │
       │           │
    ┌──▼───────────▼──┐
    │  Final Answer    │  ← Output result
    └──────────────────┘
"""

from typing import Literal, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState, ProblemType, create_initial_state
from .agents import (
    create_chat_model,
    orchestrator_node,
    prover_node,
    solver_node,
    validator_node,
    reflector_node,
)
from .memory import ShortTermMemory, LongTermMemory


# ── Router function ──────────────────────────────────────────────────────────


def router(state: AgentState) -> Literal["prover", "solver"]:
    """Route to Prover or Solver based on problem classification.

    Args:
        state: Current agent state.

    Returns:
        "prover" for proof problems, "solver" for everything else.
    """
    problem_type = state.get("problem_type", "computation")
    if problem_type == "proof":
        return "prover"
    return "solver"


def should_retry(state: AgentState) -> Literal["router", "finalize"]:
    """Determine whether to retry solving or finalize.

    Checks if the solution was correct and if max attempts haven't been exceeded.

    Args:
        state: Current agent state.

    Returns:
        "router" to retry, "finalize" to end.
    """
    validation = state.get("validation_result", {})
    is_correct = validation.get("is_correct", False)
    attempt = state.get("attempt_count", 1)
    max_attempts = state.get("max_attempts", 3)

    if is_correct:
        return "finalize"
    if attempt < max_attempts:
        return "router"
    return "finalize"


# ── Graph building ───────────────────────────────────────────────────────────


def build_graph(
    model_name: str = "deepseek-chat",
    temperature: float = 0.3,
    long_term_memory: Optional[LongTermMemory] = None,
    short_term_memory: Optional[ShortTermMemory] = None,
) -> StateGraph:
    """Build and compile the MathNexus LangGraph workflow.

    Args:
        model_name: DeepSeek model name to use.
        temperature: Model temperature.
        long_term_memory: Optional long-term memory instance.
        short_term_memory: Optional short-term memory instance.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    # Create the LLM
    llm = create_chat_model(model_name=model_name, temperature=temperature)

    # Create the graph
    workflow = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────────

    # Node: Orchestrator (classifies problem)
    # Also increments attempt_count on entry for display purposes
    async def orchestrate(state: AgentState) -> Dict[str, Any]:
        result = await orchestrator_node(state, llm)
        # Increment attempt count on first entry and re-entries
        result["attempt_count"] = state.get("attempt_count", 0) + 1
        return result

    # Node: Prover (ToT for proof problems)
    async def prove(state: AgentState) -> Dict[str, Any]:
        return await prover_node(state, llm)

    # Node: Solver (ReAct for computation/choice/short-answer)
    async def solve(state: AgentState) -> Dict[str, Any]:
        return await solver_node(state, llm, short_term_memory)

    # Node: Validator (checks solution correctness)
    async def validate(state: AgentState) -> Dict[str, Any]:
        return await validator_node(state, llm)

    # Node: Reflector (analyzes failure and suggests improvements)
    async def reflect(state: AgentState) -> Dict[str, Any]:
        return await reflector_node(state, llm)

    # Node: Finalize (prepare final answer)
    async def finalize(state: AgentState) -> Dict[str, Any]:
        """Prepare the final answer after validation or max attempts."""
        validation = state.get("validation_result", {})
        is_correct = validation.get("is_correct", False)
        score = validation.get("score", 0)
        errors = validation.get("errors", [])
        attempt = state.get("attempt_count", 1)

        solution = state.get("solution", "No solution produced.")
        problem_type = state.get("problem_type", "unknown")

        # Build final answer
        header = f"## MathNexus Solution\n"
        header += f"**Problem Type:** {problem_type}\n"
        header += f"**Validation Score:** {score}/100\n"
        header += f"**Correct:** {'✅ Yes' if is_correct else '❌ No'}\n"
        header += f"**Attempts:** {attempt}\n"

        if errors:
            header += f"**Issues identified:** {'; '.join(errors)}\n"

        final_answer = f"{header}\n### Solution\n\n{solution}"

        # Store in long-term memory
        if long_term_memory:
            try:
                key_concepts = []
                if state.get("classification_rationale"):
                    # Extract concepts from rationale
                    key_concepts = [
                        w for w in state["classification_rationale"].split()
                        if len(w) > 3
                    ][:5]
                long_term_memory.store(
                    problem=state["problem"],
                    solution=solution,
                    problem_type=problem_type,
                    success=is_correct,
                    key_concepts=key_concepts,
                )
            except Exception:
                pass  # Don't fail if memory storage fails

        return {
            "final_answer": final_answer,
            "solved": is_correct,
            "reasoning_trace": [
                f"[Finalize] Solution {'correct' if is_correct else 'incorrect'} after {attempt} attempt(s)",
                f"[Finalize] Stored in long-term memory",
            ],
        }

    workflow.add_node("orchestrator", orchestrate)
    workflow.add_node("prover", prove)
    workflow.add_node("solver", solve)
    workflow.add_node("validator", validate)
    workflow.add_node("reflector", reflect)
    workflow.add_node("finalize", finalize)

    # ── Add edges ─────────────────────────────────────────────────────────

    # Entry: orchestrator is the first node
    workflow.set_entry_point("orchestrator")

    # Orchestrator → Router → Prover or Solver
    workflow.add_conditional_edges(
        "orchestrator",
        router,
        {"prover": "prover", "solver": "solver"},
    )

    # Prover → Validator
    workflow.add_edge("prover", "validator")

    # Solver → Validator
    workflow.add_edge("solver", "validator")

    # Validator → Retry check
    workflow.add_conditional_edges(
        "validator",
        should_retry,
        {
            "router": "reflector",
            "finalize": "finalize",
        },
    )

    # Reflector → Router (retry) or Finalize
    workflow.add_conditional_edges(
        "reflector",
        lambda s: should_retry(s),
        {
            "router": "orchestrator",  # Go back through orchestrator for re-classification
            "finalize": "finalize",
        },
    )

    # Finalize → END
    workflow.add_edge("finalize", END)

    # ── Compile ───────────────────────────────────────────────────────────

    # Use in-memory checkpointer for short-term state persistence
    checkpointer = MemorySaver()
    compiled = workflow.compile(checkpointer=checkpointer)

    return compiled


# ── System wrapper ────────────────────────────────────────────────────────────


class MathNexusSystem:
    """High-level wrapper for the MathNexus multi-agent math solving system.

    Usage:
        system = MathNexusSystem()
        result = await system.solve("Prove that √2 is irrational")
        print(result["final_answer"])
    """

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        temperature: float = 0.3,
        memory_dir: str = "./math_nexus_memory",
    ):
        """Initialize the MathNexus system.

        Args:
            model_name: DeepSeek model name.
            temperature: Model temperature for generation.
            memory_dir: Directory for long-term memory persistence.
        """
        self.model_name = model_name
        self.temperature = temperature

        # Initialize memory systems
        self.long_term_memory = LongTermMemory(persist_dir=memory_dir)
        self.short_term_memory = ShortTermMemory()

        # Build the graph
        self.graph = build_graph(
            model_name=model_name,
            temperature=temperature,
            long_term_memory=self.long_term_memory,
            short_term_memory=self.short_term_memory,
        )

    async def solve(
        self,
        problem: str,
        max_attempts: int = 3,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Solve a math problem using the multi-agent system.

        Args:
            problem: The math problem text.
            max_attempts: Maximum solving attempts before giving up.
            config: Optional LangGraph config (e.g., for thread tracking).

        Returns:
            Dictionary with solution, validation, and trace information.
        """
        # Clear short-term memory for new problem
        self.short_term_memory.clear()
        self.short_term_memory.current_problem = problem

        # Retrieve similar problems from long-term memory
        retrieved = self.long_term_memory.retrieve(problem, n_results=3)

        # Create initial state
        initial_state = create_initial_state(problem, max_attempts=max_attempts)
        initial_state["retrieved_memories"] = retrieved

        # Prepare config
        if config is None:
            import hashlib
            thread_id = hashlib.md5(problem.encode()).hexdigest()[:8]
            config = {"configurable": {"thread_id": thread_id}}

        # Run the graph
        final_state = await self.graph.ainvoke(initial_state, config)

        # Record interaction in short-term memory
        self.short_term_memory.add_message("user", problem, "user")
        self.short_term_memory.add_message(
            "assistant",
            final_state.get("final_answer", "No answer produced"),
            "system",
        )

        return {
            "final_answer": final_state.get("final_answer", ""),
            "solved": final_state.get("solved", False),
            "problem_type": final_state.get("problem_type", "unknown"),
            "validation": final_state.get("validation_result", {}),
            "attempt_count": final_state.get("attempt_count", 1),
            "reasoning_trace": final_state.get("reasoning_trace", []),
            "tool_history": final_state.get("tool_history", []),
            "similar_problems": retrieved,
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory systems."""
        return {
            "long_term": self.long_term_memory.get_statistics(),
            "short_term_buffer_size": len(self.short_term_memory.buffer),
            "tool_calls_in_session": len(self.short_term_memory.tool_call_log),
            "agent_interactions": len(self.short_term_memory.agent_interactions),
        }
