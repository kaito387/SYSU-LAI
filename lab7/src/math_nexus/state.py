"""State definitions for the MathNexus multi-agent system."""

from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
import operator


ProblemType = Literal["computation", "choice", "short_answer", "proof"]


class ToolCall(TypedDict):
    """Record of a tool invocation."""
    tool_name: str
    input: str
    output: str


class ToTNode(TypedDict):
    """A node in the Tree of Thoughts."""
    id: str
    content: str
    score: float
    children: List[str]  # child node IDs
    parent: Optional[str]


class AgentState(TypedDict):
    """Shared state across all agents in the LangGraph workflow."""

    # Input
    problem: str

    # Orchestrator output
    problem_type: Optional[ProblemType]
    classification_rationale: Optional[str]

    # Solver / Prover output
    solution: Optional[str]
    reasoning_trace: Annotated[List[str], operator.add]

    # ToT state (for proof problems)
    tot_nodes: Optional[Dict[str, ToTNode]]
    tot_root_id: Optional[str]
    tot_best_path: Optional[List[str]]

    # ReAct state (for computation problems)
    react_steps: Annotated[List[Dict[str, str]], operator.add]

    # Tool call history
    tool_history: Annotated[List[ToolCall], operator.add]

    # Validator output
    validation_result: Optional[Dict[str, Any]]

    # Reflector output
    reflection: Optional[str]
    improvement_suggestions: Optional[List[str]]

    # Iteration control
    attempt_count: int
    max_attempts: int

    # Long-term memory: retrieved similar problems
    retrieved_memories: Optional[List[Dict[str, Any]]]

    # Final output
    final_answer: Optional[str]
    solved: bool

    # Messages for the chat model
    messages: Annotated[List[Dict[str, Any]], operator.add]


def create_initial_state(problem: str, max_attempts: int = 3) -> AgentState:
    """Create the initial state for a new problem."""
    return AgentState(
        problem=problem,
        problem_type=None,
        classification_rationale=None,
        solution=None,
        reasoning_trace=[],
        tot_nodes=None,
        tot_root_id=None,
        tot_best_path=None,
        react_steps=[],
        tool_history=[],
        validation_result=None,
        reflection=None,
        improvement_suggestions=None,
        attempt_count=0,
        max_attempts=max_attempts,
        retrieved_memories=None,
        final_answer=None,
        solved=False,
        messages=[],
    )
