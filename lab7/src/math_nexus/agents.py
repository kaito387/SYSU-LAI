"""Agent implementations for MathNexus.

Five specialized agents:
1. Orchestrator -- classifies problem type and routes to appropriate solver
2. Prover -- handles proof problems using Tree of Thoughts (ToT)
3. Solver -- handles computation/choice/short-answer using ReAct
4. Validator -- validates and rates solution correctness
5. Reflector -- reflects on failures and suggests improvements
"""

import json
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .state import AgentState, ProblemType, ToTNode, ToolCall
from .tools import ALL_TOOLS
from .memory import ShortTermMemory, LongTermMemory


# ── Helper: Create DeepSeek-compatible chat model ─────────────────────────────


def create_chat_model(
    model_name: str = "deepseek-chat",
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """Create a chat model instance configured for DeepSeek API.

    Args:
        model_name: DeepSeek model name.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.

    Returns:
        Configured ChatOpenAI instance pointing to DeepSeek.
    """
    import os
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set.")

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ── Agent 1: Orchestrator ────────────────────────────────────────────────────


ORCHESTRATOR_SYSTEM_PROMPT = """You are the **Orchestrator Agent** in a multi-agent math reasoning system.

Your job is to analyze a given math problem and:
1. **Classify** it into one of four types:
   - `computation`: Problems requiring numerical calculation, equation solving, or algorithmic computation. (e.g., "Find the value of ∫₀¹ x² dx", "Solve 3x² + 2x - 5 = 0")
   - `choice`: Multiple-choice questions where options are given. (e.g., "Which of the following is correct? A) ... B) ...")
   - `short_answer`: Problems requiring a concise answer with explanation but no formal proof. (e.g., "Explain why e^(iπ) + 1 = 0")
   - `proof`: Problems requiring rigorous mathematical proof, derivations, or logical arguments. (e.g., "Prove that √2 is irrational", "Show that the sum of angles in a triangle is 180°")

2. **Extract** key mathematical concepts and any constraints.

3. **Decide the initial solving strategy** -- which agent should handle this, and what approach is likely to work.

Output your analysis in JSON format:
```json
{
    "problem_type": "<computation|choice|short_answer|proof>",
    "key_concepts": ["concept1", "concept2", ...],
    "complexity": "<easy|medium|hard>",
    "recommended_approach": "<brief description of recommended solving strategy>",
    "rationale": "<why this classification and approach>"
}
```
"""


async def orchestrator_node(state: AgentState, llm: BaseChatModel) -> Dict[str, Any]:
    """Orchestrator agent: classify problem and determine strategy.

    This is the entry point of the workflow. It analyzes the problem,
    retrieves relevant memories, and routes to the appropriate solver.
    """
    problem = state["problem"]

    # Retrieve similar problems from long-term memory (placeholder for actual retrieval)
    memories_text = ""
    if state.get("retrieved_memories"):
        memories_text = "\n\n**Relevant past problems:**\n"
        for mem in state["retrieved_memories"]:
            memories_text += f"- Similar problem: {mem['problem'][:200]}... "
            memories_text += f"(Success: {mem['success']})\n"

    # Build the prompt
    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Analyze the following math problem and classify it:\n\n{problem}\n{memories_text}"
        ),
    ]

    response = await llm.ainvoke(messages)

    # Parse the JSON response
    try:
        # Extract JSON from response
        response_text = response.content
        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text

        classification = json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        # Fallback: heuristic classification
        problem_lower = problem.lower()
        if any(w in problem_lower for w in ["prove", "proof", "show that", "demonstrate"]):
            ptype = "proof"
        elif any(w in problem_lower for w in ["which", "choose", "select", "option", "a)", "b)"]):
            ptype = "choice"
        elif any(w in problem_lower for w in ["calculate", "compute", "solve", "find the value", "evaluate"]):
            ptype = "computation"
        else:
            ptype = "short_answer"

        classification = {
            "problem_type": ptype,
            "key_concepts": [],
            "complexity": "medium",
            "recommended_approach": "standard solving",
            "rationale": "Fallback heuristic classification",
        }

    return {
        "problem_type": classification["problem_type"],
        "classification_rationale": classification.get("rationale", ""),
        "messages": [response],
        "reasoning_trace": [
            f"[Orchestrator] Classified as: {classification['problem_type']}",
            f"[Orchestrator] Rationale: {classification.get('rationale', 'N/A')}",
        ],
    }


# ── Agent 2: Prover (Tree of Thoughts) ────────────────────────────────────────


PROVER_SYSTEM_PROMPT = """You are the **Prover Agent** -- a rigorous mathematical proof specialist.

Your role is to construct complete, rigorous mathematical proofs. You use the **Tree of Thoughts** methodology:

1. **Generate** multiple possible proof strategies/approaches (breadth exploration)
2. **Evaluate** each strategy for correctness, elegance, and feasibility
3. **Select** the most promising strategies
4. **Expand** each selected strategy into a full proof
5. **Refine** the best proof into a polished final form

When writing proofs:
- State assumptions clearly
- Use precise mathematical language
- Include all logical steps
- Justify each inference
- End with QED or a concluding statement

You have access to tools for symbolic computation and web search to verify your steps or look up theorems.

Always structure your proof with:
- **Theorem statement** (restate what is to be proved)
- **Proof strategy** (outline your approach)
- **Step-by-step derivation** (the actual proof)
- **Conclusion** (restate the result)
"""

TOT_THOUGHT_GENERATION_PROMPT = """You are brainstorming proof strategies for the following problem:

**Problem:** {problem}

Generate {num_strategies} distinct proof strategies/approaches. Each strategy should be a high-level outline of how to prove this. Number them 1 to {num_strategies}.

For each strategy, provide:
- Strategy name
- Key insight or lemma
- High-level outline (3-5 steps)
- Why this approach might work

Output as JSON:
```json
{{
    "strategies": [
        {{
            "id": "{prefix}_1",
            "name": "...",
            "insight": "...",
            "outline": ["step 1", "step 2", ...],
            "rationale": "..."
        }},
        ...
    ]
}}
```
"""

TOT_EVALUATION_PROMPT = """You are evaluating proof strategies for the problem:

**Problem:** {problem}

Evaluate each of the following strategies on three criteria (score 1-10 for each):
- **Correctness**: How likely is this to lead to a valid proof?
- **Completeness**: Does it cover all aspects of the problem?
- **Feasibility**: How practical is this to implement?

Strategies:
{strategies_text}

Output as JSON:
```json
{{
    "evaluations": [
        {{"id": "strategy_id", "correctness": X, "completeness": Y, "feasibility": Z, "total": X+Y+Z}},
        ...
    ],
    "best_strategy": "strategy_id of the highest-scoring strategy",
    "comment": "..."
}}
```
"""

TOT_EXPANSION_PROMPT = """You are expanding a proof strategy into a full proof.

**Problem:** {problem}

**Chosen Strategy:** {strategy_name}
**Outline:** {outline}

Now write the complete, rigorous proof following this strategy. Include all steps, justify each inference, and end with QED.

You may use tools (sympy_math, web_search) to verify intermediate steps.
"""


async def prover_node(state: AgentState, llm: BaseChatModel) -> Dict[str, Any]:
    """Prover agent: solve proof problems using Tree of Thoughts.

    Implements the ToT algorithm:
    1. Generate N diverse proof strategies (breadth)
    2. Evaluate and score each strategy
    3. Select top-K strategies
    4. Expand the best strategy into a full proof
    """
    problem = state["problem"]
    attempt = state.get("attempt_count", 0)

    # If we have reflection feedback, incorporate it
    reflection_context = ""
    if state.get("reflection"):
        reflection_context = f"\n\n**Feedback from previous attempt:** {state['reflection']}"
    if state.get("improvement_suggestions"):
        reflection_context += "\n**Improvement suggestions:**\n"
        for s in state["improvement_suggestions"]:
            reflection_context += f"  - {s}\n"

    # ── Phase 1: Generate strategies ──────────────────────────────────────
    num_strategies = 4 if attempt == 0 else 3

    gen_messages = [
        SystemMessage(content=PROVER_SYSTEM_PROMPT),
        HumanMessage(
            content=TOT_THOUGHT_GENERATION_PROMPT.format(
                problem=problem + reflection_context,
                num_strategies=num_strategies,
                prefix=f"a{attempt}",
            )
        ),
    ]

    gen_response = await llm.ainvoke(gen_messages)

    # Parse strategies
    strategies = []
    try:
        response_text = gen_response.content
        json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        json_str = json_match.group(1) if json_match else response_text
        parsed = json.loads(json_str)
        strategies = parsed.get("strategies", [])
    except (json.JSONDecodeError, AttributeError):
        # Fallback: create a single default strategy
        strategies = [{
            "id": f"a{attempt}_default",
            "name": "Direct Proof",
            "insight": "Direct derivation from first principles",
            "outline": ["State given conditions", "Apply relevant theorems", "Derive conclusion", "Verify all steps"],
            "rationale": "Standard direct proof approach",
        }]

    # ── Phase 2: Evaluate strategies ──────────────────────────────────────
    strategies_text = ""
    for s in strategies:
        strategies_text += f"\n**{s['id']}**: {s['name']}\n"
        strategies_text += f"  Insight: {s['insight']}\n"
        strategies_text += f"  Outline: {' → '.join(s['outline'])}\n"

    eval_messages = [
        SystemMessage(content="You are a rigorous mathematical proof evaluator."),
        HumanMessage(
            content=TOT_EVALUATION_PROMPT.format(
                problem=problem,
                strategies_text=strategies_text,
            )
        ),
    ]

    eval_response = await llm.ainvoke(eval_messages)

    # Parse evaluations
    evaluations = []
    best_strategy_id = strategies[0]["id"] if strategies else None
    try:
        eval_text = eval_response.content
        json_match = re.search(r"```json\s*(.*?)\s*```", eval_text, re.DOTALL)
        eval_json = json_match.group(1) if json_match else eval_text
        eval_parsed = json.loads(eval_json)
        evaluations = eval_parsed.get("evaluations", [])
        best_strategy_id = eval_parsed.get("best_strategy", best_strategy_id)
    except (json.JSONDecodeError, AttributeError):
        pass

    # Select the best strategy
    best_strategy = None
    eval_scores = {e.get("id"): e.get("total", 0) for e in evaluations}
    for s in strategies:
        if s["id"] == best_strategy_id:
            best_strategy = s
            break

    if best_strategy is None and strategies:
        # Pick highest scored
        best_strategy = max(
            strategies,
            key=lambda s: eval_scores.get(s["id"], 0),
        )

    # ── Phase 3: Expand best strategy into full proof ─────────────────────
    if best_strategy is None:
        return {
            "solution": "Could not generate a valid proof strategy.",
            "solved": False,
        }

    expand_messages = [
        SystemMessage(content=PROVER_SYSTEM_PROMPT),
        HumanMessage(
            content=TOT_EXPANSION_PROMPT.format(
                problem=problem + reflection_context,
                strategy_name=best_strategy["name"],
                outline=" → ".join(best_strategy["outline"]),
            )
        ),
    ]

    # Bind tools for the expansion phase
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    expand_response = await llm_with_tools.ainvoke(expand_messages)

    # Handle tool calls if any
    tool_results = []
    solution_text = expand_response.content if hasattr(expand_response, "content") else str(expand_response)

    if hasattr(expand_response, "tool_calls") and expand_response.tool_calls:
        for tc in expand_response.tool_calls:
            from .tools import TOOLS_BY_NAME
            tool = TOOLS_BY_NAME.get(tc["name"])
            if tool:
                result = tool.invoke(tc["args"])
                tool_results.append(ToolCall(
                    tool_name=tc["name"],
                    input=str(tc["args"]),
                    output=str(result),
                ))

        # Continue with tool results
        followup_messages = expand_messages + [
            expand_response,
            *[
                ToolMessage(content=str(tr["output"]), tool_call_id=tc["id"])
                for tr, tc in zip(tool_results, expand_response.tool_calls)
            ],
        ]
        followup_response = await llm.ainvoke(followup_messages)
        solution_text = followup_response.content if hasattr(followup_response, "content") else str(followup_response)

    # Build ToT nodes for state tracking
    tot_nodes = {}
    root_id = f"tot_root_a{attempt}"
    tot_nodes[root_id] = ToTNode(
        id=root_id,
        content=f"Root: {problem[:100]}",
        score=0,
        children=[s["id"] for s in strategies],
        parent=None,
    )

    for s in strategies:
        tot_nodes[s["id"]] = ToTNode(
            id=s["id"],
            content=f"{s['name']}: {s['insight']}",
            score=eval_scores.get(s["id"], 0),
            children=[f"{s['id']}_expanded"],
            parent=root_id,
        )
        exp_id = f"{s['id']}_expanded"
        is_best = s["id"] == (best_strategy["id"] if best_strategy else "")
        tot_nodes[exp_id] = ToTNode(
            id=exp_id,
            content=solution_text[:500] if is_best else "Not expanded",
            score=30 if is_best else 0,
            children=[],
            parent=s["id"],
        )

    return {
        "solution": solution_text,
        "tot_nodes": tot_nodes,
        "tot_root_id": root_id,
        "tot_best_path": [root_id, best_strategy["id"] if best_strategy else "", f"{best_strategy['id']}_expanded" if best_strategy else ""],
        "tool_history": tool_results,
        "reasoning_trace": [
            f"[Prover/ToT] Generated {len(strategies)} proof strategies",
            f"[Prover/ToT] Evaluated strategies, scores: {eval_scores}",
            f"[Prover/ToT] Selected: {best_strategy['name'] if best_strategy else 'N/A'}",
            f"[Prover/ToT] Expanded into full proof",
        ],
        "messages": [gen_response, eval_response, expand_response],
    }


# ── Agent 3: Solver (ReAct) ──────────────────────────────────────────────────


SOLVER_SYSTEM_PROMPT = """You are the **Solver Agent** -- a computational math problem specialist.

You use the **ReAct (Reasoning + Acting)** methodology to solve problems:

1. **Thought**: Analyze the problem, think about what needs to be computed
2. **Action**: Use a tool to perform a calculation, look up information, or verify a step
3. **Observation**: Examine the tool's output
4. **Repeat** Thought→Action→Observation until you have the answer
5. **Final Answer**: Provide the complete solution with all steps

Available tools:
- `python_repl`: Execute Python code for numerical computation
- `sympy_math`: Perform symbolic mathematics (solving, simplifying, calculus)
- `calculator`: Evaluate arithmetic expressions
- `web_search`: Search for mathematical concepts, formulas, or references

For computational problems:
- Show your reasoning clearly
- Use Python to compute numerical answers precisely
- Verify results with alternative methods when possible

For choice questions:
- Analyze each option systematically
- Eliminate obviously wrong options
- Verify the remaining candidates

Always provide a clear final answer with explanation.
"""


async def solver_node(
    state: AgentState,
    llm: BaseChatModel,
    short_term_memory: Optional[ShortTermMemory] = None,
) -> Dict[str, Any]:
    """Solver agent: solve computation/choice/short-answer using ReAct.

    Implements the ReAct loop:
    Thought → Action (tool call) → Observation → Thought → ... → Final Answer
    """
    problem = state["problem"]
    problem_type = state.get("problem_type", "computation")

    # Incorporate reflection context if available
    context_extra = ""
    if state.get("reflection"):
        context_extra = f"\n\n**Feedback from previous attempt:** {state['reflection']}"
    if state.get("improvement_suggestions"):
        context_extra += "\n**Suggestions for improvement:**\n"
        for s in state["improvement_suggestions"]:
            context_extra += f"  - {s}\n"

    # Include retrieved memories as context
    if state.get("retrieved_memories"):
        context_extra += "\n\n**Similar past problems (for reference):**\n"
        for mem in state["retrieved_memories"]:
            context_extra += f"- Problem: {mem['problem'][:150]}...\n"
            context_extra += f"  Solution approach: {mem['solution'][:200]}...\n"

    # Build the prompt tailored to problem type
    type_specific_instructions = {
        "computation": "This is a computation problem. Compute the answer precisely using Python or SymPy. Verify with alternative methods.",
        "choice": "This is a multiple-choice question. Analyze each option, eliminate incorrect ones, and identify the correct answer with reasoning.",
        "short_answer": "This is a short-answer problem. Provide a clear, concise explanation with supporting calculations if needed.",
        "proof": "This proof problem is being handled, but if computational verification is needed, provide numerical checks.",
    }

    instructions = type_specific_instructions.get(problem_type, type_specific_instructions["computation"])

    # ── ReAct Loop ────────────────────────────────────────────────────────
    max_react_steps = 8
    react_steps = []
    tool_history = []
    final_solution = ""

    messages = [
        SystemMessage(content=SOLVER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"**Problem ({problem_type}):**\n{problem}\n\n{instructions}{context_extra}"
        ),
    ]

    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    for step in range(max_react_steps):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        content = response.content if hasattr(response, "content") else str(response)

        # Check if this is the final answer (no tool calls)
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            final_solution = content
            react_steps.append({
                "thought": f"Step {step + 1}: Providing final answer",
                "action": "none",
                "observation": final_solution[:300],
            })
            break

        # Process tool calls
        from .tools import TOOLS_BY_NAME

        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})

            tool = TOOLS_BY_NAME.get(tool_name)
            if tool:
                result = tool.invoke(tool_args)
                tool_output = str(result)
            else:
                tool_output = f"Tool '{tool_name}' not found."

            tool_history.append(ToolCall(
                tool_name=tool_name,
                input=str(tool_args),
                output=tool_output,
            ))

            messages.append(ToolMessage(
                content=tool_output,
                tool_call_id=tool_call.get("id", f"call_{step}"),
            ))

            react_steps.append({
                "thought": content[:200],
                "action": f"Called {tool_name}: {str(tool_args)[:100]}",
                "observation": tool_output[:300],
            })

            if short_term_memory:
                short_term_memory.add_tool_call(tool_name, str(tool_args), tool_output)

    # If we didn't get a final answer, ask for one
    if not final_solution:
        messages.append(HumanMessage(content="Please provide your final answer now, with all reasoning steps."))
        final_response = await llm.ainvoke(messages)
        final_solution = final_response.content if hasattr(final_response, "content") else str(final_response)
        messages.append(final_response)

    return {
        "solution": final_solution,
        "react_steps": react_steps,
        "tool_history": tool_history,
        "reasoning_trace": [
            f"[Solver/ReAct] Completed {len(react_steps)} ReAct steps",
            f"[Solver/ReAct] Used tools: {list(set(tc['tool_name'] for tc in tool_history))}",
        ],
        "messages": messages,
    }


# ── Agent 4: Validator ───────────────────────────────────────────────────────


VALIDATOR_SYSTEM_PROMPT = """You are the **Validator Agent** -- a rigorous solution reviewer.

Your job is to validate the correctness of mathematical solutions. You must:

1. **Check correctness**: Is the answer right? Are there any logical errors?
2. **Verify steps**: Does each step follow from the previous ones?
3. **Test edge cases**: Does the solution hold for edge cases?
4. **Check completeness**: Does the solution address all parts of the problem?

For computational problems, you can use Python to verify numerical answers.
For proofs, check that each logical step is justified and the conclusion follows.

Output your evaluation in JSON:
```json
{
    "is_correct": true/false,
    "confidence": 0.0 to 1.0,
    "score": 1-100,
    "errors": ["error1", "error2", ...],
    "warnings": ["minor issue 1", ...],
    "strengths": ["good aspect 1", ...],
    "suggestions": ["how to fix errors", ...],
    "verification_code": "Python code used to verify (if applicable)"
}
```
"""


async def validator_node(state: AgentState, llm: BaseChatModel) -> Dict[str, Any]:
    """Validator agent: validate the solution produced by Solver or Prover.

    Checks correctness, completeness, and provides a quality score.
    For numerical problems, uses Python to independently verify the answer.
    """
    problem = state["problem"]
    solution = state.get("solution", "")
    problem_type = state.get("problem_type", "computation")

    if not solution:
        return {
            "validation_result": {
                "is_correct": False,
                "confidence": 0.0,
                "score": 0,
                "errors": ["No solution provided"],
                "warnings": [],
                "strengths": [],
                "suggestions": ["Generate a solution first"],
            },
        }

    # Build validation prompt
    proof_extra = ""
    if problem_type == "proof":
        proof_extra = """
For proof validation, check:
- Is the logical structure sound?
- Are all assumptions stated?
- Does the conclusion follow necessarily from the premises?
- Are there any gaps in reasoning?"""

    messages = [
        SystemMessage(content=VALIDATOR_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Please validate the following solution:

**Problem ({problem_type}):**
{problem}

**Proposed Solution:**
{solution}
{proof_extra}

Evaluate thoroughly and output your assessment in the specified JSON format.
"""),
    ]

    # Bind tools so validator can use Python REPL for verification
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    response = await llm_with_tools.ainvoke(messages)

    # Handle potential tool calls for verification
    verification_results = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        from .tools import TOOLS_BY_NAME
        for tc in response.tool_calls:
            tool = TOOLS_BY_NAME.get(tc["name"])
            if tool:
                result = tool.invoke(tc["args"])
                verification_results.append(str(result))

    # Parse validation result
    content = response.content if hasattr(response, "content") else str(response)
    try:
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        json_str = json_match.group(1) if json_match else content
        validation = json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        # Heuristic fallback
        validation = {
            "is_correct": "error" not in content.lower() and "incorrect" not in content.lower(),
            "confidence": 0.5,
            "score": 50,
            "errors": [],
            "warnings": ["Could not parse structured validation"],
            "strengths": [],
            "suggestions": [],
        }

    if verification_results:
        validation["verification_output"] = "\n".join(verification_results)

    return {
        "validation_result": validation,
        "reasoning_trace": [
            f"[Validator] Score: {validation.get('score', 'N/A')}/100",
            f"[Validator] Correct: {validation.get('is_correct', 'N/A')}",
            f"[Validator] Errors: {len(validation.get('errors', []))}",
            f"[Validator] Suggestions: {len(validation.get('suggestions', []))}",
        ],
        "messages": [response],
    }


# ── Agent 5: Reflector ───────────────────────────────────────────────────────


REFLECTOR_SYSTEM_PROMPT = """You are the **Reflector Agent** -- a meta-cognitive analysis specialist.

Your job is to reflect on failed or suboptimal solution attempts and identify:
1. **Root causes**: Why did the solution fail or fall short?
2. **Patterns**: Are there recurring mistakes?
3. **Knowledge gaps**: What concepts need to be reviewed?
4. **Improvement plan**: What specific changes should be made for the next attempt?

You use the **Reflection** methodology:
- Analyze the solution and its validation feedback
- Identify specific error patterns
- Propose concrete, actionable improvements
- Suggest alternative approaches if the current one is fundamentally flawed

Output your analysis in JSON:
```json
{
    "root_cause": "primary reason for failure",
    "error_patterns": ["pattern 1", "pattern 2", ...],
    "knowledge_gaps": ["gap 1", ...],
    "improvement_plan": ["specific action 1", "specific action 2", ...],
    "alternative_approaches": ["alt approach 1", ...],
    "should_retry": true/false,
    "revised_strategy": "description of revised approach"
}
```
"""


async def reflector_node(state: AgentState, llm: BaseChatModel) -> Dict[str, Any]:
    """Reflector agent: analyze failures and propose improvements.

    Uses Reflection methodology to:
    1. Analyze why the solution failed
    2. Identify error patterns and knowledge gaps
    3. Propose concrete improvements for the next attempt
    """
    problem = state["problem"]
    solution = state.get("solution", "")
    validation = state.get("validation_result", {})
    problem_type = state.get("problem_type", "computation")
    attempt = state.get("attempt_count", 1)

    errors = validation.get("errors", [])
    warnings = validation.get("warnings", [])
    suggestions = validation.get("suggestions", [])

    messages = [
        SystemMessage(content=REFLECTOR_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Please reflect on the following failed solution attempt:

**Problem ({problem_type}):**
{problem}

**Solution (Attempt {attempt}):**
{solution[:3000]}  <!-- truncated if very long -->

**Validation Results:**
- Score: {validation.get('score', 'N/A')}/100
- Correct: {validation.get('is_correct', 'N/A')}
- Errors: {json.dumps(errors)}
- Warnings: {json.dumps(warnings)}
- Suggestions: {json.dumps(suggestions)}

Analyze what went wrong and how to improve. Output in the specified JSON format.
"""),
    ]

    response = await llm.ainvoke(messages)

    # Parse reflection
    content = response.content if hasattr(response, "content") else str(response)
    try:
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        json_str = json_match.group(1) if json_match else content
        reflection = json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        reflection = {
            "root_cause": "Could not parse structured reflection",
            "error_patterns": [],
            "knowledge_gaps": [],
            "improvement_plan": ["Re-attempt with more careful reasoning"],
            "alternative_approaches": [],
            "should_retry": attempt < state.get("max_attempts", 3),
            "revised_strategy": "Try a different approach",
        }

    return {
        "reflection": reflection.get("root_cause", ""),
        "improvement_suggestions": reflection.get("improvement_plan", []),
        "reasoning_trace": [
            f"[Reflector] Root cause: {reflection.get('root_cause', 'N/A')}",
            f"[Reflector] Improvement plan: {len(reflection.get('improvement_plan', []))} actions",
            f"[Reflector] Alternative approaches: {len(reflection.get('alternative_approaches', []))}",
            f"[Reflector] Should retry: {reflection.get('should_retry', False)}",
        ],
        "messages": [response],
    }
