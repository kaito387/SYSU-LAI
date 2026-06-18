"""MathNexus CLI — Multi-Agent Math Reasoning System.

Usage:
    python main.py "Prove that √2 is irrational"
    python main.py --interactive
    python main.py --test
"""

import asyncio
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from math_nexus import MathNexusSystem


async def solve_single(problem: str, system: MathNexusSystem):
    """Solve a single math problem and print the result."""
    print(f"\n{'='*60}")
    print(f"Problem: {problem}")
    print(f"{'='*60}\n")

    result = await system.solve(problem)

    print(result["final_answer"])
    print(f"\n{'─'*60}")
    print(f"Solved: {result['solved']}")
    print(f"Attempts: {result['attempt_count']}")
    print(f"Type: {result['problem_type']}")

    if result.get("validation"):
        v = result["validation"]
        print(f"Score: {v.get('score', 'N/A')}/100")
        if v.get("errors"):
            print(f"Errors: {', '.join(v['errors'])}")

    if result.get("tool_history"):
        print(f"\nTool usage ({len(result['tool_history'])} calls):")
        for tc in result["tool_history"]:
            print(f"  - {tc['tool_name']}: {tc['input'][:80]}")

    return result


async def interactive_mode(system: MathNexusSystem):
    """Run the system in interactive mode."""
    print("=" * 60)
    print("  MathNexus — Multi-Agent Math Reasoning System")
    print("  Type 'quit' to exit, 'stats' for memory stats")
    print("=" * 60)

    while True:
        try:
            problem = input("\n📐 Enter a math problem: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not problem:
            continue
        if problem.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if problem.lower() == "stats":
            stats = system.get_memory_stats()
            print("\nMemory Statistics:")
            print(f"  Long-term memory: {stats['long_term']}")
            print(f"  Short-term buffer: {stats['short_term_buffer_size']} messages")
            print(f"  Tool calls this session: {stats['tool_calls_in_session']}")
            continue

        await solve_single(problem, system)


# ── Test problems ─────────────────────────────────────────────────────────────

TEST_PROBLEMS = [
    # Computation
    {
        "problem": "Compute the value of ∫₀¹ x² eˣ dx.",
        "type": "computation",
        "expected_concepts": ["integration", "integration by parts", "definite integral"],
    },
    # Proof
    {
        "problem": "Prove that √2 is irrational.",
        "type": "proof",
        "expected_concepts": ["irrational numbers", "proof by contradiction", "number theory"],
    },
    # Short answer
    {
        "problem": "Explain why the derivative of eˣ is eˣ.",
        "type": "short_answer",
        "expected_concepts": ["derivative", "exponential function", "limit definition"],
    },
    # Choice
    {
        "problem": "Which of the following is the general solution to the differential equation dy/dx = ky?\nA) y = kx + C\nB) y = Ce^(kx)\nC) y = k^x + C\nD) y = C/x",
        "type": "choice",
        "expected_concepts": ["differential equations", "exponential growth", "separation of variables"],
    },
    # Computation (harder)
    {
        "problem": "Solve the system of equations: x + y + z = 6, x² + y² + z² = 14, x³ + y³ + z³ = 36. Find x·y·z.",
        "type": "computation",
        "expected_concepts": ["systems of equations", "symmetric polynomials", "Newton's identities"],
    },
    # Proof (geometry)
    {
        "problem": "Prove that the sum of the angles in any triangle is 180 degrees.",
        "type": "proof",
        "expected_concepts": ["geometry", "parallel lines", "alternate interior angles"],
    },
]


async def run_tests(system: MathNexusSystem, n_problems: int = 3):
    """Run the system on test problems."""
    print("=" * 60)
    print(f"  MathNexus Test Suite — {min(n_problems, len(TEST_PROBLEMS))} problems")
    print("=" * 60)

    results = []
    for i, tp in enumerate(TEST_PROBLEMS[:n_problems], 1):
        print(f"\n{'─'*60}")
        print(f"Test {i}/{min(n_problems, len(TEST_PROBLEMS))}: {tp['expected_concepts']}")
        result = await solve_single(tp["problem"], system)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("  Test Summary")
    print(f"{'='*60}")

    solved = sum(1 for r in results if r["solved"])
    total_score = sum(r.get("validation", {}).get("score", 0) for r in results)

    print(f"Problems solved: {solved}/{len(results)}")
    print(f"Average score: {total_score / len(results):.1f}/100" if results else "N/A")

    for i, (tp, r) in enumerate(zip(TEST_PROBLEMS[:n_problems], results), 1):
        status = "✅" if r["solved"] else "❌"
        print(f"  {status} Test {i}: {tp['problem'][:80]}...")
        print(f"     Type: {r['problem_type']}, Attempts: {r['attempt_count']}")

    # Memory stats
    stats = system.get_memory_stats()
    print(f"\nMemory: {stats['long_term']['total_entries']} entries in long-term memory")


# ── Main ──────────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(
        description="MathNexus — Multi-Agent Math Reasoning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Prove that √2 is irrational"
  python main.py -i
  python main.py --test
        """,
    )
    parser.add_argument(
        "problem", nargs="?", default=None,
        help="Math problem to solve (wrap in quotes)",
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run test suite",
    )
    parser.add_argument(
        "--test-all", action="store_true",
        help="Run all test problems",
    )
    parser.add_argument(
        "--model", default="deepseek-chat",
        help="DeepSeek model name (default: deepseek-chat)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Model temperature (default: 0.3)",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=3,
        help="Maximum solving attempts (default: 3)",
    )
    parser.add_argument(
        "--memory-dir", default="./math_nexus_memory",
        help="Long-term memory directory",
    )

    args = parser.parse_args()

    # Check API key
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
        print("Please set it with: export DEEPSEEK_API_KEY=your_key_here")
        sys.exit(1)

    # Initialize system
    print("Initializing MathNexus system...")
    system = MathNexusSystem(
        model_name=args.model,
        temperature=args.temperature,
        memory_dir=args.memory_dir,
    )
    print("Ready!\n")

    try:
        if args.test or args.test_all:
            n = len(TEST_PROBLEMS) if args.test_all else 3
            await run_tests(system, n_problems=n)
        elif args.interactive:
            await interactive_mode(system)
        elif args.problem:
            await solve_single(args.problem, system)
        else:
            # Default: run a demo test
            print("No problem provided. Running demo tests...")
            await run_tests(system, n_problems=2)
    finally:
        # Print memory stats
        stats = system.get_memory_stats()
        print(f"\n📊 Session stats: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
