"""Memory systems for MathNexus.

Implements two memory mechanisms:
1. Short-term memory: Conversation buffer and state tracking within a session
   (LangGraph's built-in state management + InMemorySaver checkpointer)
2. Long-term memory: ChromaDB vector store for persisting past problem-solution
   pairs across sessions, enabling retrieval-augmented solving.
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings


# ── Long-term Memory: ChromaDB-backed problem-solution store ─────────────────


class LongTermMemory:
    """Vector-store backed long-term memory for problem-solution pairs.

    Stores: problem text, solution, problem type, success status, key concepts,
    and metadata. Uses ChromaDB with sentence-transformer embeddings for
    semantic retrieval of similar past problems.
    """

    def __init__(self, persist_dir: str = "./math_nexus_memory"):
        """Initialize long-term memory with ChromaDB persistent storage.

        Args:
            persist_dir: Directory to persist the ChromaDB database.
        """
        self.persist_dir = os.path.abspath(persist_dir)
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create the collection for math problems
        self.collection = self.client.get_or_create_collection(
            name="math_problems",
            metadata={"description": "Math problem-solution pairs for retrieval"},
        )

    def store(
        self,
        problem: str,
        solution: str,
        problem_type: str,
        success: bool = True,
        key_concepts: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Store a problem-solution pair in long-term memory.

        Args:
            problem: The problem text.
            solution: The solution text.
            problem_type: Type of problem (computation, proof, choice, short_answer).
            success: Whether the solution was correct.
            key_concepts: List of key mathematical concepts involved.
            metadata: Additional metadata to store.
        """
        # Generate a unique ID from the problem
        problem_id = hashlib.md5(problem.encode()).hexdigest()[:16]

        # Build document text for embedding
        doc_text = f"Problem: {problem}\nType: {problem_type}\nSolution: {solution}"

        # Prepare metadata
        meta = {
            "problem": problem,
            "solution": solution,
            "problem_type": problem_type,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "key_concepts": json.dumps(key_concepts or []),
            **(metadata or {}),
        }

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=[problem_id],
            documents=[doc_text],
            metadatas=[meta],
        )

    def retrieve(
        self, query: str, n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve similar past problems from long-term memory.

        Args:
            query: The query text (usually the current problem).
            n_results: Number of similar problems to retrieve.

        Returns:
            List of retrieved memory entries with problem, solution, and metadata.
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count()),
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else None

                # Parse key_concepts back from JSON
                if "key_concepts" in meta and isinstance(meta["key_concepts"], str):
                    meta["key_concepts"] = json.loads(meta["key_concepts"])

                memories.append({
                    "id": doc_id,
                    "problem": meta.get("problem", ""),
                    "solution": meta.get("solution", ""),
                    "problem_type": meta.get("problem_type", ""),
                    "success": meta.get("success", False),
                    "key_concepts": meta.get("key_concepts", []),
                    "similarity_score": 1 - distance if distance else None,
                })

        return memories

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the long-term memory store."""
        count = self.collection.count()
        if count == 0:
            return {"total_entries": 0}

        all_meta = self.collection.get()
        if not all_meta["metadatas"]:
            return {"total_entries": count}

        successes = sum(
            1 for m in all_meta["metadatas"] if m.get("success", False)
        )
        types = {}
        for m in all_meta["metadatas"]:
            pt = m.get("problem_type", "unknown")
            types[pt] = types.get(pt, 0) + 1

        return {
            "total_entries": count,
            "successful_solutions": successes,
            "success_rate": f"{successes / count * 100:.1f}%" if count > 0 else "N/A",
            "by_type": types,
        }

    def clear(self):
        """Clear all entries from long-term memory."""
        self.client.delete_collection("math_problems")
        self.collection = self.client.get_or_create_collection(
            name="math_problems",
            metadata={"description": "Math problem-solution pairs for retrieval"},
        )


# ── Short-term Memory: Session-internal conversation buffer ──────────────────


class ShortTermMemory:
    """In-memory conversation buffer for the current solving session.

    Tracks recent interactions, intermediate reasoning steps, tool calls,
    and agent communications within a single problem-solving session.
    Uses LangGraph's state management for persistence within a session.
    """

    def __init__(self, max_buffer_size: int = 50):
        """Initialize short-term memory buffer.

        Args:
            max_buffer_size: Maximum number of messages to retain in buffer.
        """
        self.max_buffer_size = max_buffer_size
        self.buffer: List[Dict[str, Any]] = []
        self.current_problem: Optional[str] = None
        self.tool_call_log: List[Dict[str, Any]] = []
        self.agent_interactions: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str, agent: str = "system"):
        """Add a message to the short-term buffer.

        Args:
            role: Message role (user, assistant, system, tool).
            content: Message content.
            agent: Which agent produced this message.
        """
        entry = {
            "role": role,
            "content": content,
            "agent": agent,
            "timestamp": datetime.now().isoformat(),
        }
        self.buffer.append(entry)

        # Trim buffer if needed
        if len(self.buffer) > self.max_buffer_size:
            self.buffer = self.buffer[-self.max_buffer_size:]

    def add_tool_call(self, tool_name: str, input_str: str, output_str: str):
        """Log a tool invocation.

        Args:
            tool_name: Name of the tool that was called.
            input_str: Input provided to the tool.
            output_str: Output returned by the tool.
        """
        self.tool_call_log.append({
            "tool": tool_name,
            "input": input_str,
            "output": output_str,
            "timestamp": datetime.now().isoformat(),
        })

    def add_interaction(self, from_agent: str, to_agent: str, message: str):
        """Log an inter-agent communication.

        Args:
            from_agent: Source agent.
            to_agent: Destination agent.
            message: The message content.
        """
        self.agent_interactions.append({
            "from": from_agent,
            "to": to_agent,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })

    def get_recent_context(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the n most recent buffer entries.

        Args:
            n: Number of recent entries to return.

        Returns:
            List of recent buffer entries.
        """
        return self.buffer[-n:] if n > 0 else self.buffer

    def get_tool_history_summary(self) -> str:
        """Get a human-readable summary of tool usage."""
        if not self.tool_call_log:
            return "No tools used yet."
        lines = [f"Tool call history ({len(self.tool_call_log)} calls):"]
        for tc in self.tool_call_log:
            lines.append(f"  - {tc['tool']}: {tc['input'][:80]} → {tc['output'][:80]}")
        return "\n".join(lines)

    def clear(self):
        """Reset short-term memory for a new problem."""
        self.buffer.clear()
        self.tool_call_log.clear()
        self.agent_interactions.clear()
        self.current_problem = None
