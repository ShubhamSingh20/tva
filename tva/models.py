from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

class AnalyzeStrategy(Enum):
    VIDEO_ANALYSIS = "VIDEO_ANALYSIS"
    FRAME_ANALYSIS = "FRAME_ANALYSIS"

class LLMAnalysis(Enum):
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"

@dataclass
class Task:
    """A single step the agent was supposed to perform, enriched with LLM
    analysis after video review."""

    # --- From planner (populated at parse time) ---
    action: str              # was next_step
    plan: str
    terminate: str
    final_response: str
    target_helper: str
    assert_summary: str
    timedelta: int

    # --- Paired user/agent execution response ---
    user_agent_response: str | None = None

    # --- Layer 1: Video/frame analysis ---
    llm_is_passed: LLMAnalysis | None = None
    llm_reasoning: str | None = None

    # --- Layer 2: Comparison (our reasoning vs agent's claim) ---
    final_is_passed: LLMAnalysis | None = None
    final_reasoning: str | None = None

    def __repr__(self) -> str:
        fields = {k: v for k, v in self.__dict__.items() if k != "plan"}
        inner = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"Task({inner})"


@dataclass
class AgentInput:
    agent_inner_logs: Dict[str, Any]
    raw_video_path: str
    result_xml: Optional[Dict[str, Any]] = None
    converted_video_path: Optional[str] = None

    @property
    def tasks(self) -> List[Task]:
        logs = self.agent_inner_logs

        # Build an index of each log entry's position in the full list
        # so we can look ahead for the paired user response.
        assistant_entries: List[tuple[int, dict]] = [
            (i, log) for i, log in enumerate(logs) if log['role'] == 'assistant'
        ]
        assistant_entries.sort(key=lambda x: x[1]['server_timestamp'])

        if not assistant_entries:
            return []

        first_timestamp = assistant_entries[0][1]['server_timestamp']

        tasks: List[Task] = []
        for idx, (pos, log) in enumerate(assistant_entries):
            # The paired user response is the next entry in the full log list
            # (if it exists and is a user message).
            user_response: str | None = None
            next_pos = pos + 1
            if next_pos < len(logs) and logs[next_pos]['role'] == 'user':
                user_response = logs[next_pos]['content']

            tasks.append(Task(
                action=log['content']['next_step'],
                plan=log['content']['plan'],
                terminate=log['content']['terminate'],
                final_response=log['content']['final_response'],
                target_helper=log['content']['target_helper'],
                assert_summary=log['content'].get('assert_summary', ''),
                timedelta=log['server_timestamp'] - first_timestamp,
                user_agent_response=user_response,
            ))

        return tasks
