"""tva - TestZeus Validation Agent core package."""

from .models import AgentInput, Task, AnalyzeStrategy, LLMAnalysis
from .agent import VideoAnalysisAgent
from .llm import LLM, GeminiLLM, OpenAILLM
from .transcoding import MediaTranscoder
from .prompts import (
    VIDEO_ANALYSIS_PROMPT,
    FRAME_ANALYSIS_PROMPT,
    COMPARISON_PROMPT,
    SUMMARY_PROMPT,
)
from .tui import console, step, step_error
from .utils import load_json, load_xml_as_dict, load_as_json

__all__ = [
    # Models
    "AgentInput",
    "Task",
    "AnalyzeStrategy",
    "LLMAnalysis",
    # Agent
    "VideoAnalysisAgent",
    # LLM
    "LLM",
    "GeminiLLM",
    "OpenAILLM",
    # Media
    "MediaTranscoder",
    # Prompts
    "VIDEO_ANALYSIS_PROMPT",
    "FRAME_ANALYSIS_PROMPT",
    "COMPARISON_PROMPT",
    "SUMMARY_PROMPT",
    # TUI
    "console",
    "step",
    "step_error",
    # Utils
    "load_json",
    "load_xml_as_dict",
    "load_as_json",
]
