import os
import json
from typing import List
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent

from .transcoding import MediaTranscoder
from .llm import GeminiLLM, OpenAILLM
from .models import AgentInput, Task, AnalyzeStrategy, LLMAnalysis
from .prompts import VIDEO_ANALYSIS_PROMPT, FRAME_ANALYSIS_PROMPT, COMPARISON_PROMPT, SUMMARY_PROMPT
from .utils import load_as_json

class VideoAnalysisAgent:
    def __init__(self, agent_input: AgentInput, analyze_strategy: AnalyzeStrategy = AnalyzeStrategy.VIDEO_ANALYSIS):
        self.agent_input = agent_input
        self.tasks = agent_input.tasks
        self.gemini_llm = GeminiLLM()
        self.openai_llm = OpenAILLM()
        self.media_transcoder = MediaTranscoder(agent_input.converted_video_path)
        self.analyze_strategy = analyze_strategy

    def clip_segments(self, add_padding: bool = True) -> List[str]:
        """Clip the source video into segments aligned with task timedeltas.
        Each chunk is written to ``inputs/{testname}/chunk_NNN/video.mp4``."""
        base_dir = os.path.dirname(self.agent_input.converted_video_path)
        timedeltas = [task.timedelta for task in self.tasks]

        # for video analysis, we add padding to the chunks to avoid the last frame being the action frame
        add_padding = self.analyze_strategy == AnalyzeStrategy.VIDEO_ANALYSIS
        return self.media_transcoder.clip_segments(
            timedeltas, base_dir, add_padding=add_padding,
        )

    def _analyze_video_chunk(self, task: Task, chunk_path: str) -> None:
        """
        Send a single chunk + its task action to the LLM and populate
        the task's llm_* fields in-place.
        """
        prompt = f"USER TASK: {task.action}"
        response = self.gemini_llm.generate_response(
            prompt, VIDEO_ANALYSIS_PROMPT, [chunk_path],
        )
        result = load_as_json(response)
        is_passed = result.get("is_passed", None)

        if is_passed is None:
            task.llm_is_passed = LLMAnalysis.UNCERTAIN
            task.llm_reasoning = "No is_passed value found in the response"
            return

        task.llm_is_passed = LLMAnalysis(is_passed.lower())
        task.llm_reasoning = result.get("reasoning", "")

    def _analyze_frame_chunk(self, task: Task, chunk_path: str) -> None:
        """Extract frames from a video chunk and send them to OpenAI
        for visual verification. Populates the task's llm_* fields in-place."""
        frame_paths = MediaTranscoder.extract_frames(chunk_path, deduplicate=True)

        if not frame_paths:
            task.llm_is_passed = LLMAnalysis.UNCERTAIN
            task.llm_reasoning = "No frames could be extracted from the chunk"
            return

        prompt = f"USER TASK: {task.action}"
        response = self.openai_llm.generate_response(
            prompt, FRAME_ANALYSIS_PROMPT, frame_paths,
        )
        result = load_as_json(response)
        is_passed = result.get("is_passed", None)

        if is_passed is None:
            task.llm_is_passed = LLMAnalysis.UNCERTAIN
            task.llm_reasoning = "No is_passed value found in the response"
            return

        task.llm_is_passed = LLMAnalysis(is_passed.lower())
        task.llm_reasoning = result.get("reasoning", "")

    def analyze_chunk(self, task: Task, chunk_path: str) -> None:
        """
        Send a single chunk + its task action to the LLM and populate
        the task's llm_* fields in-place.
        """

        if self.analyze_strategy == AnalyzeStrategy.VIDEO_ANALYSIS:
            self._analyze_video_chunk(task, chunk_path)

            if task.llm_is_passed == LLMAnalysis.UNCERTAIN:
                # fallback to frame analysis
                self._analyze_frame_chunk(task, chunk_path)

        elif self.analyze_strategy == AnalyzeStrategy.FRAME_ANALYSIS:
            self._analyze_frame_chunk(task, chunk_path)
        else:
            raise ValueError(f"Invalid analyze strategy: {self.analyze_strategy}")

    def compare_with_agent_response(self, task: Task) -> None:
        """Layer 2: Cross-reference our video/frame validation reasoning with
        the user agent's self-reported response. Populates final_is_passed and
        final_reasoning on the task."""
        if not task.user_agent_response:
            # No agent response to compare against; carry Layer 1 forward
            task.final_is_passed = task.llm_is_passed
            task.final_reasoning = task.llm_reasoning
            return

        prompt = (
            f"TASK: {task.action}\n\n"
            f"VIDEO VALIDATION REASONING: {task.llm_reasoning}\n\n"
            f"USER AGENT RESPONSE: {task.user_agent_response}"
        )
        response = self.openai_llm.generate_response(prompt, COMPARISON_PROMPT)
        result = load_as_json(response)
        is_passed = result.get("is_passed", None)

        if is_passed is None:
            task.final_is_passed = LLMAnalysis.UNCERTAIN
            task.final_reasoning = "No is_passed value found in comparison response"
            return

        task.final_is_passed = LLMAnalysis(is_passed.lower())
        task.final_reasoning = result.get("reasoning", "")

    def generate_summary(self) -> str:
        """Generate a 1-2 line summary of deviations from the validation report.
        Returns the summary string and stores it on self.summary."""
        report_lines = []
        for i, task in enumerate(self.tasks, 1):
            if task.terminate == "yes":
                continue
            verdict = task.final_is_passed.value if task.final_is_passed else "unknown"
            report_lines.append(
                f"  Task {i}: {task.action}\n"
                f"  Verdict: {verdict}\n"
                f"  Reasoning: {task.final_reasoning or 'N/A'}"
            )

        prompt = "VALIDATION REPORT:\n\n" + "\n\n".join(report_lines)
        response = self.openai_llm.generate_response(prompt, SUMMARY_PROMPT)
        self.summary = response
        return self.summary

    def _validation_dir(self) -> str:
        path = os.path.join(
            os.path.dirname(self.agent_input.converted_video_path), "validation",
        )
        os.makedirs(path, exist_ok=True)
        return path

    def _tasks_as_dicts(self) -> List[dict]:
        return [
            {
                "action": task.action,
                "user_agent_response": task.user_agent_response,
                "llm_is_passed": task.llm_is_passed.value if task.llm_is_passed else None,
                "llm_reasoning": task.llm_reasoning,
                "final_is_passed": task.final_is_passed.value if task.final_is_passed else None,
                "final_reasoning": task.final_reasoning,
            }
            for task in self.tasks
        ]

    def save_json(self) -> str:
        """Persist task results to ``responses.json``. Returns the output path."""
        output_path = os.path.join(self._validation_dir(), "responses.json")
        data = {
            "summary": getattr(self, "summary", None),
            "tasks": self._tasks_as_dicts(),
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        return output_path

    def save_xml(self) -> str:
        """Persist task results to ``responses.xml`` in JUnit-style format.
        Returns the output path."""
        output_path = os.path.join(self._validation_dir(), "responses.xml")

        testsuites = Element("testsuites")
        testsuite = SubElement(testsuites, "testsuite",
            name="validation",
            tests=str(len(self.tasks)),
        )

        # Summary element
        summary_text = getattr(self, "summary", None)
        if summary_text:
            summary_el = SubElement(testsuite, "system-out")
            summary_el.text = summary_text

        for i, task in enumerate(self.tasks):
            llm_passed = task.llm_is_passed.value if task.llm_is_passed else "unknown"
            final_passed = task.final_is_passed.value if task.final_is_passed else "unknown"

            testcase = SubElement(testsuite, "testcase",
                name=f"task_{i:03d}",
                classname="validation",
            )

            action_el = SubElement(testcase, "system-out")
            action_el.text = task.action

            props = SubElement(testcase, "properties")
            SubElement(props, "property", name="llm_is_passed", value=llm_passed)
            SubElement(props, "property", name="llm_reasoning", value=task.llm_reasoning or "")
            SubElement(props, "property", name="user_agent_response", value=task.user_agent_response or "")
            SubElement(props, "property", name="final_is_passed", value=final_passed)
            SubElement(props, "property", name="final_reasoning", value=task.final_reasoning or "")

            if final_passed == "no":
                failure = SubElement(testcase, "failure", message=task.final_reasoning or "")
                failure.text = task.action

        indent(testsuites, space="    ")
        tree = ElementTree(testsuites)
        tree.write(output_path, encoding="unicode", xml_declaration=True)
        return output_path
