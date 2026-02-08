"""
TestZeus Validation Agent — entry-point.

Validates whether an automated browser agent actually performed the tasks it
claimed by cross-referencing video evidence with agent logs.

Usage:
    python -m tva <test_name>
"""
import os
import shutil
import sys
from dotenv import load_dotenv
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn
from rich.table import Table
from tva.models import AgentInput, Task, AnalyzeStrategy, LLMAnalysis
from tva.utils import load_json, load_xml_as_dict
from tva.transcoding import MediaTranscoder
from tva.agent import VideoAnalysisAgent
from tva.tui import console, step, step_error

load_dotenv()

INPUTS_DIR = "inputs"
AGENT_LOGS_FILE = "agent_inner_logs.json"
RESULTS_XML_FILE = "results.xml"
RAW_VIDEO_FILE = "test_video.webm"


def _require_path(path: str, label: str, step_n: int) -> str:
    """Exit with a clear error if *path* does not exist."""
    if not os.path.exists(path):
        step_error(step_n, f"{label} not found: '{path}'")
        sys.exit(1)
    return path


def _truncate(text: str, max_len: int) -> str:
    return text[:max_len - 3] + "..." if len(text) > max_len else text


def _verdict_display(analysis: LLMAnalysis | None) -> str:
    value = analysis.value if analysis else ""
    return {
        "yes": "[green]\u2713 yes[/]",
        "no": "[red]\u2717 no[/]",
        "uncertain": "[yellow]? uncertain[/]",
    }.get(value, "\u2014")


def load_test_inputs(test_name: str) -> AgentInput:
    """Load and validate all artefacts for *test_name* from ``inputs/<test_name>/``."""
    test_dir = _require_path(
        os.path.join(INPUTS_DIR, test_name), "Test directory", step_n=0,
    )

    # 1. Agent inner logs (required)
    logs_path = _require_path(
        os.path.join(test_dir, AGENT_LOGS_FILE), AGENT_LOGS_FILE, step_n=1,
    )
    with console.status(f"[bold cyan]Loading {AGENT_LOGS_FILE}..."):
        agent_inner_logs = load_json(logs_path)["planner_agent"]
    step(1, f"Loaded {AGENT_LOGS_FILE} ({len(agent_inner_logs)} planner entries)")

    # 2. Results XML (optional — may be gitignored)
    xml_path = os.path.join(test_dir, RESULTS_XML_FILE)
    result_xml = None
    if os.path.isfile(xml_path):
        with console.status(f"[bold cyan]Loading {RESULTS_XML_FILE}..."):
            result_xml = load_xml_as_dict(xml_path)
        step(2, f"Loaded {RESULTS_XML_FILE}")
    else:
        step(2, f"{RESULTS_XML_FILE} not present — skipping")

    # 3. Raw video (required) — validate & convert
    raw_video_path = _require_path(
        os.path.join(test_dir, RAW_VIDEO_FILE), RAW_VIDEO_FILE, step_n=3,
    )
    transcoder = MediaTranscoder(raw_video_path)

    with console.status(f"[bold cyan]Validating {RAW_VIDEO_FILE}..."):
        if not transcoder.is_valid_webm():
            step_error(3, f"'{raw_video_path}' is not a valid webm file.")
            sys.exit(1)
    step(3, f"Validated {RAW_VIDEO_FILE}")

    with console.status("[bold cyan]Converting webm → mp4..."):
        converted_video_path = transcoder.to_mp4()
    step(4, "Converted webm → mp4")

    return AgentInput(
        agent_inner_logs=agent_inner_logs,
        raw_video_path=raw_video_path,
        result_xml=result_xml,
        converted_video_path=converted_video_path,
    )


def print_results_table(tasks: list[Task]) -> None:
    """Print the Analysis Results table showing task actions and LLM verdicts."""
    results_table = Table(title="Analysis Results", show_lines=True)
    results_table.add_column("#", style="dim", width=4)
    results_table.add_column("Task")
    results_table.add_column("L1 Video", width=12)
    results_table.add_column("L1 Reasoning")
    results_table.add_column("L2 Final", width=12)
    results_table.add_column("L2 Reasoning")

    for i, task in enumerate(tasks, 1):
        if task.terminate == "yes":
            continue

        results_table.add_row(
            str(i),
            task.action,
            _verdict_display(task.llm_is_passed),
            task.llm_reasoning or "",
            _verdict_display(task.final_is_passed),
            task.final_reasoning or "",
        )

    console.print()
    console.print(results_table)

def run_validation(agent: VideoAnalysisAgent) -> list[Task]:
    """Orchestrate the agent pipeline with Rich progress / step output."""

    with console.status("[bold cyan]Clipping video into segments..."):
        chunks = agent.clip_segments(add_padding=True)
    step(5, f"Clipped video into {len(chunks)} padded segments")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Step 6[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.description}"),
        console=console,
    ) as progress:
        prog_task = progress.add_task("Analyzing...", total=len(chunks))
        for task, chunk_path in zip(agent.tasks, chunks):
            progress.update(prog_task, description=_truncate(task.action, 60))
            agent.analyze_chunk(task, chunk_path)
            progress.advance(prog_task)

    step(6, f"Analyzed {len(chunks)} chunks")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Step 7[/]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.description}"),
        console=console,
    ) as progress:
        prog_task = progress.add_task("Comparing...", total=len(agent.tasks))
        for task in agent.tasks:
            progress.update(prog_task, description=_truncate(task.action, 60))
            agent.compare_with_agent_response(task)
            progress.advance(prog_task)

    step(7, "Compared validation results with agent responses")

    print_results_table(agent.tasks)
    console.print()

    with console.status("[bold cyan]Generating summary..."):
        summary = agent.generate_summary()
    step(8, "Generated deviation summary")

    has_failures = any(
        t.final_is_passed == LLMAnalysis.NO
        for t in agent.tasks if t.terminate != "yes"
    )
    border = "red" if has_failures else "green"
    title = "Deviations Found" if has_failures else "Validation Summary"

    console.print()
    console.print(Panel(summary, title=f"[bold]{title}[/]", border_style=border, expand=False))
    console.print()

    json_path = agent.save_json()
    xml_path = agent.save_xml()
    step(9, f"Saved results to {json_path} and {xml_path}")

    return agent.tasks



def _check_ffmpeg() -> None:
    """Ensure ffmpeg and ffprobe are available on PATH."""
    for cmd in ("ffmpeg", "ffprobe"):
        if shutil.which(cmd) is None:
            console.print(
                f"[bold red]Error:[/] [bold]{cmd}[/] not found on PATH.\n"
                f"  Install ffmpeg: https://ffmpeg.org/download.html"
            )
            sys.exit(1)


def main():
    _check_ffmpeg()

    if len(sys.argv) < 2:
        console.print("Usage: python -m tva <test_name>")
        available = ", ".join(os.listdir(INPUTS_DIR)) if os.path.isdir(INPUTS_DIR) else "none"
        console.print(f"  Available tests: {available}")
        sys.exit(1)

    test_name = sys.argv[1]

    console.print()
    console.print(Panel(
        f"[bold]Video Validation Agent[/]\n[dim]Test: {test_name}[/]",
        expand=False,
    ))
    console.print()

    agent_input = load_test_inputs(test_name)

    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_column(style="bold")
    info.add_column()
    info.add_row("Planner entries", str(len(agent_input.agent_inner_logs)))
    info.add_row("Raw video", agent_input.raw_video_path)
    info.add_row("Converted video", agent_input.converted_video_path)
    console.print(info)
    console.print()

    agent = VideoAnalysisAgent(
        agent_input, 
        analyze_strategy=AnalyzeStrategy.VIDEO_ANALYSIS
    )
    run_validation(agent)


if __name__ == "__main__":
    main()
