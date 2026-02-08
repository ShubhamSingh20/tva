# TestZeus Validation Agent (TVA)

Validates whether automated browser agents actually performed the tasks they
claimed to perform, by cross-referencing **video evidence** against
**agent logs** using LLM-based analysis.

```
                          inputs/{testname}/
                         +-------------------+
                         | agent_inner_logs   |
                         | test_video.webm    |
                         | results.xml        |
                         +--------+----------+
                                  |
                     +------------v-------------+
                     |   Convert webm -> mp4    |
                     |   Clip into per-task     |
                     |   video chunks           |
                     +------------+-------------+
                                  |
              +-------------------+-------------------+
              |                                       |
   +----------v-----------+              +------------v-----------+
   |  Layer 1a: Video     |              |  Layer 1b: Frame       |
   |  Analysis (Gemini)   |              |  Analysis (OpenAI)     |
   |                      |              |                        |
   |  Send full chunk to  |  fallback    |  Extract 1-fps frames, |
   |  Gemini for verdict  +------------->|  deduplicate via dhash,|
   |                      |  if uncertain|  send to GPT for       |
   +----------+-----------+              |  verdict               |
              |                          +------------+-----------+
              +-------------------+-------------------+
                                  |
                        llm_is_passed / llm_reasoning
                                  |
                     +------------v-------------+
                     |  Layer 2: Comparison     |
                     |                          |
                     |  Cross-reference L1      |
                     |  reasoning against the   |
                     |  agent's self-reported   |
                     |  user_agent_response     |
                     +------------+-------------+
                                  |
                      final_is_passed / final_reasoning
                                  |
                     +------------v-------------+
                     |  Summary Generation      |
                     |                          |
                     |  1-2 line deviation      |
                     |  summary from full       |
                     |  validation report       |
                     +------------+-------------+
                                  |
                     +------------v-------------+
                     |  Output                  |
                     |  - Terminal table + panel |
                     |  - responses.json        |
                     |  - responses.xml (JUnit) |
                     +--------------------------+
```

## Why?

Browser automation agents (like TestZeus Hercules) report what they did via
logs. But agents can be wrong -- they might claim they applied a filter when the
page didn't actually change, or report a successful click that never happened.

TVA acts as an independent auditor: it watches the actual screen recording and
decides whether each task was truly performed, then compares its findings against
the agent's own claims.

## Quick start

```bash
# 1. Clone and install
git clone <repo-url> && cd testzeus-validation-agent
pip install -r requirements.txt

# 2. Set up API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and GEMINI_API_KEY

# 3. Place test inputs
#    inputs/<test_name>/agent_inner_logs.json
#    inputs/<test_name>/test_video.webm

# 4. Run
python -m tva <test_name>
```

## Prerequisites

- **Python 3.11+**
- **ffmpeg / ffprobe** installed and on PATH
- **OpenAI API key** (for frame analysis + comparison + summary)
- **Gemini API key** (for video analysis)

## Input structure

Each test lives in its own directory under `inputs/`:

```
inputs/search_sweater/
    agent_inner_logs.json    # Planner agent conversation logs
    test_video.webm          # Screen recording of the session
    results.xml              # Original test results (optional)
```

### agent_inner_logs.json

Contains the planner agent's conversation log -- alternating `assistant`
(planner directives) and `user` (execution results) messages. TVA pairs them
so each task has both the directive and the agent's self-reported outcome.

## Output structure

Results are written to `inputs/<test_name>/validation/`:

```
inputs/search_sweater/
    validation/
        responses.json       # Full report with summary + per-task results
        responses.xml        # JUnit-style XML (CI/CD compatible)
    chunk_000/
        video.mp4            # Clipped video segment
        frames/
            000_frame.jpeg   # Deduplicated frame stills
            001_frame.jpeg
    chunk_001/
        ...
```

## How it works

### Step 1-4: Load & prepare

Load agent logs and video, validate the webm, convert to mp4.

### Step 5: Clip video into segments

Using timestamps from the agent logs, the video is split into per-task chunks.
Each chunk lands in `chunk_NNN/video.mp4`.

### Step 6: Layer 1 -- Video / frame analysis

Each chunk is analyzed independently by an LLM to determine if the task was
performed. Two strategies are available:

| Strategy | LLM | Input | Best for |
|---|---|---|---|
| `VIDEO_ANALYSIS` | Gemini | Full video chunk | Fast, cheap, good for clear actions |
| `FRAME_ANALYSIS` | OpenAI GPT | Deduplicated 1-fps frame stills | Detailed UI inspection |

With `VIDEO_ANALYSIS`, if the result is `uncertain`, it automatically falls
back to `FRAME_ANALYSIS`.

**Frame deduplication**: Frames are extracted at 1 fps, then consecutive frames
with a perceptual hash (dhash) distance below the threshold are discarded. This
cuts token costs by removing visually identical frames (static screens, loading
states).

### Step 7: Layer 2 -- Comparison

A decoupled comparison layer cross-references:
- **L1 reasoning** (what the video showed)
- **user_agent_response** (what the agent claimed it did)

This catches the most important class of bugs: **the agent lying or being wrong
about its own actions**. For example, the agent might say "filter applied
successfully" while the video shows unchanged search results.

### Step 8: Summary

The full report is fed to the LLM to generate a concise 1-2 line summary of
any deviations found. This is printed in a coloured panel on the terminal
(green = all passed, red = deviations found).

### Step 9: Save

Results are persisted as JSON and JUnit-style XML.

## Architecture

```
tva/
    __init__.py       # Public API exports
    __main__.py       # CLI entry point + pipeline orchestration
    agent.py          # VideoAnalysisAgent -- core analysis + comparison
    llm.py            # LLM abstractions (OpenAI, Gemini)
    models.py         # Task, AgentInput, enums
    prompts.py        # All LLM prompt templates
    transcoding.py    # MediaTranscoder -- ffmpeg operations
    tui.py            # Rich console helpers
    utils.py          # JSON/XML loaders
```

## Configuration

| Env variable | Required | Used by |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Frame analysis, comparison, summary |
| `GEMINI_API_KEY` | Yes | Video analysis |

## Analysis strategies

Set the strategy when creating the agent:

```python
from tva import VideoAnalysisAgent, AnalyzeStrategy

agent = VideoAnalysisAgent(
    agent_input,
    analyze_strategy=AnalyzeStrategy.FRAME_ANALYSIS,
)
```

- `VIDEO_ANALYSIS` -- sends full video chunks to Gemini (with fallback to frames if uncertain)
- `FRAME_ANALYSIS` -- extracts frames and sends to OpenAI directly
