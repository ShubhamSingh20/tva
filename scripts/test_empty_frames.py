"""
test_empty_frames.py

Takes a video chunk as input and produces a new video with the last frame
repeated (frozen) for an additional 2 seconds at the end.

Usage:
    python scripts/test_empty_frames.py <input_video> [output_video]

Examples:
    python scripts/test_empty_frames.py chunks/chunk_000.mp4
    python scripts/test_empty_frames.py chunks/chunk_000.mp4 output/padded.mp4
"""

import os
import subprocess
import sys


def probe_fps(video_path: str) -> float:
    """Use ffprobe to get the frame rate of the video."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        raw = result.stdout.strip()
        if "/" in raw:
            num, den = raw.split("/")
            return float(num) / float(den)
        return float(raw)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, ZeroDivisionError):
        return 30.0  # sensible default


def probe_duration(video_path: str) -> float | None:
    """Use ffprobe to get the duration of the video in seconds."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        val = result.stdout.strip()
        return float(val) if val else None
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def append_frozen_frames(
    input_path: str,
    output_path: str,
    pad_duration: float = 2.0,
) -> str:
    """Append frozen (last-frame-repeated) frames for `pad_duration` seconds.

    Uses the ffmpeg `tpad` video filter which clones the last frame for the
    requested duration.  Audio is padded with silence to match.

    Args:
        input_path:   Path to the source video.
        output_path:  Path for the padded output video.
        pad_duration: Seconds of frozen frames to append (default 2).

    Returns:
        The output path on success.
    """
    if not os.path.isfile(input_path):
        print(f"[ERROR] Input file not found: '{input_path}'")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # tpad=stop_mode=clone clones the last frame for stop_duration seconds.
    # apad + shortest ensures audio is also padded with silence to match.
    vf = f"tpad=stop_mode=clone:stop_duration={pad_duration}"
    af = f"apad=pad_dur={pad_duration}"

    print(f"[INFO] Appending {pad_duration}s of frozen last-frame to '{input_path}'")
    print(f"[INFO] Output -> '{output_path}'")

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-vf", vf,
                "-af", af,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-strict", "experimental",
                "-loglevel", "warning",
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"[ERROR] ffmpeg failed:\n{result.stderr}")
            sys.exit(1)
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Please install ffmpeg.")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("[ERROR] ffmpeg timed out.")
        sys.exit(1)

    # Report durations
    orig_dur = probe_duration(input_path)
    new_dur = probe_duration(output_path)
    fps = probe_fps(input_path)

    print(f"[OK] Done.")
    print(f"     Original duration : {orig_dur:.2f}s" if orig_dur else "     Original duration : unknown")
    print(f"     New duration      : {new_dur:.2f}s" if new_dur else "     New duration      : unknown")
    print(f"     FPS               : {fps:.2f}")
    print(f"     Frozen frames     : ~{int(pad_duration * fps)} frames ({pad_duration}s)")

    return output_path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_empty_frames.py <input_video> [output_video]")
        sys.exit(1)

    input_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # Default: same name with _padded suffix
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_padded{ext}"

    append_frozen_frames(input_path, output_path)


if __name__ == "__main__":
    main()
