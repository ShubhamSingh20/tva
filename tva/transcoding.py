import os
import subprocess
import sys
from typing import List

import imagehash
from PIL import Image

from .tui import console


class MediaTranscoder:
    """Handles validation and transcoding of media files using ffmpeg/ffprobe."""

    VALID_WEBM_FORMATS = ("webm", "matroska")

    def __init__(self, source_path: str) -> None:
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Media file not found: '{source_path}'")
        self.source_path = source_path

    def is_valid_webm(self) -> bool:
        """Check whether the source file is a valid webm/matroska container."""
        format_name = self._probe_format()
        if format_name is None:
            return False
        return any(fmt in format_name for fmt in self.VALID_WEBM_FORMATS)

    def to_mp4(self, output_path: str | None = None, skip_if_exists: bool = True) -> str:
        """Convert the source webm to mp4.

        Args:
            output_path: Destination path for the mp4. Defaults to the source
            path with a .mp4 extension.
            skip_if_exists: If True and the output file already exists, skip
                            the conversion and return the existing path.

        Returns:
            The absolute path to the resulting mp4 file.
        """
        if output_path is None:
            output_path = os.path.splitext(self.source_path)[0] + ".mp4"

        if skip_if_exists and os.path.isfile(output_path):
            console.print(f"  [dim]MP4 already exists, skipping conversion[/]")
            return output_path

        return self._convert(self.source_path, output_path)

    def clip_segments(
        self,
        timedeltas: List[int],
        base_dir: str,
        add_padding: bool = False,
    ) -> List[str]:
        """Clip the source video into segments defined by consecutive timedeltas.

        Each chunk is written to its own directory::

            base_dir/chunk_000/video.mp4   ->  0s to 13s
            base_dir/chunk_001/video.mp4   -> 13s to 24s
            base_dir/chunk_002/video.mp4   -> 24s to 38s
            base_dir/chunk_003/video.mp4   -> 38s to end

        When ``add_padding`` is True each chunk video is padded with 2 s of
        frozen last-frame (the padded result replaces the original ``video.mp4``).

        Args:
            timedeltas: Sorted list of second offsets (starting from 0).
            base_dir:   Parent directory for chunk folders
                        (e.g. ``inputs/{testname}``).
            add_padding: If True, pad each chunk and replace original video.

        Returns:
            Ordered list of chunk video paths (``…/chunk_NNN/video.mp4``).
        """
        if not timedeltas:
            return []

        video_duration = self._probe_duration()

        # Build (start, end) pairs; last segment runs to end of video
        segments: List[tuple[int, float | int]] = []
        for i, start in enumerate(timedeltas):
            end = timedeltas[i + 1] if i + 1 < len(timedeltas) else video_duration
            if end is not None and start < end:
                segments.append((start, end))

        # Build per-chunk directory and video paths
        chunk_dirs = [
            os.path.join(base_dir, f"chunk_{i:03d}")
            for i in range(len(segments))
        ]
        chunk_paths = [
            os.path.join(d, "video.mp4") for d in chunk_dirs
        ]

        # Check cache
        if all(os.path.isfile(p) for p in chunk_paths):
            console.print(f"  [dim]All {len(chunk_paths)} chunks already exist, skipping[/]")
            return chunk_paths

        for chunk_dir in chunk_dirs:
            os.makedirs(chunk_dir, exist_ok=True)

        for idx, ((start, end), out_path) in enumerate(zip(segments, chunk_paths)):
            duration = end - start
            try:
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss", str(start),
                        "-i", self.source_path,
                        "-t", str(duration),
                        "-c:v", "libx264",
                        "-c:a", "aac",
                        "-strict", "experimental",
                        "-loglevel", "warning",
                        out_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    console.print(f"[bold red]ffmpeg clip failed for chunk {idx}:[/]\n{result.stderr}")
                    sys.exit(1)
            except FileNotFoundError:
                console.print("[bold red]ffmpeg not found. Please install ffmpeg.[/]")
                sys.exit(1)
            except subprocess.TimeoutExpired:
                console.print(f"[bold red]ffmpeg timed out clipping chunk {idx}.[/]")
                sys.exit(1)

        if add_padding:
            for chunk_path in chunk_paths:
                self._pad_chunk_inplace(chunk_path, pad_duration=2.0)

        return chunk_paths

    @staticmethod
    def extract_frames(
        chunk_path: str,
        deduplicate: bool = True,
        hash_threshold: int = 5,
    ) -> List[str]:
        """Extract 1-fps frames from a video chunk.

        Frames are written to a ``frames/`` subdirectory inside the chunk
        folder (e.g. ``inputs/test/chunk_000/frames/000_frame.jpeg``).

        When *deduplicate* is True, perceptual *dhash* is used to compare each
        frame to the previous one; frames whose hamming distance is below
        *hash_threshold* are discarded.  Set *deduplicate* to False to keep
        every extracted frame.

        Args:
            chunk_path:     Path to the chunk video
                            (e.g. ``…/chunk_000/video.mp4``).
            deduplicate:    If True, drop visually similar consecutive frames.
            hash_threshold: Maximum hamming distance to consider two consecutive
                            frames identical (default 5, ignored when
                            *deduplicate* is False).

        Returns:
            Sorted list of paths to the frame images.
        """
        chunk_dir = os.path.dirname(chunk_path)
        frames_dir = os.path.join(chunk_dir, "frames")

        # --- Check cache: if frames dir already has images, return them ---
        if os.path.isdir(frames_dir):
            existing = sorted(
                os.path.join(frames_dir, f)
                for f in os.listdir(frames_dir)
                if f.endswith(".jpeg")
            )
            if existing:
                console.print(f"  [dim]{len(existing)} cached frames in {frames_dir}[/]")
                return existing

        os.makedirs(frames_dir, exist_ok=True)

        raw_pattern = os.path.join(frames_dir, "raw_%04d.jpeg")
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", chunk_path,
                    "-vf", "fps=1",
                    "-loglevel", "warning",
                    raw_pattern,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                console.print(
                    f"[bold red]ffmpeg frame extraction failed for '{chunk_path}':[/]\n{result.stderr}"
                )
                sys.exit(1)
        except FileNotFoundError:
            console.print("[bold red]ffmpeg not found. Please install ffmpeg.[/]")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            console.print(f"[bold red]ffmpeg timed out extracting frames from '{chunk_path}'.[/]")
            sys.exit(1)

        raw_frames = sorted(
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.startswith("raw_") and f.endswith(".jpeg")
        )

        kept_paths: List[str] = []
        prev_hash: imagehash.ImageHash | None = None

        for raw_path in raw_frames:
            if deduplicate:
                img = Image.open(raw_path)
                current_hash = imagehash.dhash(img)

                if prev_hash is not None and (current_hash - prev_hash) < hash_threshold:
                    os.remove(raw_path)
                    continue

                prev_hash = current_hash

            # Rename raw_XXXX.jpeg → NNN_frame.jpeg
            kept_name = f"{len(kept_paths):03d}_frame.jpeg"
            kept_path = os.path.join(frames_dir, kept_name)
            os.rename(raw_path, kept_path)
            kept_paths.append(kept_path)

        # Clean up any leftover raw_ files (shouldn't be any, but just in case)
        for f in os.listdir(frames_dir):
            if f.startswith("raw_"):
                os.remove(os.path.join(frames_dir, f))

        label = f"kept {len(kept_paths)} unique" if deduplicate else f"kept all {len(kept_paths)}"
        console.print(
            f"  [dim]Extracted {len(raw_frames)} frames, {label}[/]"
        )
        return kept_paths

    def _probe_duration(self) -> float | None:
        """Use ffprobe to get the duration of the source file in seconds."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    self.source_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            val = result.stdout.strip()
            return float(val) if val else None
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            return None

    def _probe_format(self) -> str | None:
        """Use ffprobe to detect the container format of the source file."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=format_name",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    self.source_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout.strip().lower() or None
        except FileNotFoundError:
            console.print("[bold red]ffprobe not found. Please install ffmpeg.[/]")
            return None
        except subprocess.TimeoutExpired:
            console.print("[bold red]ffprobe timed out while probing the media.[/]")
            return None

    @staticmethod
    def _pad_chunk_inplace(chunk_path: str, pad_duration: float = 2.0) -> None:
        """Pad a video in-place by repeating its last frame for *pad_duration* seconds.

        Writes to a temporary file, then replaces the original ``video.mp4``.

        Args:
            chunk_path:   Path to the chunk video file (``…/chunk_NNN/video.mp4``).
            pad_duration: Seconds of frozen last-frame to append.
        """
        tmp_path = chunk_path + ".tmp.mp4"

        vf = f"tpad=stop_mode=clone:stop_duration={pad_duration}"
        af = f"apad=pad_dur={pad_duration}"

        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", chunk_path,
                    "-vf", vf,
                    "-af", af,
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-strict", "experimental",
                    "-loglevel", "warning",
                    tmp_path,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                console.print(f"[bold red]ffmpeg pad failed for '{chunk_path}':[/]\n{result.stderr}")
                sys.exit(1)
        except FileNotFoundError:
            console.print("[bold red]ffmpeg not found. Please install ffmpeg.[/]")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            console.print(f"[bold red]ffmpeg timed out padding '{chunk_path}'.[/]")
            sys.exit(1)

        os.replace(tmp_path, chunk_path)

    @staticmethod
    def _convert(webm_path: str, mp4_path: str) -> str:
        """Run ffmpeg to transcode webm -> mp4. Returns the output path."""
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", webm_path,
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-strict", "experimental",
                    "-loglevel", "warning",
                    mp4_path,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                console.print(f"[bold red]ffmpeg failed:[/]\n{result.stderr}")
                sys.exit(1)
            return mp4_path
        except FileNotFoundError:
            console.print("[bold red]ffmpeg not found. Please install ffmpeg.[/]")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            console.print("[bold red]ffmpeg timed out during conversion.[/]")
            sys.exit(1)
