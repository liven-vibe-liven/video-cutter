"""
Video processing pipeline:
1. Detect scene changes via FFmpeg
2. Extract thumbnails for each scene
3. Export selected scenes concatenated
4. (Optional) AI analysis via Whisper + Claude
"""

import os
import subprocess
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
THUMBNAILS_DIR = Path("thumbnails")
THUMBNAILS_DIR.mkdir(exist_ok=True)


@dataclass
class JobStatus:
    job_id: str
    status: str       # detecting | ready | exporting | done | error
    step: str
    progress: int
    input_path: str
    output_path: str
    scenes: list = field(default_factory=list)
    error: Optional[str] = None


jobs: dict[str, JobStatus] = {}


def update_job(job_id: str, **kwargs):
    if job_id in jobs:
        for k, v in kwargs.items():
            setattr(jobs[job_id], k, v)


def get_video_duration(path: str) -> float:
    result = subprocess.run(
        [FFMPEG, "-i", path, "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    for line in result.stderr.splitlines():
        if "Duration:" in line:
            dur_str = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = dur_str.split(":")
            return float(h) * 3600 + float(m) * 60 + float(s)
    return 90.0


def detect_scenes(video_path: str, threshold: float = 0.3) -> list[dict]:
    """
    Detect scene/shot changes using FFmpeg select filter.
    Returns list of {index, start, end, duration} dicts.
    """
    duration = get_video_duration(video_path)

    # FFmpeg scene detection: select frames where scene score > threshold
    cmd = [
        FFMPEG, "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-",
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )

    # Parse pts_time from showinfo lines
    scene_times = []
    for line in result.stderr.splitlines():
        if "pts_time:" in line and "Parsed_showinfo" in line:
            try:
                pts_time = float(line.split("pts_time:")[1].split()[0])
                if pts_time > 0.2:  # skip very start
                    scene_times.append(pts_time)
            except (ValueError, IndexError):
                pass

    # Build scene boundary list
    boundaries = sorted({0.0} | set(scene_times))
    boundaries.append(duration)

    scenes = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        dur = end - start
        if dur >= 0.3:
            scenes.append({
                "index": len(scenes),
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(dur, 3),
            })

    # Fallback: if nothing detected, split into ~5s segments
    if len(scenes) <= 1:
        scenes = []
        step = max(3.0, duration / 20)
        t = 0.0
        while t < duration - 0.3:
            end = min(t + step, duration)
            scenes.append({
                "index": len(scenes),
                "start": round(t, 3),
                "end": round(end, 3),
                "duration": round(end - t, 3),
            })
            t = end

    return scenes


def extract_thumbnail(video_path: str, time: float, output_path: str):
    """Extract one frame at `time` seconds as a JPEG thumbnail."""
    subprocess.run(
        [
            FFMPEG, "-y",
            "-ss", str(max(0.0, time)),
            "-i", video_path,
            "-vframes", "1",
            "-vf", "scale=320:-2",
            "-q:v", "5",
            output_path,
        ],
        capture_output=True, encoding="utf-8", errors="replace",
    )


def cut_and_concat(input_path: str, scenes: list[dict], output_path: str):
    """
    Cut scenes from input and concatenate using temp files + concat demuxer.
    This approach reliably preserves audio across all formats.
    """
    import tempfile
    import shutil

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # Step 1: extract each scene to its own temp file (re-encode to ensure compatible streams)
        temp_files = []
        for i, s in enumerate(scenes):
            tmp_out = str(tmp_dir / f"seg_{i:04d}.mp4")
            cmd = [
                FFMPEG, "-y",
                "-ss", str(s["start"]),
                "-i", input_path,
                "-t", str(round(s["duration"], 3)),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                tmp_out,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                lines = [l for l in r.stderr.splitlines() if l.strip()]
                raise RuntimeError(f"Segment {i} failed:\n" + "\n".join(lines[-8:]))
            temp_files.append(tmp_out)

        # Step 2: write concat list
        list_path = str(tmp_dir / "list.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for tf in temp_files:
                # escape single quotes in path just in case
                safe = tf.replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        # Step 3: concat + re-encode to normalize timestamps across segments
        cmd = [
            FFMPEG, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            lines = [l for l in r.stderr.splitlines() if l.strip()]
            raise RuntimeError("Concat failed:\n" + "\n".join(lines[-8:]))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Async jobs ────────────────────────────────────────────────────────────────

async def process_upload(job_id: str, threshold: float = 0.3):
    """Detect scenes + extract thumbnails. Updates job when done."""
    job = jobs.get(job_id)
    if not job:
        return
    try:
        update_job(job_id, step="Detecting scene changes...", progress=10)
        scenes = detect_scenes(job.input_path, threshold)

        thumb_dir = THUMBNAILS_DIR / job_id
        thumb_dir.mkdir(exist_ok=True)

        total = len(scenes)
        for i, scene in enumerate(scenes):
            update_job(
                job_id,
                step=f"Extracting thumbnails ({i + 1}/{total})...",
                progress=10 + int(85 * (i + 1) / total),
            )
            t = scene["start"] + scene["duration"] / 2
            thumb_path = str(thumb_dir / f"{i:04d}.jpg")
            extract_thumbnail(job.input_path, t, thumb_path)
            scene["thumbnail"] = f"/thumbnail/{job_id}/{i}"

        # Try to transcribe audio (optional — skipped if faster-whisper not installed)
        update_job(job_id, step="Transcribing audio...", progress=96)
        try:
            import tempfile, shutil as _shutil
            _tmp = Path(tempfile.mkdtemp())
            _audio = str(_tmp / "audio.wav")
            _extract_audio_for_transcription(job.input_path, _audio)

            from faster_whisper import WhisperModel
            _model = WhisperModel("tiny", device="cpu", compute_type="int8")
            _segs, _ = _model.transcribe(_audio, word_timestamps=True)

            _texts: list[list[str]] = [[] for _ in scenes]
            for seg in _segs:
                if seg.words:
                    for word in seg.words:
                        mid = (word.start + word.end) / 2
                        for sc in scenes:
                            if sc["start"] <= mid < sc["end"]:
                                _texts[sc["index"]].append(word.word)
                                break
                else:
                    mid = (seg.start + seg.end) / 2
                    for sc in scenes:
                        if sc["start"] <= mid < sc["end"]:
                            _texts[sc["index"]].append(seg.text.strip())
                            break

            for sc in scenes:
                sc["transcript"] = " ".join(_texts[sc["index"]]).strip()

            _shutil.rmtree(str(_tmp), ignore_errors=True)
        except Exception:
            pass  # no transcript — UI just shows nothing

        update_job(job_id, status="ready", step="Ready", progress=100, scenes=scenes)

    except Exception as e:
        update_job(job_id, status="error", step="Error", error=str(e))


async def export_video(job_id: str, keep_indices: list[int]):
    """Export video keeping only selected scenes."""
    job = jobs.get(job_id)
    if not job:
        return
    try:
        update_job(job_id, status="exporting", step="Exporting video...", progress=20)
        selected = [s for s in job.scenes if s["index"] in keep_indices]
        if not selected:
            raise ValueError("No scenes selected")
        cut_and_concat(job.input_path, selected, job.output_path)
        update_job(job_id, status="done", step="Done", progress=100)
    except Exception as e:
        update_job(job_id, status="ready", step="Ready", error=str(e))


def _extract_audio_for_transcription(video_path: str, audio_path: str):
    """Extract audio as 16 kHz mono WAV for Whisper."""
    subprocess.run(
        [
            FFMPEG, "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-f", "wav",
            audio_path,
        ],
        capture_output=True, encoding="utf-8", errors="replace",
    )


async def analyze_scenes_with_ai(job_id: str) -> dict:
    """
    Transcribe audio with faster-whisper, then ask Claude which scenes to keep.
    Returns {"suggestions": [{index, keep, reason}]}.
    """
    import tempfile
    import shutil

    job = jobs.get(job_id)
    if not job:
        raise ValueError("Job not found")

    scenes = job.scenes
    if not scenes:
        raise ValueError("No scenes available")

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        # 1. Extract audio
        audio_path = str(tmp_dir / "audio.wav")
        await asyncio.get_event_loop().run_in_executor(
            None, _extract_audio_for_transcription, job.input_path, audio_path
        )

        # 2. Transcribe with faster-whisper
        transcript_by_scene: list[str] = [""] * len(scenes)
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(audio_path, word_timestamps=True)

            # Map words to scenes by timestamp
            scene_texts: list[list[str]] = [[] for _ in scenes]
            for seg in segments:
                mid = (seg.start + seg.end) / 2
                for sc in scenes:
                    if sc["start"] <= mid < sc["end"]:
                        scene_texts[sc["index"]].append(seg.text.strip())
                        break

            transcript_by_scene = [" ".join(t) for t in scene_texts]
        except Exception:
            # Whisper not available or failed — continue without transcription
            transcript_by_scene = [""] * len(scenes)

        # 3. Build scene summary for Claude
        scene_lines = []
        for sc in scenes:
            idx = sc["index"]
            dur = sc["duration"]
            txt = transcript_by_scene[idx] if idx < len(transcript_by_scene) else ""
            line = f"Scene {idx}: duration={dur:.1f}s"
            if txt:
                line += f', speech="{txt}"'
            scene_lines.append(line)

        scenes_text = "\n".join(scene_lines)

        # 4. Ask Claude
        import anthropic
        client = anthropic.Anthropic()

        prompt = (
            "You are a video editor. Below is a list of scenes from a short video.\n"
            "Decide which scenes to KEEP and which to REMOVE to make the video concise and engaging.\n"
            "Prefer removing: long silences, repeated content, off-topic tangents, awkward pauses.\n"
            "Prefer keeping: key information, interesting moments, clear speech.\n\n"
            f"Scenes:\n{scenes_text}\n\n"
            "Reply with a JSON array (and nothing else) like:\n"
            '[{"index": 0, "keep": true, "reason": "..."}, ...]\n'
            "Include every scene index."
        )

        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        import json
        suggestions = json.loads(raw)
        return {"suggestions": suggestions}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
