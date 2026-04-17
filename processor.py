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

import shutil as _shutil_sys
import imageio_ffmpeg

# Prefer system ffmpeg (Railway installs it via nixpacks) — it runs without
# the memory constraints of the static imageio-ffmpeg binary.
_system_ffmpeg = _shutil_sys.which("ffmpeg")
FFMPEG = _system_ffmpeg if _system_ffmpeg else imageio_ffmpeg.get_ffmpeg_exe()
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


def _get_video_resolution(path: str):
    """Return (width, height) of the first video stream, or None."""
    import re
    r = subprocess.run(
        [FFMPEG, "-i", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    for line in r.stderr.splitlines():
        if "Video:" in line:
            m = re.search(r"(\d{3,5})x(\d{3,5})", line)
            if m:
                return int(m.group(1)), int(m.group(2))
    return None


def _has_audio(path: str) -> bool:
    """Return True if the file has at least one audio stream."""
    r = subprocess.run(
        [FFMPEG, "-i", path],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return any("Audio:" in line for line in r.stderr.splitlines())


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
    Stream-copy each scene to a temp file, then concat.
    libx264 encoding fails on Railway (OOM/signal); stream copy is stable.
    """
    import tempfile, shutil

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        temp_files = []
        for i, sc in enumerate(scenes):
            tmp_out = str(tmp_dir / f"seg_{i:04d}.mp4")
            dur = round(sc["duration"], 3)
            # Re-encode for precise frame-accurate cuts
            cmd = [
                FFMPEG, "-y",
                "-ss", str(sc["start"]),
                "-i", input_path,
                "-t", str(dur),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-threads", "1",
                "-c:a", "aac", "-b:a", "128k",
                tmp_out,
            ]
            r = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            # If re-encoding fails, fall back to stream copy (less precise but works)
            if r.returncode != 0 or not Path(tmp_out).exists() or Path(tmp_out).stat().st_size < 1000:
                cmd = [
                    FFMPEG, "-y",
                    "-ss", str(sc["start"]),
                    "-i", input_path,
                    "-t", str(dur),
                    "-c", "copy",
                    "-avoid_negative_ts", "make_zero",
                    tmp_out,
                ]
                r2 = subprocess.run(
                    cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                )
                if r2.returncode != 0:
                    lines = [l for l in r2.stderr.splitlines() if l.strip()]
                    raise RuntimeError(f"Segment {i} failed:\n" + "\n".join(lines[-5:]))
            temp_files.append(tmp_out)

        list_path = str(tmp_dir / "list.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for tf in temp_files:
                f.write(f"file '{tf}'\n")

        cmd = [
            FFMPEG, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
        r = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if r.returncode != 0:
            lines = [l for l in r.stderr.splitlines() if l.strip()]
            raise RuntimeError("Concat failed:\n" + "\n".join(lines[-5:]))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Async jobs ────────────────────────────────────────────────────────────────

def _split_by_punctuation(
    scenes: list[dict], all_words: list[tuple]
) -> list[dict]:
    """
    Split scenes at sentence boundaries (., , ! ?).
    Each resulting sub-scene is at least MIN_DUR seconds.
    """
    MIN_DUR = 3.0
    new_scenes: list[dict] = []

    for sc in scenes:
        # words whose midpoint falls inside this scene
        sc_words = [
            (ws, we, wt) for ws, we, wt in all_words
            if sc["start"] <= (ws + we) / 2 < sc["end"]
        ]

        # candidate split points: end-time of words with punctuation
        candidates: list[float] = []
        for ws, we, wt in sc_words:
            if wt.strip().endswith((".", ",", "!", "?")):
                candidates.append(we)

        # keep only splits that leave at least MIN_DUR on each side
        valid_splits: list[float] = []
        last = sc["start"]
        for c in candidates:
            if c - last >= MIN_DUR and sc["end"] - c >= MIN_DUR:
                valid_splits.append(c)
                last = c

        if not valid_splits:
            new_scenes.append(sc)
            continue

        boundaries = [sc["start"]] + valid_splits + [sc["end"]]
        for i in range(len(boundaries) - 1):
            new_scenes.append({
                "index": len(new_scenes),
                "start": round(boundaries[i], 3),
                "end": round(boundaries[i + 1], 3),
                "duration": round(boundaries[i + 1] - boundaries[i], 3),
            })

    for i, sc in enumerate(new_scenes):
        sc["index"] = i

    return new_scenes


async def process_upload(job_id: str, threshold: float = 0.3):
    """Detect scenes, transcribe, split by punctuation, extract thumbnails."""
    job = jobs.get(job_id)
    if not job:
        return
    try:
        # 1. Visual scene detection
        update_job(job_id, step="Detecting scene changes...", progress=10)
        scenes = detect_scenes(job.input_path, threshold)

        # 2. Transcribe audio (optional)
        update_job(job_id, step="Transcribing audio...", progress=20)
        all_words: list[tuple] = []  # (start, end, word_text)
        try:
            import tempfile, shutil as _shutil
            _tmp = Path(tempfile.mkdtemp())
            _audio = str(_tmp / "audio.wav")
            _extract_audio_for_transcription(job.input_path, _audio)

            from faster_whisper import WhisperModel
            _model = WhisperModel("tiny", device="cpu", compute_type="int8")
            _segs, _ = _model.transcribe(_audio, word_timestamps=True)

            for seg in _segs:
                if seg.words:
                    for word in seg.words:
                        all_words.append((word.start, word.end, word.word))
                else:
                    all_words.append((seg.start, seg.end, seg.text.strip()))

            _shutil.rmtree(str(_tmp), ignore_errors=True)
        except Exception:
            pass  # transcription optional

        # 3. Split scenes at punctuation boundaries
        if all_words:
            update_job(job_id, step="Splitting at sentence boundaries...", progress=35)
            scenes = _split_by_punctuation(scenes, all_words)

        # 4. Assign transcript text to each final scene
        if all_words:
            texts: list[list[str]] = [[] for _ in scenes]
            for ws, we, wt in all_words:
                mid = (ws + we) / 2
                for sc in scenes:
                    if sc["start"] <= mid < sc["end"]:
                        texts[sc["index"]].append(wt)
                        break
            for sc in scenes:
                sc["transcript"] = " ".join(texts[sc["index"]]).strip()

        # 5. Extract thumbnails for final scene list
        thumb_dir = THUMBNAILS_DIR / job_id
        thumb_dir.mkdir(exist_ok=True)
        total = len(scenes)
        for i, scene in enumerate(scenes):
            update_job(
                job_id,
                step=f"Extracting thumbnails ({i + 1}/{total})...",
                progress=40 + int(55 * (i + 1) / total),
            )
            t = scene["start"] + scene["duration"] / 2
            thumb_path = str(thumb_dir / f"{i:04d}.jpg")
            extract_thumbnail(job.input_path, t, thumb_path)
            scene["thumbnail"] = f"/thumbnail/{job_id}/{i}"

        update_job(job_id, status="ready", step="Ready", progress=100, scenes=scenes)

    except Exception as e:
        update_job(job_id, status="error", step="Error", error=str(e))


async def export_mix_video(timeline: list[dict], output_path: str, mix_job_id: str):
    """Export a mix of scenes from multiple source videos.

    KEY CONSTRAINT: every segment MUST be encoded with THE SAME codec.
    If segment A uses libx264 and segment B uses mpeg4, stream-copy concat
    produces a broken file (video freezes, audio from broken segment plays
    quietly in background). So we probe once, pick one encoder, use it for all.
    """
    import tempfile, shutil as _shutil
    try:
        update_job(mix_job_id, step="Preparing...", progress=5)
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            # ── Step 1: Probe to choose encoder (once for all segments) ──────
            first_item = timeline[0]
            first_source = jobs.get(first_item["job_id"])
            if not first_source:
                raise ValueError(f"Job {first_item['job_id']} not found")
            first_scene = next(
                (s for s in first_source.scenes if s["index"] == first_item["scene_index"]),
                None,
            )
            if not first_scene:
                raise ValueError("First scene not found")

            update_job(mix_job_id, step="Selecting encoder...", progress=5)
            probe_out = str(tmp_dir / "probe.mp4")
            probe_r = subprocess.run(
                [
                    FFMPEG, "-y",
                    "-ss", str(first_scene["start"]),
                    "-i", first_source.input_path,
                    "-t", "1",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-threads", "1",
                    "-c:a", "aac", "-b:a", "64k", "-ar", "44100", "-ac", "2",
                    probe_out,
                ],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            libx264_works = (
                probe_r.returncode == 0
                and Path(probe_out).exists()
                and Path(probe_out).stat().st_size > 100
            )

            # Consistent audio for ALL segments: stereo 44100 Hz AAC.
            # This is critical — if segments have different sample rates or
            # channel layouts, stream-copy concat silences audio permanently.
            audio_args = ["-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"]

            if libx264_works:
                video_enc = [
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-threads", "1",
                ]
            else:
                video_enc = ["-c:v", "mpeg4", "-q:v", "6", "-pix_fmt", "yuv420p"]

            # ── Step 2: Detect target resolution from first source ───────────
            # All segments are scaled to the same W×H so stream-copy concat works
            # even when source videos have different resolutions.
            target_res = _get_video_resolution(first_source.input_path)
            if target_res:
                w, h = target_res
                scale_flags = ["-vf", f"scale={w}:{h}:force_original_aspect_ratio=disable,setsar=1"]
            else:
                scale_flags = []

            # ── Step 3: Encode all segments with the chosen encoder ──────────
            temp_files = []
            source_has_audio: dict[str, bool] = {}
            total = len(timeline)
            for i, item in enumerate(timeline):
                job_id = item["job_id"]
                scene_index = item["scene_index"]
                if job_id not in jobs:
                    raise ValueError(f"Job {job_id} not found")
                source_job = jobs[job_id]
                scene = next((s for s in source_job.scenes if s["index"] == scene_index), None)
                if not scene:
                    raise ValueError(f"Scene {scene_index} not found")
                tmp_out = str(tmp_dir / f"seg_{i:04d}.mp4")
                dur = round(scene["duration"], 3)
                update_job(
                    mix_job_id,
                    step=f"Cutting scene {i + 1}/{total}...",
                    progress=10 + int(80 * (i + 1) // total),
                )

                if job_id not in source_has_audio:
                    source_has_audio[job_id] = _has_audio(source_job.input_path)

                if source_has_audio[job_id]:
                    cmd = [
                        FFMPEG, "-y",
                        "-ss", str(scene["start"]),
                        "-i", source_job.input_path,
                        "-t", str(dur),
                    ] + video_enc + scale_flags + audio_args + [tmp_out]
                else:
                    # Source has no audio — synthesize a silent audio track so
                    # all segments have identical stream layout for concat.
                    cmd = [
                        FFMPEG, "-y",
                        "-ss", str(scene["start"]),
                        "-i", source_job.input_path,
                        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                        "-t", str(dur),
                        "-map", "0:v",
                        "-map", "1:a",
                    ] + video_enc + scale_flags + audio_args + [tmp_out]

                r = subprocess.run(
                    cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
                )
                if r.returncode != 0 or not Path(tmp_out).exists() or Path(tmp_out).stat().st_size < 1000:
                    lines = [l for l in r.stderr.splitlines() if l.strip()]
                    raise RuntimeError(f"Segment {i} encode failed:\n" + "\n".join(lines[-5:]))
                temp_files.append(tmp_out)

            # ── Step 4: Concat (safe — all segments share the same codec) ────
            update_job(mix_job_id, step="Concatenating...", progress=92)
            list_path = str(tmp_dir / "list.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for tf in temp_files:
                    f.write(f"file '{tf}'\n")
            cmd = [
                FFMPEG, "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                "-movflags", "+faststart",
                output_path,
            ]
            r = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            if r.returncode != 0:
                lines = [l for l in r.stderr.splitlines() if l.strip()]
                raise RuntimeError("Concat failed:\n" + "\n".join(lines[-5:]))
            update_job(mix_job_id, status="done", step="Done", progress=100)
        finally:
            _shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception as e:
        update_job(mix_job_id, status="error", step="Error", error=str(e))


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
