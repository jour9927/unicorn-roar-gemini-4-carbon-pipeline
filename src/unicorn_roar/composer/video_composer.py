"""Broadcast Video Composer for 獨角獸那麼叫 (@unicorn-roar): 16:9 Long + 9:16 Traffic-Push Short.

Reconstructs the exact visual, subtitle, citation, and audio mix aesthetics:
- Long (1920x1080 @ 30fps):
  - `ai_scene` with smooth 30fps Ken Burns zoom/pan + top-left stroked source citation + bottom dark pill captions
  - `evidence_doc` with white document, red marker underline, top-left `accessed:YYYYMMDD` source citation + pill captions
  - `kinetic_black` with `#000000` canvas, yellow/red kinetic stats, comparison ruler + clean white captions
- Short (1080x1920 @ 30fps):
  - Iconic 3-band layout:
    1) Top `#000000` header (`y=0..360`) with bold slanted (`skewX`) `#FFE800` yellow + `#FFFFFF` 2-line hook
    2) Center visual stage (`y=360..1480`) with AI scene / evidence card / kinetic card + punchy subtitles
    3) Bottom `#000000` footer (`y=1480..1920`) with Long-video push CTA pill + authentic EN italic disclaimer
- Audio Mastering:
  - `-16 LUFS` cloned owner narration + subtle procedural investigative ambient bed (`-31 dBFS`) + subtle transition cues.
"""
from __future__ import annotations

import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

from unicorn_roar.config import load_channel_profile
from unicorn_roar.discovery.web_crawler import ResearchDossier, SourceCitation
from unicorn_roar.scriptwriter.generator import EpisodeBlueprint, SceneSpec, ShortVideoSpec
from unicorn_roar.visuals.image_gen import (
    draw_citation_watermark,
    load_cjk_font,
    load_latin_italic_font,
    render_evidence_doc_card,
    render_kinetic_black_card,
)


def _video_encoder_args() -> list[str]:
    if platform.system() == "Darwin":
        return ["-c:v", "h264_videotoolbox", "-b:v", "6500k", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]


def _draw_subtitle_pill(
    draw: ImageDraw.ImageDraw,
    text: str,
    canvas_w: int,
    bottom_y: int,
    font_size: int = 50,
    with_pill: bool = True,
) -> None:
    if not text:
        return
    font = load_cjk_font(font_size, bold=True)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = max(32, (canvas_w - tw) // 2)
    y = bottom_y - th

    if with_pill:
        pad_x, pad_y = 22, 12
        draw.rectangle(
            [x - pad_x, y - pad_y, x + tw + pad_x, y + th + pad_y + 6],
            fill=(32, 32, 32, 214),
        )
        draw.text((x, y - bbox[1]), text, font=font, fill=(255, 255, 255, 255))
    else:
        draw.text(
            (x, y - bbox[1]),
            text,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=3,
            stroke_fill=(0, 0, 0, 255),
        )


def _render_slanted_hook_banner(
    width: int,
    height: int,
    line1: str,
    line2: str,
    highlight_words: list[str],
) -> Image.Image:
    """Render `@unicorn-roar` signature slanted (`skewX(-8deg)`) yellow/white Short top hook."""
    layer = Image.new("RGBA", (width + 120, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    max_len = max(len(line1), len(line2), 8)
    font_size = min(74, max(48, int((width - 80) / max_len * 0.96)))
    font = load_cjk_font(font_size, bold=True)

    for idx, line in enumerate([line1, line2]):
        if not line:
            continue
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        cur_x = max(40, (width + 120 - tw) // 2)
        cur_y = 58 + idx * int(font_size * 1.38)

        # If highlight_words provided, color matching keywords yellow `#FFE800` and rest white,
        # otherwise render full line in signature `#FFE800` bright yellow!
        any_hl = any(hw and hw in line for hw in highlight_words)
        if not any_hl:
            draw.text(
                (cur_x, cur_y),
                line,
                font=font,
                fill=(255, 232, 0, 255),
                stroke_width=3,
                stroke_fill=(0, 0, 0, 255),
            )
        else:
            rem = line
            while rem:
                best_pos = len(rem)
                best_hw = ""
                for hw in highlight_words:
                    if not hw:
                        continue
                    pos = rem.find(hw)
                    if pos != -1 and pos < best_pos:
                        best_pos = pos
                        best_hw = hw
                if not best_hw:
                    draw.text((cur_x, cur_y), rem, font=font, fill=(255, 255, 255, 255))
                    break
                if best_pos > 0:
                    pre = rem[:best_pos]
                    draw.text((cur_x, cur_y), pre, font=font, fill=(255, 255, 255, 255))
                    cur_x += draw.textbbox((0, 0), pre, font=font)[2]
                draw.text((cur_x, cur_y), best_hw, font=font, fill=(255, 232, 0, 255))
                cur_x += draw.textbbox((0, 0), best_hw, font=font)[2]
                rem = rem[best_pos + len(best_hw) :]

    # Apply subtle italic shear transform `x' = x + 0.14 * y`
    sheared = layer.transform(
        (width, height),
        Image.Transform.AFFINE,
        (1.0, 0.14, -30.0, 0.0, 1.0, 0.0),
        resample=Image.Resampling.BICUBIC,
    )
    return sheared


def _synthesize_ambient_bed_and_mix(
    scene_wavs: list[tuple[Path, str]],
    out_mixed_wav: Path,
    inter_pause_sec: float = 0.34,
) -> tuple[Path, list[tuple[float, float]]]:
    """Concatenate scene narrations with natural pauses, subtle ambient pulse bed, and soft transition cues."""
    sr = 44100
    chunks: list[np.ndarray] = []
    timings: list[tuple[float, float]] = []
    cursor = 0.0
    pause_samples = int(inter_pause_sec * sr)
    silence = np.zeros((pause_samples, 2), dtype=np.float32)

    cue_positions_samples: list[tuple[int, str]] = []

    for i, (wav_path, mode) in enumerate(scene_wavs):
        data, file_sr = sf.read(str(wav_path), dtype="float32")
        if file_sr != sr:
            raise ValueError(f"Expected 44100Hz WAV, got {file_sr} for {wav_path}")
        if data.ndim == 1:
            data = np.stack([data, data], axis=1)
        start_t = cursor
        dur = len(data) / float(sr)
        end_t = start_t + dur
        timings.append((start_t, end_t))
        cue_positions_samples.append((int(start_t * sr), mode))
        chunks.append(data)
        cursor = end_t
        if i < len(scene_wavs) - 1:
            chunks.append(silence)
            cursor += inter_pause_sec

    # Add 0.45s tail pad
    tail = np.zeros((int(0.45 * sr), 2), dtype=np.float32)
    chunks.append(tail)
    voice_track = np.concatenate(chunks, axis=0)
    total_samples = len(voice_track)
    t = np.arange(total_samples, dtype=np.float32) / float(sr)

    # Subtle investigative warm minor drone/pulse at -32 dBFS (never competes with voice!)
    bed_mono = (
        0.014 * np.sin(2.0 * math.pi * 110.0 * t)
        + 0.009 * np.sin(2.0 * math.pi * 164.81 * t)
        + 0.006 * np.sin(2.0 * math.pi * 220.0 * t) * (0.6 + 0.4 * np.sin(2.0 * math.pi * 0.25 * t))
    ).astype(np.float32)
    # Fade in 0.8s and fade out 1.2s on bed
    fi = min(int(0.8 * sr), total_samples // 4)
    fo = min(int(1.2 * sr), total_samples // 4)
    bed_mono[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    bed_mono[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    bed = np.stack([bed_mono, bed_mono], axis=1)

    # Subtle transition cue on evidence_doc / kinetic_black scene cuts
    for pos, mode in cue_positions_samples[1:]:
        cue_len = int(0.09 * sr)
        if pos + cue_len < total_samples:
            ct = np.arange(cue_len, dtype=np.float32) / float(sr)
            env = np.exp(-ct * 42.0)
            freq = 520.0 if mode == "evidence_doc" else 180.0
            cue = (0.045 * np.sin(2.0 * math.pi * freq * ct) * env).astype(np.float32)
            bed[pos : pos + cue_len, 0] += cue
            bed[pos : pos + cue_len, 1] += cue

    mixed = np.clip(voice_track + bed, -0.96, 0.96)
    out_mixed_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_mixed_wav), mixed, sr, subtype="PCM_16")
    return out_mixed_wav, timings


def _render_long_scene_clip(
    idx: int,
    scene: SceneSpec,
    tts_info: dict[str, Any],
    scene_total_dur: float,
    citation: SourceCitation | None,
    ai_images: dict[str, Path],
    work_dir: Path,
) -> Path:
    W, H, FPS = 1920, 1080, 30
    scene_dir = work_dir / f"long_{scene.scene_id}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = work_dir / f"seg_{idx:02d}_{scene.scene_id}.mp4"

    subs = list(tts_info["subtitles"])
    if subs:
        subs[-1] = {**subs[-1], "end": round(scene_total_dur, 3)}
    else:
        subs = [{"start": 0.0, "end": round(scene_total_dur, 3), "text": scene.narration[:18]}]

    if scene.scene_mode in ("evidence_doc", "kinetic_black"):
        base_card = scene_dir / "base.png"
        if scene.scene_mode == "evidence_doc":
            render_evidence_doc_card(scene, citation, W, H, base_card, include_watermark=True)
        else:
            render_kinetic_black_card(scene, citation, W, H, base_card, include_watermark=False)

        base_img = Image.open(base_card).convert("RGBA")
        concat_lines = []
        for s_i, sub in enumerate(subs):
            dur = max(0.12, round(float(sub["end"]) - float(sub["start"]), 3))
            frame = base_img.copy()
            draw = ImageDraw.Draw(frame)
            _draw_subtitle_pill(
                draw,
                sub["text"],
                canvas_w=W,
                bottom_y=H - 68,
                font_size=50,
                with_pill=(scene.scene_mode == "evidence_doc"),
            )
            f_path = scene_dir / f"sub_{s_i:02d}.png"
            frame.convert("RGB").save(f_path)
            sub_mp4 = scene_dir / f"sub_{s_i:02d}.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loop",
                    "1",
                    "-i",
                    str(f_path),
                    "-t",
                    f"{dur:.3f}",
                    "-r",
                    str(FPS),
                    *_video_encoder_args(),
                    str(sub_mp4),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            concat_lines.append(f"file '{sub_mp4.name}'")

        list_txt = scene_dir / "concat.txt"
        list_txt.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_txt),
                "-c",
                "copy",
                str(out_mp4),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return out_mp4

    # `ai_scene`: Smooth 30fps Ken Burns zoom/pan + transparent RGBA subtitle & citation overlays!
    raw_img_path = ai_images[scene.scene_id]
    fitted_bg = scene_dir / "bg_1920.png"
    im = Image.open(raw_img_path).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    im.save(fitted_bg)

    n_frames = max(15, int(round(scene_total_dur * FPS)))
    zoom_expr = (
        "min(zoom+0.00065,1.11)"
        if idx % 2 == 0
        else "if(eq(on,1),1.11,max(1.0,zoom-0.00065))"
    )

    inputs = ["-loop", "1", "-t", f"{scene_total_dur:.3f}", "-i", str(fitted_bg)]
    filter_parts = [
        f"[0:v]zoompan=z='{zoom_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n_frames}:s={W}x{H}:fps={FPS},format=rgba[v0]"
    ]
    prev_label = "v0"

    for s_i, sub in enumerate(subs):
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(ov)
        if citation:
            draw_citation_watermark(odraw, citation, x=14, y=10, font_size=24)
        _draw_subtitle_pill(
            odraw,
            sub["text"],
            canvas_w=W,
            bottom_y=H - 68,
            font_size=50,
            with_pill=True,
        )
        ov_path = scene_dir / f"ov_{s_i:02d}.png"
        ov.save(ov_path)
        in_idx = s_i + 1
        inputs += ["-loop", "1", "-t", f"{scene_total_dur:.3f}", "-i", str(ov_path)]
        s_t = float(sub["start"])
        e_t = float(sub["end"])
        next_label = f"v{in_idx}"
        enable_expr = f"between(t,{s_t:.3f},{e_t:.3f})"
        filter_parts.append(
            f"[{prev_label}][{in_idx}:v]overlay=0:0:enable='{enable_expr}'[{next_label}]"
        )
        prev_label = next_label

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            f"[{prev_label}]",
            "-t",
            f"{scene_total_dur:.3f}",
            "-r",
            str(FPS),
            *_video_encoder_args(),
            str(out_mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_mp4


def _render_short_scene_clip(
    idx: int,
    scene: SceneSpec,
    short_spec: ShortVideoSpec,
    tts_info: dict[str, Any],
    scene_total_dur: float,
    citation: SourceCitation | None,
    ai_images: dict[str, Path],
    work_dir: Path,
    disclaimer_lines: list[str],
) -> Path:
    """Render 1080x1920 9:16 vertical Short segment using `@unicorn-roar` 3-band template."""
    W, H, FPS = 1080, 1920, 30
    TOP_H = 360
    STAGE_H = 1100
    scene_dir = work_dir / f"short_{scene.scene_id}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = work_dir / f"short_seg_{idx:02d}_{scene.scene_id}.mp4"

    subs = list(tts_info["subtitles"])
    if subs:
        subs[-1] = {**subs[-1], "end": round(scene_total_dur, 3)}
    else:
        subs = [{"start": 0.0, "end": round(scene_total_dur, 3), "text": scene.narration[:14]}]

    # Build static 1080x1920 chrome template (top slanted hook + bottom CTA + EN italic disclaimer)
    chrome = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(chrome)
    cdraw.rectangle([0, 0, W, TOP_H], fill=(0, 0, 0, 255))
    cdraw.rectangle([0, TOP_H + STAGE_H, W, H], fill=(0, 0, 0, 255))

    hook_img = _render_slanted_hook_banner(
        W,
        TOP_H,
        short_spec.top_hook_line1,
        short_spec.top_hook_line2,
        short_spec.hook_highlight_words,
    )
    chrome.alpha_composite(hook_img, (0, 0))

    # Push-to-Long CTA pill in upper footer band
    cta_font = load_cjk_font(34, bold=True)
    cta_txt = short_spec.push_cta_banner or "▶ 點擊下方連結觀看完整正片拆解"
    cbbox = cdraw.textbbox((0, 0), cta_txt, font=cta_font)
    ctw = cbbox[2] - cbbox[0]
    cx = max(36, (W - ctw) // 2)
    cy = TOP_H + STAGE_H + 54
    cdraw.rounded_rectangle(
        [cx - 24, cy - 12, cx + ctw + 24, cy + 48],
        radius=24,
        fill=(28, 28, 28, 255),
        outline=(255, 232, 0, 220),
        width=2,
    )
    cdraw.text((cx, cy), cta_txt, font=cta_font, fill=(255, 232, 0, 255))

    # Authentic bottom-left subtle gray italic English disclaimer (matching 0p9h-fymu3k_8s.jpg!)
    disc_font = load_latin_italic_font(23)
    dy = H - 148
    for dline in disclaimer_lines[:4]:
        cdraw.text((16, dy), dline, font=disc_font, fill=(176, 176, 176, 255))
        dy += 31

    # Stage content (`1080 x 1100`)
    stage_path = scene_dir / "stage.png"
    if scene.scene_mode == "evidence_doc":
        render_evidence_doc_card(scene, citation, W, STAGE_H, stage_path, include_watermark=True)
    elif scene.scene_mode == "kinetic_black":
        render_kinetic_black_card(scene, citation, W, STAGE_H, stage_path, include_watermark=True)
    else:
        raw_img = Image.open(ai_images[scene.scene_id]).convert("RGB")
        # Center-crop/resize to 1080x1100
        stage_im = raw_img.resize((W, STAGE_H), Image.Resampling.LANCZOS)
        stage_im.save(stage_path)

    if scene.scene_mode in ("evidence_doc", "kinetic_black"):
        stage_rgba = Image.open(stage_path).convert("RGBA")
        concat_lines = []
        for s_i, sub in enumerate(subs):
            dur = max(0.12, round(float(sub["end"]) - float(sub["start"]), 3))
            frame = Image.new("RGBA", (W, H), (0, 0, 0, 255))
            frame.alpha_composite(stage_rgba, (0, TOP_H))
            frame.alpha_composite(chrome, (0, 0))
            fdraw = ImageDraw.Draw(frame)
            _draw_subtitle_pill(
                fdraw,
                sub["text"],
                canvas_w=W,
                bottom_y=TOP_H + STAGE_H - 48,
                font_size=54,
                with_pill=True,
            )
            fp = scene_dir / f"sframe_{s_i:02d}.png"
            frame.convert("RGB").save(fp)
            smp4 = scene_dir / f"sframe_{s_i:02d}.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loop",
                    "1",
                    "-i",
                    str(fp),
                    "-t",
                    f"{dur:.3f}",
                    "-r",
                    str(FPS),
                    *_video_encoder_args(),
                    str(smp4),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            concat_lines.append(f"file '{smp4.name}'")

        list_txt = scene_dir / "concat.txt"
        list_txt.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_txt), "-c", "copy", str(out_mp4)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return out_mp4

    # `ai_scene` in Short: Ken Burns inside `1080x1100` stage placed at `y=TOP_H` + RGBA chrome & subtitle overlays
    n_frames = max(15, int(round(scene_total_dur * FPS)))
    zoom_expr = "min(zoom+0.0009,1.13)" if idx % 2 == 0 else "if(eq(on,1),1.13,max(1.0,zoom-0.0009))"
    inputs = ["-loop", "1", "-t", f"{scene_total_dur:.3f}", "-i", str(stage_path)]
    filter_parts = [
        f"[0:v]zoompan=z='{zoom_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n_frames}:s={W}x{STAGE_H}:fps={FPS},pad={W}:{H}:0:{TOP_H}:black,format=rgba[v0]"
    ]
    prev_label = "v0"

    for s_i, sub in enumerate(subs):
        ov = chrome.copy()
        odraw = ImageDraw.Draw(ov)
        if citation:
            draw_citation_watermark(odraw, citation, x=14, y=TOP_H + 10, font_size=22)
        _draw_subtitle_pill(
            odraw,
            sub["text"],
            canvas_w=W,
            bottom_y=TOP_H + STAGE_H - 48,
            font_size=54,
            with_pill=True,
        )
        ov_path = scene_dir / f"sov_{s_i:02d}.png"
        ov.save(ov_path)
        in_idx = s_i + 1
        inputs += ["-loop", "1", "-t", f"{scene_total_dur:.3f}", "-i", str(ov_path)]
        s_t = float(sub["start"])
        e_t = float(sub["end"])
        next_label = f"v{in_idx}"
        filter_parts.append(
            f"[{prev_label}][{in_idx}:v]overlay=0:0:enable='between(t,{s_t:.3f},{e_t:.3f})'[{next_label}]"
        )
        prev_label = next_label

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            f"[{prev_label}]",
            "-t",
            f"{scene_total_dur:.3f}",
            "-r",
            str(FPS),
            *_video_encoder_args(),
            str(out_mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_mp4


def _export_srt(
    scenes: list[SceneSpec],
    tts_map: dict[str, dict[str, Any]],
    scene_timings: list[tuple[float, float]],
    srt_path: Path,
) -> Path:
    def _fmt_ts(sec: float) -> str:
        ms = int(round(sec * 1000))
        h, rem = divmod(ms, 3600000)
        m, rem = divmod(rem, 60000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    counter = 1
    for sc, (s_offset, _) in zip(scenes, scene_timings):
        for sub in tts_map[sc.scene_id]["subtitles"]:
            st = s_offset + float(sub["start"])
            et = s_offset + float(sub["end"])
            blocks.append(f"{counter}\n{_fmt_ts(st)} --> {_fmt_ts(et)}\n{sub['text']}\n")
            counter += 1
    srt_path.write_text("\n".join(blocks), encoding="utf-8")
    return srt_path


def compose_long_and_short_videos(
    blueprint: EpisodeBlueprint,
    dossier: ResearchDossier,
    tts_map: dict[str, dict[str, Any]],
    ai_images: dict[str, Path],
    run_dir: Path,
) -> dict[str, Any]:
    """Compose both the 16:9 Long Episode MP4 and the 9:16 Traffic-Push Short MP4."""
    profile = load_channel_profile()
    disclaimer_en = profile["visual_specs"]["short_video"]["footer_disclaimer_en"]
    citations_by_id = {c.source_id: c for c in dossier.citations}
    default_cite = dossier.citations[0] if dossier.citations else None

    work_dir = run_dir / "temp" / "compose"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 1. COMPOSE 16:9 LONG VIDEO ===
    long_scenes = blueprint.long_video.scenes
    long_wav_list = [
        (Path(tts_map[sc.scene_id]["local_path"]), sc.scene_mode) for sc in long_scenes
    ]
    long_mix_wav, long_timings = _synthesize_ambient_bed_and_mix(
        long_wav_list, work_dir / "long_master_audio.wav", inter_pause_sec=0.35
    )
    long_clips: list[Path] = []
    for i, sc in enumerate(long_scenes):
        start_t, _ = long_timings[i]
        next_start = (
            long_timings[i + 1][0]
            if i + 1 < len(long_timings)
            else (long_timings[i][1] + 0.45)
        )
        slot_dur = round(next_start - start_t, 3)
        cite = citations_by_id.get(sc.citation_id or "", default_cite)
        clip = _render_long_scene_clip(
            i, sc, tts_map[sc.scene_id], slot_dur, cite, ai_images, work_dir
        )
        long_clips.append(clip)

    long_concat_txt = work_dir / "long_concat.txt"
    long_concat_txt.write_text(
        "\n".join(f"file '{c.resolve()}'" for c in long_clips) + "\n",
        encoding="utf-8",
    )
    long_video_only = work_dir / "long_video_only.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(long_concat_txt), "-c", "copy", str(long_video_only)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    long_final_mp4 = out_dir / "long_episode_1080p.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(long_video_only),
            "-i",
            str(long_mix_wav),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-shortest",
            "-movflags",
            "+faststart",
            str(long_final_mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    long_srt = _export_srt(long_scenes, tts_map, long_timings, out_dir / "long_episode.zh-TW.srt")

    # === 2. COMPOSE 9:16 TRAFFIC-PUSH SHORT VIDEO ===
    short_scenes = blueprint.short_video.scenes
    short_wav_list = [
        (Path(tts_map[sc.scene_id]["local_path"]), sc.scene_mode) for sc in short_scenes
    ]
    short_mix_wav, short_timings = _synthesize_ambient_bed_and_mix(
        short_wav_list, work_dir / "short_master_audio.wav", inter_pause_sec=0.22
    )
    short_clips: list[Path] = []
    for i, sc in enumerate(short_scenes):
        start_t, _ = short_timings[i]
        next_start = (
            short_timings[i + 1][0]
            if i + 1 < len(short_timings)
            else (short_timings[i][1] + 0.45)
        )
        slot_dur = round(next_start - start_t, 3)
        cite = citations_by_id.get(sc.citation_id or "", default_cite)
        clip = _render_short_scene_clip(
            i,
            sc,
            blueprint.short_video,
            tts_map[sc.scene_id],
            slot_dur,
            cite,
            ai_images,
            work_dir,
            disclaimer_en,
        )
        short_clips.append(clip)

    short_concat_txt = work_dir / "short_concat.txt"
    short_concat_txt.write_text(
        "\n".join(f"file '{c.resolve()}'" for c in short_clips) + "\n",
        encoding="utf-8",
    )
    short_video_only = work_dir / "short_video_only.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(short_concat_txt), "-c", "copy", str(short_video_only)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    short_final_mp4 = out_dir / "short_promo_1080x1920.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(short_video_only),
            "-i",
            str(short_mix_wav),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-shortest",
            "-movflags",
            "+faststart",
            str(short_final_mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    short_srt = _export_srt(short_scenes, tts_map, short_timings, out_dir / "short_promo.zh-TW.srt")

    # Write ready-to-publish YouTube metadata & source citation manifest
    (out_dir / "youtube_long_description.txt").write_text(blueprint.long_description, encoding="utf-8")
    (out_dir / "youtube_short_description.txt").write_text(blueprint.short_description, encoding="utf-8")

    return {
        "long_video_path": str(long_final_mp4),
        "long_srt_path": str(long_srt),
        "long_duration_sec": round(long_timings[-1][1] + 0.45, 2),
        "short_video_path": str(short_final_mp4),
        "short_srt_path": str(short_srt),
        "short_duration_sec": round(short_timings[-1][1] + 0.45, 2),
    }
