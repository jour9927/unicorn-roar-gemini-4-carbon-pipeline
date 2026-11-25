#!/usr/bin/env python3
"""Telegram Delivery Helper for 獨角獸那麼叫 (@unicorn-roar) V2 Unattended Pipeline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import httpx


def _ensure_tg_stream_copy(mp4_path: Path, out_name: str, scale_wh: str, max_kbps: int, max_mb: float = 46.0) -> Path:
    size_mb = mp4_path.stat().st_size / (1024 * 1024)
    if size_mb <= max_mb:
        return mp4_path
    out = mp4_path.parent / out_name
    if out.exists() and out.stat().st_mtime >= mp4_path.stat().st_mtime and (out.stat().st_size / (1024 * 1024)) <= max_mb:
        return out
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(mp4_path),
            "-vf",
            f"scale={scale_wh}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "25",
            "-maxrate",
            f"{max_kbps}k",
            "-bufsize",
            f"{max_kbps * 2}k",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "44100",
            "-movflags",
            "+faststart",
            str(out),
        ],
        check=True,
    )
    return out


def _build_1080p_master_excerpt(long_mp4: Path, dur_sec: int = 75) -> Path:
    out = long_mp4.parent / f"tg_long_1080p_master_{dur_sec}s.mp4"
    if out.exists() and out.stat().st_mtime >= long_mp4.stat().st_mtime:
        return out
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "0",
            "-t",
            str(dur_sec),
            "-i",
            str(long_mp4),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-maxrate",
            "3200k",
            "-bufsize",
            "6400k",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "44100",
            "-movflags",
            "+faststart",
            str(out),
        ],
        check=True,
    )
    return out


def send_run_to_telegram(run_dir: Path, chat_id: str, token: str) -> dict:
    summary = json.loads((run_dir / "06_run_summary.json").read_text(encoding="utf-8"))
    long_mp4 = Path(summary["outputs"]["long_video_path"])
    short_mp4 = Path(summary["outputs"]["short_video_path"])
    long_probe = summary["verification"]["long_probe"]
    short_probe = summary["verification"]["short_probe"]

    tg_master_75s = _build_1080p_master_excerpt(long_mp4, dur_sec=75)
    tg_long = _ensure_tg_stream_copy(long_mp4, "tg_long_full_stream_720p.mp4", "1280:720", max_kbps=330)
    tg_short = _ensure_tg_stream_copy(short_mp4, "tg_short_stream_1080x1920.mp4", "1080:1920", max_kbps=2200)
    base_url = f"https://api.telegram.org/bot{token}"
    sent_ids = []

    with httpx.Client(timeout=300.0) as client:
        with tg_master_75s.open("rb") as fh:
            r0 = client.post(
                f"{base_url}/sendVideo",
                data={
                    "chat_id": chat_id,
                    "caption": (
                        f"🎧【V4 配音精修＋Gemini 4 Hydrogen｜1080p 無損音質前 75 秒試聽】\n"
                        f"{summary['long_title']}\n"
                        f"• 音訊：192kbps AAC｜台灣語境正音（零中式英文數字／零破音字）＋原聲抑揚頓挫＋輕撥弦配樂"
                    ),
                    "width": "1920",
                    "height": "1080",
                    "duration": "75",
                    "supports_streaming": "true",
                },
                files={"video": (tg_master_75s.name, fh, "video/mp4")},
            ).json()
            sent_ids.append(("long_1080p_75s", r0.get("ok"), r0.get("result", {}).get("message_id")))

        with tg_long.open("rb") as fh:
            r1 = client.post(
                f"{base_url}/sendVideo",
                data={
                    "chat_id": chat_id,
                    "caption": (
                        f"🎬【V4 完整正片｜{long_probe['duration_min']} 分鐘｜Gemini 4 Hydrogen】\n"
                        f"{summary['long_title']}\n"
                        f"• 字數：{summary['long_cjk_chars']} 字｜57 場景｜160kbps 高音質串流版"
                    ),
                    "width": "1280",
                    "height": "720",
                    "duration": str(int(round(long_probe["duration_sec"]))),
                    "supports_streaming": "true",
                },
                files={"video": (tg_long.name, fh, "video/mp4")},
            ).json()
            sent_ids.append(("long_full", r1.get("ok"), r1.get("result", {}).get("message_id")))

        with tg_short.open("rb") as fh:
            r2 = client.post(
                f"{base_url}/sendVideo",
                data={
                    "chat_id": chat_id,
                    "caption": (
                        f"📱【V4 對應導流 Short｜1080x1920 ({short_probe['duration_sec']}s)｜Gemini 4 Hydrogen】\n"
                        f"{summary['short_title']}"
                    ),
                    "width": "1080",
                    "height": "1920",
                    "duration": str(int(round(short_probe["duration_sec"]))),
                    "supports_streaming": "true",
                },
                files={"video": (tg_short.name, fh, "video/mp4")},
            ).json()
            sent_ids.append(("short_1080x1920", r2.get("ok"), r2.get("result", {}).get("message_id")))

    return {"ok": all(x[1] for x in sent_ids), "sent": sent_ids}


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a rendered Unicorn Roar episode to Telegram.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--chat-id", default=os.environ.get("TELEGRAM_CHAT_ID", "542571922"))
    args = parser.parse_args()

    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("FIVERR_TG_BOT_TOKEN")
    if not token:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN or FIVERR_TG_BOT_TOKEN environment variable.")

    res = send_run_to_telegram(args.run_dir, args.chat_id, token)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
