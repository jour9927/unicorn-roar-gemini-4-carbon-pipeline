#!/usr/bin/env python3
"""Telegram Delivery Helper for 獨角獸那麼叫 (@unicorn-roar) V2 Unattended Pipeline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import httpx


def _ensure_tg_stream_copy(long_mp4: Path, max_mb: float = 48.0) -> Path:
    size_mb = long_mp4.stat().st_size / (1024 * 1024)
    if size_mb <= max_mb:
        return long_mp4
    out = long_mp4.parent / "tg_long_full_stream_720p.mp4"
    if out.exists() and out.stat().st_mtime >= long_mp4.stat().st_mtime:
        return out
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(long_mp4),
            "-vf",
            "scale=1280:720",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "25",
            "-maxrate",
            "380k",
            "-bufsize",
            "760k",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
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

    tg_long = _ensure_tg_stream_copy(long_mp4)
    base_url = f"https://api.telegram.org/bot{token}"
    sent_ids = []

    with httpx.Client(timeout=300.0) as client:
        with tg_long.open("rb") as fh:
            r1 = client.post(
                f"{base_url}/sendVideo",
                data={
                    "chat_id": chat_id,
                    "caption": (
                        f"🎬【V2 完整正片｜{long_probe['duration_min']} 分鐘】\n"
                        f"{summary['long_title']}\n"
                        f"• 字數：{summary['long_cjk_chars']} 字｜{summary.get('long_scene_count', 69)} 場景\n"
                        f"• 音訊：本人聲線克隆 + 自動剪氣口 + -13.3 LUFS"
                    ),
                    "width": "1280",
                    "height": "720",
                    "duration": str(int(round(long_probe["duration_sec"]))),
                    "supports_streaming": "true",
                },
                files={"video": (tg_long.name, fh, "video/mp4")},
            ).json()
            sent_ids.append(("long_full", r1.get("ok"), r1.get("result", {}).get("message_id")))

        with short_mp4.open("rb") as fh:
            r2 = client.post(
                f"{base_url}/sendVideo",
                data={
                    "chat_id": chat_id,
                    "caption": (
                        f"📱【V2 對應推流 Short｜1080x1920 原畫質 ({short_probe['duration_sec']}s)】\n"
                        f"{summary['short_title']}"
                    ),
                    "width": "1080",
                    "height": "1920",
                    "duration": str(int(round(short_probe["duration_sec"]))),
                    "supports_streaming": "true",
                },
                files={"video": (short_mp4.name, fh, "video/mp4")},
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
