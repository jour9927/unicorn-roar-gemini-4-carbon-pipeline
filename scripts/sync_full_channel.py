#!/usr/bin/env python3
"""Authorized Full-Channel Archiver for 獨角獸那麼叫 (@unicorn-roar).

Supports downloading all Long-form videos and Shorts with Traditional Chinese metadata,
creator subtitles (.vtt), and resumable archive tracking.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

CHANNEL_VIDEOS_URL = "https://www.youtube.com/@unicorn-roar/videos"
CHANNEL_SHORTS_URL = "https://www.youtube.com/@unicorn-roar/shorts"


def sync_playlist(
    url: str,
    out_dir: Path,
    archive_file: Path,
    max_height: int = 1080,
    limit: int | None = None,
    audio_only: bool = False,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "-i",
        "--no-warnings",
        "--extractor-args",
        "youtube:lang=zh-TW",
        "--download-archive",
        str(archive_file),
        "--write-info-json",
        "--write-subs",
        "--sub-langs",
        "zh-TW,zh-Hant,zh",
        "-o",
        str(out_dir / "%(id)s.%(ext)s"),
    ]
    if audio_only:
        cmd += ["-x", "--audio-format", "wav", "--audio-quality", "0"]
    else:
        cmd += [
            "-f",
            f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/b[height<={max_height}][ext=mp4]/b",
            "--merge-output-format",
            "mp4",
        ]
    if limit is not None and limit > 0:
        cmd += ["--playlist-end", str(limit)]
    cmd.append(url)
    cp = subprocess.run(cmd)
    return cp.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync @unicorn-roar videos & Shorts")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("reference_channel/downloads"),
        help="Destination directory for downloaded channel media",
    )
    parser.add_argument("--limit-long", type=int, default=None, help="Max long videos (None = all)")
    parser.add_argument("--limit-shorts", type=int, default=None, help="Max Shorts (None = all)")
    parser.add_argument("--max-height", type=int, default=1080)
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--skip-shorts", action="store_true")
    args = parser.parse_args()

    archive_dir = args.dest / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    if not args.skip_long:
        rc_long = sync_playlist(
            CHANNEL_VIDEOS_URL,
            args.dest / "long",
            archive_dir / "long_archive.txt",
            max_height=args.max_height,
            limit=args.limit_long,
            audio_only=args.audio_only,
        )
        summary["long_rc"] = rc_long

    if not args.skip_shorts:
        rc_shorts = sync_playlist(
            CHANNEL_SHORTS_URL,
            args.dest / "shorts",
            archive_dir / "shorts_archive.txt",
            max_height=max(args.max_height, 1920),
            limit=args.limit_shorts,
            audio_only=args.audio_only,
        )
        summary["shorts_rc"] = rc_shorts

    print(json.dumps({"ok": True, "dest": str(args.dest), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
