"""Unattended End-to-End Pipeline Orchestrator for 獨角獸那麼叫 (@unicorn-roar)."""
from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console

from unicorn_roar.composer.video_composer import compose_long_and_short_videos
from unicorn_roar.config import RUNS_DIR
from unicorn_roar.discovery.topic_radar import discover_and_select_topic
from unicorn_roar.discovery.web_crawler import build_research_dossier
from unicorn_roar.scriptwriter.generator import generate_episode_blueprint
from unicorn_roar.visuals.image_gen import generate_episode_ai_images
from unicorn_roar.voice.cloner import synthesize_episode_narration

console = Console()


def _probe_video(mp4_path: Path) -> dict[str, Any]:
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(mp4_path),
        ],
        text=True,
    )
    data = json.loads(raw)
    v_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    a_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    return {
        "width": v_stream.get("width"),
        "height": v_stream.get("height"),
        "fps": v_stream.get("r_frame_rate"),
        "video_codec": v_stream.get("codec_name"),
        "audio_codec": a_stream.get("codec_name"),
        "sample_rate": a_stream.get("sample_rate"),
        "duration_sec": round(float(data.get("format", {}).get("duration", 0.0)), 2),
        "size_mb": round(int(data.get("format", {}).get("size", 0)) / (1024 * 1024), 2),
    }


def run_unattended_episode(
    pillar: str | None = None,
    manual_topic: str | None = None,
    long_scenes: int = 10,
    short_scenes: int = 5,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Execute the full unattended pipeline from autonomous topic radar to Long + Short MP4s."""
    rid = run_id or dt.datetime.now().strftime("ep_%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold yellow]獨角獸那麼叫 無人值守雙軌產線｜Run {rid}")

    # Phase 1: Autonomous Topic Radar
    console.print("[cyan][1/6] 自動巡檢公開議題與篩選最佳主題...")
    winner, all_candidates = discover_and_select_topic(
        pillar_filter=pillar,
        manual_topic=manual_topic,
    )
    (run_dir / "01_topic_candidates.json").write_text(
        json.dumps([c.model_dump() for c in all_candidates], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"  ✓ 鎖定主題：[bold white]{winner.proposed_title}[/] (score={winner.score})")

    # Phase 2: Public Web Crawler + Source Citations
    console.print("[cyan][2/6] 啟動多源公開資訊爬蟲並建立來源引用檔案...")
    dossier = build_research_dossier(
        topic=winner.proposed_title,
        category=winner.pillar,
        extra_queries=winner.search_queries,
        max_citations=10,
    )
    (run_dir / "02_research_dossier.json").write_text(
        dossier.model_dump_json(indent=2),
        encoding="utf-8",
    )
    console.print(f"  ✓ 爬取並綁定 [bold green]{len(dossier.citations)}[/] 筆公開來源與法規引用")

    # Phase 3: Dual-Track AI Script Generation
    console.print("[cyan][3/6] 生成「獨角獸那麼叫」16:9 正片腳本 + 9:16 推流 Short 腳本...")
    blueprint = generate_episode_blueprint(
        candidate=winner,
        dossier=dossier,
        run_dir=run_dir,
        target_long_scenes=long_scenes,
        target_short_scenes=short_scenes,
    )
    (run_dir / "03_episode_blueprint.json").write_text(
        blueprint.model_dump_json(indent=2),
        encoding="utf-8",
    )
    console.print(f"  ✓ 正片標題：[bold yellow]{blueprint.long_video.title}[/]")
    console.print(f"  ✓ Short 標題：[bold yellow]{blueprint.short_video.title}[/]")

    # Phase 4: PyTorch Owner Voice Cloning TTS
    console.print("[cyan][4/6] 執行 PyTorch 頻道主本人聲音克隆配音與逐句時間軸對齊...")
    tts_map = synthesize_episode_narration(blueprint, run_dir)
    console.print(f"  ✓ 完成 [bold green]{len(tts_map)}[/] 段本人聲線克隆配音（-16 LUFS 廣播級響度標準化）")

    # Phase 5: AI Visuals Generation
    console.print("[cyan][5/6] 執行 PyTorch AI 情境產圖與公文紅線證據卡準備...")
    ai_images = generate_episode_ai_images(blueprint, run_dir)
    console.print(f"  ✓ 完成 [bold green]{len(ai_images)}[/] 張高解析 AI 視覺場景圖")

    # Phase 6: Broadcast Composition (Long + Short)
    console.print("[cyan][6/6] 合成 1920x1080 正片與 1080x1920 導流 Short 成片...")
    composed = compose_long_and_short_videos(
        blueprint=blueprint,
        dossier=dossier,
        tts_map=tts_map,
        ai_images=ai_images,
        run_dir=run_dir,
    )

    long_probe = _probe_video(Path(composed["long_video_path"]))
    short_probe = _probe_video(Path(composed["short_video_path"]))

    summary = {
        "ok": True,
        "run_id": rid,
        "run_dir": str(run_dir),
        "selected_topic": winner.model_dump(),
        "citations_count": len(dossier.citations),
        "citations": [c.model_dump() for c in dossier.citations],
        "long_title": blueprint.long_video.title,
        "short_title": blueprint.short_video.title,
        "outputs": composed,
        "verification": {
            "long_probe": long_probe,
            "short_probe": short_probe,
        },
    }
    (run_dir / "06_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print("[bold green]✓ 無人值守雙軌成片產出完成！[/]")
    return summary
