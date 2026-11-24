#!/usr/bin/env python3
"""Master Script Distillation Builder for 獨角獸那麼叫 (@unicorn-roar).

Ingests 20+ full-length creator-authored `.vtt` human scripts + RTX PRO 6000 Whisper Short scripts,
analyzes empirical length/pacing/structure laws, and writes:
- `assets/script_distillation/corpus/*.json`
- `assets/script_distillation/distilled_playbook.json`
- `assets/script_distillation/distilled_playbook.md`
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from unicorn_roar.config import REFERENCE_DIR
from unicorn_roar.scriptwriter.distiller import (
    CORPUS_DIR,
    DISTILL_DIR,
    PLAYBOOK_JSON,
    PLAYBOOK_MD,
    SIX_ACT_SCHEMA,
    classify_episode_pillar,
    extract_rhetorical_gems,
    parse_vtt_to_segments,
    slice_episode_into_six_acts,
)


def _find_vtt_and_info_pairs() -> list[tuple[Path, Path]]:
    dl = REFERENCE_DIR / "downloads"
    vtts: dict[str, Path] = {}
    infos: dict[str, Path] = {}
    for sub in ("long", "long_extra"):
        d = dl / sub
        if not d.exists():
            continue
        for vp in d.glob("*.vtt"):
            vid = vp.name.split(".")[0]
            vtts[vid] = vp
        for ip in d.glob("*.info.json"):
            vid = ip.name[: -len(".info.json")]
            infos[vid] = ip
    pairs = []
    for vid, vp in sorted(vtts.items()):
        if vid in infos:
            pairs.append((vp, infos[vid]))
    return pairs


def main() -> None:
    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    long_episodes = []
    for vtt_path, info_path in _find_vtt_and_info_pairs():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        vid = info.get("id") or vtt_path.name.split(".")[0]
        title = info.get("title", vid)
        dur = float(info.get("duration") or 0.0)
        segs = parse_vtt_to_segments(vtt_path)
        if len(segs) < 30:
            continue
        if dur <= 0:
            dur = float(segs[-1]["end"])
        full_text = "\n".join(s["text"] for s in segs)
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", full_text))
        cues_per_min = round(len(segs) / max(dur / 60.0, 0.1), 1)
        chars_per_min = round(cjk_chars / max(dur / 60.0, 0.1), 1)
        pillar = classify_episode_pillar(title)
        acts = slice_episode_into_six_acts(segs, dur)

        ep_record = {
            "id": vid,
            "kind": "long",
            "pillar": pillar,
            "title": title,
            "duration_sec": round(dur, 1),
            "duration_min": round(dur / 60.0, 2),
            "cue_count": len(segs),
            "cjk_char_count": cjk_chars,
            "chars_per_min": chars_per_min,
            "cues_per_min": cues_per_min,
            "description_sources": (info.get("description") or "")[:1200],
            "opening_60s": "，".join(s["text"] for s in segs if float(s["end"]) <= 65.0),
            "closing_60s": "，".join(s["text"] for s in segs if float(s["start"]) >= max(0.0, dur - 65.0)),
            "acts": acts,
            "segments": segs,
        }
        (CORPUS_DIR / f"long_{vid}.json").write_text(
            json.dumps(ep_record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        long_episodes.append(ep_record)

    # Load RTX PRO 6000 Whisper Short transcripts
    short_tx_file = DISTILL_DIR / "shorts_whisper_raw.json"
    short_episodes = []
    if short_tx_file.exists():
        raw_shorts = json.loads(short_tx_file.read_text(encoding="utf-8"))
        for stem, tx in raw_shorts.items():
            sid = stem.replace("short_", "")
            info_p = REFERENCE_DIR / "downloads" / "shorts" / f"{sid}.info.json"
            title = sid
            if info_p.exists():
                title = json.loads(info_p.read_text(encoding="utf-8")).get("title", sid)
            s_rec = {
                "id": sid,
                "kind": "short",
                "pillar": classify_episode_pillar(title),
                "title": title,
                "duration_sec": tx["duration_sec"],
                "cjk_char_count": tx["cjk_char_count"],
                "chars_per_min": tx["chars_per_min"],
                "full_text": tx["full_text"],
                "segments": tx["segments"],
            }
            (CORPUS_DIR / f"short_{sid}.json").write_text(
                json.dumps(s_rec, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            short_episodes.append(s_rec)

    # Empirical metrics across all distilled Long episodes
    durs = [e["duration_sec"] for e in long_episodes]
    chars = [e["cjk_char_count"] for e in long_episodes]
    cpms = [e["chars_per_min"] for e in long_episodes]
    cues = [e["cue_count"] for e in long_episodes]

    gems = extract_rhetorical_gems(long_episodes)

    # Lightweight exemplar bank (without raw segment arrays so prompt loading is instant)
    long_exemplars = [
        {
            "id": e["id"],
            "pillar": e["pillar"],
            "title": e["title"],
            "duration_sec": e["duration_sec"],
            "cjk_char_count": e["cjk_char_count"],
            "chars_per_min": e["chars_per_min"],
            "opening_60s": e["opening_60s"],
            "closing_60s": e["closing_60s"],
            "acts": e["acts"],
        }
        for e in long_episodes
    ]
    short_exemplars = [
        {
            "id": s["id"],
            "pillar": s["pillar"],
            "title": s["title"],
            "duration_sec": s["duration_sec"],
            "cjk_char_count": s["cjk_char_count"],
            "full_text": s["full_text"],
        }
        for s in short_episodes
    ]

    metrics = {
        "long_episode_count": len(long_episodes),
        "short_episode_count": len(short_episodes),
        "total_distilled_cjk_chars": sum(chars) + sum(s["cjk_char_count"] for s in short_episodes),
        "long_avg_duration_sec": round(statistics.mean(durs), 1) if durs else 0,
        "long_median_duration_sec": round(statistics.median(durs), 1) if durs else 0,
        "long_min_max_duration_sec": [round(min(durs), 1), round(max(durs), 1)] if durs else [0, 0],
        "long_avg_cjk_chars": round(statistics.mean(chars)) if chars else 0,
        "long_median_cjk_chars": round(statistics.median(chars)) if chars else 0,
        "long_avg_chars_per_min": round(statistics.mean(cpms), 1) if cpms else 0,
        "long_avg_subtitle_cues": round(statistics.mean(cues)) if cues else 0,
        "short_avg_duration_sec": round(statistics.mean([s["duration_sec"] for s in short_episodes]), 1)
        if short_episodes
        else 0,
        "short_avg_cjk_chars": round(statistics.mean([s["cjk_char_count"] for s in short_episodes]))
        if short_episodes
        else 0,
    }

    playbook = {
        "channel": "獨角獸那麼叫 (@unicorn-roar)",
        "empirical_metrics": metrics,
        "six_act_architecture": SIX_ACT_SCHEMA,
        "rhetorical_gems": gems,
        "long_exemplars": long_exemplars,
        "short_exemplars": short_exemplars,
    }
    PLAYBOOK_JSON.write_text(json.dumps(playbook, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build human-readable Markdown Distillation Whitepaper
    md = [
        "# 《獨角獸那麼叫 (@unicorn-roar)》頻道文案深度蒸餾白皮書",
        "",
        "本知識庫由頻道真實公開長片（作者人工精修 `.zh.vtt` 逐字稿）與對應推流 Shorts 全文轉錄語料自動蒸餾而成，作為雙軌長短片 AI 產稿引擎的結構、字數密度與語氣檢索基準（RAG Playbook）。",
        "",
        "## 一、頻道量化時長與字數定律（解決長片過短之核心依據）",
        "",
        f"- **蒸餾正片樣本數**：{metrics['long_episode_count']} 部完整長片 + {metrics['short_episode_count']} 部對應推流 Shorts",
        f"- **累積蒸餾繁中總字數**：**{metrics['total_distilled_cjk_chars']:,} 字**",
        f"- **正片真實平均時長**：**{metrics['long_avg_duration_sec']} 秒（約 {round(metrics['long_avg_duration_sec']/60, 1)} 分鐘）**｜中位數：{metrics['long_median_duration_sec']} 秒｜區間：{metrics['long_min_max_duration_sec'][0]}s – {metrics['long_min_max_duration_sec'][1]}s",
        f"- **正片真實平均中文字數**：**{metrics['long_avg_cjk_chars']:,} 字**（中位數：{metrics['long_median_cjk_chars']:,} 字）",
        f"- **正片真實語速密度**：平均 **{metrics['long_avg_chars_per_min']} 中文字／分鐘**，平均每支正片切分為 **{metrics['long_avg_subtitle_cues']} 句短字幕**",
        f"- **推流 Short 真實平均規格**：平均 **{metrics['short_avg_duration_sec']} 秒**、平均 **{metrics['short_avg_cjk_chars']} 中文字**",
        "",
        "> **關鍵工程結論**：單次要求 LLM 輸出 JSON 會導致模型偷懶將每段壓縮至 30 字（總計僅 400 字／約 80 秒）。要穩定還原頻道 12～15 分鐘正片，**必須採「六幕分段式深度生成（6-Act Multi-Pass Generation）」，每幕獨立對標 550～800 字蒸餾範本，總字數鎖定 3,800～4,500 字**。",
        "",
        "---",
        "",
        "## 二、六幕式敘事骨架（6-Act Narrative Architecture）",
        "",
    ]
    for act in SIX_ACT_SCHEMA:
        lo_p, hi_p = int(act["time_ratio"][0] * 100), int(act["time_ratio"][1] * 100)
        md.append(f"### {act['name']}（時間軸 {lo_p}%–{hi_p}%｜目標字數：{act['target_cjk_chars']} 字）")
        md.append(f"- **敘事任務**：{act['role_guide']}")
        md.append("")

    md.append("---")
    md.append("")
    md.append("## 三、原片蒸餾語氣轉折與金句句式庫（Rhetorical Moves）")
    md.append("")
    for cat, items in gems.items():
        md.append(f"### {cat}")
        for it in items[:8]:
            md.append(f"- 「{it}」")
        md.append("")

    md.append("---")
    md.append("")
    md.append("## 四、已蒸餾代表正片黃金開場（前 60 秒原片逐字實錄）")
    md.append("")
    for ep in long_exemplars[:10]:
        md.append(
            f"### `{ep['id']}`｜{ep['title']}（{ep['duration_sec']}s / {ep['cjk_char_count']}字 / {ep['chars_per_min']}字/分）"
        )
        md.append(f"> {ep['opening_60s'][:360]}…")
        md.append("")

    md.append("---")
    md.append("")
    md.append("## 五、已蒸餾推流 Shorts 全文結構範本")
    md.append("")
    for sh in short_exemplars[:5]:
        md.append(f"### Short `{sh['id']}`｜{sh['title']}（{sh['duration_sec']}s / {sh['cjk_char_count']}字）")
        md.append(f"> {sh['full_text'][:360].replace(chr(10), '，')}…")
        md.append("")

    PLAYBOOK_MD.write_text("\n".join(md), encoding="utf-8")
    print("DISTILLATION_COMPLETE", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
