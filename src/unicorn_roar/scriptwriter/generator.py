"""Distilled 6-Act Dual-Track AI Scriptwriter for 獨角獸那麼叫 (@unicorn-roar).

Powered by `assets/script_distillation/distilled_playbook.json` (distilled from 25 full-length
human-authored @unicorn-roar scripts + 7 Shorts, totaling 110,956 CJK characters):
- Generates Long episodes via 6 dedicated Act passes (`act_1`..`act_6`), retrieving real
  pillar-matched @unicorn-roar exemplar passages per Act and enforcing >= 4,000 CJK characters
  (~12.5 to 15 minutes @ ~315 chars/min, matching the channel's 793s empirical mean).
- Generates Corresponding Traffic-Push Shorts (~750-950 CJK chars, ~125-155s) modeled on
  real @unicorn-roar Shorts ("回答我!" hook -> core conflict -> push to full Long video).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from unicorn_roar.config import REPO_ROOT, get_runtime_endpoint_path, load_channel_profile
from unicorn_roar.discovery.topic_radar import TopicCandidate
from unicorn_roar.discovery.web_crawler import ResearchDossier
from unicorn_roar.remote.rtx6000_client import (
    RTX6000WorkerClient,
    RemoteWorkerEndpoint,
)
from unicorn_roar.scriptwriter.distiller import (
    SIX_ACT_SCHEMA,
    load_distilled_playbook,
    retrieve_act_exemplars,
)


class KineticLine(BaseModel):
    text: str
    highlight: str = ""
    highlight_color: Literal["yellow", "red"] = "yellow"


class SceneSpec(BaseModel):
    scene_id: str
    scene_mode: Literal["evidence_doc", "kinetic_black", "ai_scene"]
    voice_style: Literal["hook_question", "narrative_explain", "satire_punch", "short_energetic"] = "narrative_explain"
    narration: str
    citation_id: str | None = None
    document_heading: str = ""
    document_excerpt: str = ""
    red_underline_quote: str = ""
    kinetic_lines: list[KineticLine] = Field(default_factory=list)
    image_prompt: str = ""


class LongVideoSpec(BaseModel):
    title: str
    seo_tags: list[str] = Field(default_factory=list)
    scenes: list[SceneSpec]


class ShortVideoSpec(BaseModel):
    title: str
    top_hook_line1: str
    top_hook_line2: str
    hook_highlight_words: list[str] = Field(default_factory=list)
    push_cta_banner: str = "▶ 點擊下方連結觀看完整正片拆解"
    scenes: list[SceneSpec]


class EpisodeBlueprint(BaseModel):
    topic: str
    pillar: str
    long_cjk_chars: int = 0
    short_cjk_chars: int = 0
    long_video: LongVideoSpec
    short_video: ShortVideoSpec
    long_description: str = ""
    short_description: str = ""


BASE_SYSTEM_PROMPT = """你是台灣百萬觀看級深度消費與制度拆解頻道「獨角獸那麼叫 (@unicorn-roar)」本尊編劇。
你必須嚴格按照「頻道原片蒸餾語氣、句式節奏與字數密度」撰寫逐字口播稿：

### 頻道不可妥協之寫作鐵律：
1. **純台灣繁體口語**：使用「主計總處、食藥署、農業部、消基會、超商、盤子、統計口徑、約定俗成、業者」，語氣犀利、冷幽默吐槽與硬核法規算術交替推進。
2. **禁止空洞摘要句**：每一段 `narration` 必須是 **可以直接唸出來的完整口語段落（每段 105～140 個中文字，包含具體數字試算、條文門檻、生活場景舉例與反問句）**，絕不可寫成短短二三十字的投影片大綱！
3. **三種視覺場景交替編排 (`scene_mode`)**：
   - `"ai_scene"`：寫實電影感或諷刺隱喻畫面，`image_prompt` 用具體英文攝影描述（禁止要求印字）。
   - `"evidence_doc"`：白底公文/法規/新聞證據卡，綁定真實 `citation_id`（如 `"S1"`），`document_heading` 寫公文或報導標題，`document_excerpt` 寫 75～115 字具體法規條文或數據重點摘要（不可照抄標題），`red_underline_quote` 擷取 `document_excerpt` 中最關鍵的 12～24 字（系統自動畫紅線並壓左上角 `accessed:YYYYMMDD` 來源標記）。
   - `"kinetic_black"`：純黑底黃紅高對比大字公式或排名對照圖卡，填寫 2～3 行 `kinetic_lines`（含 `text`, `highlight`, `highlight_color`: `"yellow"` 或 `"red"`）。
4. 只輸出合法 JSON，不加任何額外說明文字。"""


def _format_sources_block(dossier: ResearchDossier) -> str:
    lines = []
    for c in dossier.citations:
        lines.append(
            f"- [{c.source_id}] {c.publisher}（{c.published_date}）：{c.title}\n"
            f"  摘要：{c.excerpt}"
        )
    if dossier.background_notes:
        lines.append("\n【百科與法規背景知識】：\n" + "\n".join(dossier.background_notes))
    return "\n".join(lines)


def _build_six_act_passes(
    candidate: TopicCandidate,
    dossier: ResearchDossier,
    playbook: dict[str, Any],
    scale_factor: float = 1.0,
) -> list[dict[str, Any]]:
    sources_txt = _format_sources_block(dossier)
    gems = playbook.get("rhetorical_gems", {})
    gems_txt = "\n".join(
        f"• {cat}：" + " ／ ".join(f"「{g[:55]}」" for g in items[:3])
        for cat, items in gems.items()
    )

    passes: list[dict[str, Any]] = []
    for act_idx, act in enumerate(SIX_ACT_SCHEMA, 1):
        act_id = act["act_id"]
        target_chars = max(320, int(round(act["target_cjk_chars"] * scale_factor)))
        target_scenes = max(4, int(round(act["target_scenes"] * scale_factor)))
        exemplars = retrieve_act_exemplars(playbook, candidate.pillar, act_id, max_exemplars=2, max_chars_each=400)
        ex_block = "\n\n".join(
            f"【原片同幕真實逐字範本 {i+1}：《{ex['title'][:32]}》】\n「{ex['excerpt']}」"
            for i, ex in enumerate(exemplars)
        )

        extra_meta_req = ""
        if act_idx == 1:
            extra_meta_req = (
                '"title": "具強烈反差感與雙問號的獨角獸那麼叫風格正片長標題（35-58字）",\n'
                '  "seo_tags": ["標籤1", "標籤2", "標籤3", "標籤4", "標籤5", "標籤6"],\n  '
            )

        user_prompt = f"""現在請撰寫整支 13 分鐘深度長片的 **【{act['name']}】（第 {act_idx}/6 幕）**。

【本片總主題】：{candidate.proposed_title}
【核心制度矛盾】：{candidate.core_contradiction}

【本幕專屬敘事任務】：
{act['role_guide']}

【頻道原片蒸餾語氣轉折句式參考（請自然融入類似節奏，勿照抄無關主題詞彙）】：
{gems_txt}

{ex_block}

【可引用之爬蟲公開來源與數據（請挑選與本幕最相關的 S 編號深入拆解）】：
{sources_txt}

【本幕硬性規格要求】：
1. 請產出 **{target_scenes} 個場景**，每個場景的 `narration` 必須寫滿 **110～140 個繁體中文字（4～6 句口語完整推論、算術對比或實例吐槽）**，本幕合計中文字數必須 **不少於 {target_chars} 字**！
2. 本幕內請交替安排至少 2 個 `"ai_scene"`、2 個 `"evidence_doc"`、1～2 個 `"kinetic_black"`。
3. 請嚴格輸出以下合法 JSON 結構：
{{
  {extra_meta_req}"scenes": [
    {{
      "scene_id": "TEMP_1",
      "scene_mode": "ai_scene",
      "voice_style": "{"hook_question" if act_idx == 1 else "narrative_explain"}",
      "narration": "請在此寫滿 110 到 140 字的道地獨角獸那麼叫口播逐字稿...",
      "citation_id": "S1",
      "document_heading": "",
      "document_excerpt": "",
      "red_underline_quote": "",
      "kinetic_lines": [],
      "image_prompt": "Specific English photorealistic editorial scene matching this paragraph, no text"
    }}
  ]
}}"""
        passes.append(
            {
                "id": act_id,
                "system": BASE_SYSTEM_PROMPT,
                "user": user_prompt,
                "min_cjk_chars": target_chars,
                "max_new_tokens": 3600,
                "temperature": 0.56,
                "inject_prior_summary": act_idx > 1,
            }
        )

    # Pass 7: Full-Length Corresponding Traffic-Push Short (~130-150s, >= 720 CJK chars)
    short_target_chars = max(360, int(round(760 * scale_factor)))
    short_target_scenes = max(5, int(round(7 * scale_factor)))
    short_exs = playbook.get("short_exemplars", [])[:2]
    short_ex_txt = "\n\n".join(
        f"【原片推流 Short 真實逐字範本 {i+1}：《{sx['title'][:30]}》（{sx['duration_sec']}秒 / {sx['cjk_char_count']}字）】\n「{sx['full_text'][:380]}…」"
        for i, sx in enumerate(short_exs)
    )

    short_user = f"""現在請針對同一支長片主題，撰寫對應導流推流的 **【9:16 完整版 YouTube Short 腳本】（目標時長約 120～150 秒，總字數至少 {short_target_chars} 字）**。

【對應長片主題】：{candidate.proposed_title}
【建議靈魂拷問 Hook】：{candidate.short_hook_question}

{short_ex_txt}

【可引用之爬蟲公開來源】：
{sources_txt}

【推流 Short 硬性規格要求】：
1. 產出 **{short_target_scenes} 個快節奏場景**（`S01`..`S{short_target_scenes:02d}`），每段 `narration` 寫滿 **95～130 個繁體中文字**，合計中文字數 **不少於 {short_target_chars} 字**！
2. 開頭第一段直接向護航話術或荒謬現象發起靈魂拷問（如「回答我！」），中段拆解最反直覺的 2 個公開數據反差，最後一段 `S{short_target_scenes:02d}` 留下關鍵制度盲點懸念，明確引導觀眾：「至於各國法規怎麼罰、完整公文數據怎麼算，我全部整理在下方關聯正片裡，點進去一次看懂！」
3. 請嚴格輸出以下合法 JSON：
{{
  "title": "一句犀利繁中問句標題 #標籤1 #標籤2 #回答我",
  "top_hook_line1": "上方黑底第一行大字(8-12字)",
  "top_hook_line2": "第二行靈魂拷問(8-12字)!",
  "hook_highlight_words": ["關鍵詞1", "回答我!"],
  "push_cta_banner": "▶ 點擊下方連結觀看完整正片拆解",
  "scenes": [
    {{
      "scene_id": "S01",
      "scene_mode": "ai_scene",
      "voice_style": "hook_question",
      "narration": "請在此寫滿 95 到 130 字的快節奏靈魂拷問口播稿...",
      "citation_id": "S1",
      "document_heading": "",
      "document_excerpt": "",
      "red_underline_quote": "",
      "kinetic_lines": [],
      "image_prompt": "Specific English dramatic close-up scene matching this topic, no text"
    }}
  ]
}}"""

    passes.append(
        {
            "id": "short_promo_pass",
            "system": BASE_SYSTEM_PROMPT,
            "user": short_user,
            "min_cjk_chars": short_target_chars,
            "max_new_tokens": 3400,
            "temperature": 0.58,
            "inject_prior_summary": False,
        }
    )
    return passes


def _extract_json_dict(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if m:
        cleaned = m.group(1)
    else:
        s = cleaned.find("{")
        e = cleaned.rfind("}")
        if s != -1 and e != -1 and e > s:
            cleaned = cleaned[s : e + 1]
    return json.loads(cleaned)


def _execute_passes_rtx6000(passes: list[dict[str, Any]], run_dir: Path) -> list[dict[str, str]]:
    ep = RemoteWorkerEndpoint.from_file(get_runtime_endpoint_path())
    if not ep:
        raise RuntimeError("No RTX PRO 6000 endpoint configured")
    client = RTX6000WorkerClient(ep)
    req_payload = {"model": "Qwen/Qwen2.5-14B-Instruct", "passes": passes}
    local_req = run_dir / "temp" / "llm_passes_req.json"
    local_req.parent.mkdir(parents=True, exist_ok=True)
    local_req.write_text(json.dumps(req_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    remote_job_dir = f"runs/{run_dir.name}"
    client.upload_file(REPO_ROOT / "scripts/rtx6000_gpu_engine.py", "scripts/rtx6000_gpu_engine.py")
    client.upload_file(local_req, f"{remote_job_dir}/llm_passes_req.json")
    jid = client.submit_job(
        f".venv/bin/python scripts/rtx6000_gpu_engine.py --task llm_generate "
        f"--req {remote_job_dir}/llm_passes_req.json --out {remote_job_dir}/llm_passes_out.json",
        job_id=f"llm6act_{run_dir.name}",
    )
    info = client.wait_job(jid, poll_interval=4.0, timeout=1500.0)
    if info.get("rc") != 0:
        raise RuntimeError(f"RTX6000 6-Act LLM job failed: {info.get('log_tail', '')[-1800:]}")
    local_out = run_dir / "temp" / "llm_passes_out.json"
    client.download_file(f"{remote_job_dir}/llm_passes_out.json", local_out)
    return json.loads(local_out.read_text(encoding="utf-8"))["passes"]


def _execute_passes_local_or_api(passes: list[dict[str, Any]], run_dir: Path) -> list[dict[str, str]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    results: list[dict[str, str]] = []
    prior: list[str] = []

    for p in passes:
        user_msg = p["user"]
        if prior and p.get("inject_prior_summary", True):
            user_msg += "\n\n【前序章節已講重點（請勿重複）】：\n" + "\n".join(f"- {x}" for x in prior[-5:])

        if api_key:
            with httpx.Client(timeout=180.0) as client:
                r = client.post(
                    f"{api_base.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": p["system"]},
                            {"role": "user", "content": user_msg},
                        ],
                    },
                )
                r.raise_for_status()
                raw = r.json()["choices"][0]["message"]["content"]
        else:
            with httpx.Client(timeout=360.0) as client:
                r = client.post(
                    "http://127.0.0.1:11434/api/chat",
                    json={
                        "model": "qwen3:8b",
                        "stream": False,
                        "format": "json",
                        "messages": [
                            {"role": "system", "content": p["system"] + "\n/no_think"},
                            {"role": "user", "content": user_msg},
                        ],
                        "options": {"temperature": 0.55, "num_ctx": 12288, "num_predict": 3600},
                    },
                )
                r.raise_for_status()
                raw = r.json()["message"]["content"]

        try:
            obj = _extract_json_dict(raw)
            scs = obj.get("scenes", [])
            if scs:
                prior.append(f"{p['id']}: " + " / ".join(s.get("narration", "")[:26] for s in scs[:3]))
        except Exception:
            pass
        results.append({"id": p["id"], "content": raw})

    local_out = run_dir / "temp" / "llm_passes_out.json"
    local_out.parent.mkdir(parents=True, exist_ok=True)
    local_out.write_text(json.dumps({"ok": True, "passes": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def _sanitize_scene_list(
    raw_scenes: list[dict[str, Any]],
    prefix: str,
    dossier: ResearchDossier,
    is_short: bool = False,
) -> list[dict[str, Any]]:
    valid_modes = {"evidence_doc", "kinetic_black", "ai_scene"}
    valid_voices = {"hook_question", "narrative_explain", "satire_punch", "short_energetic"}
    valid_ids = {c.source_id: c for c in dossier.citations}
    cite_keys = list(valid_ids.keys())

    cleaned: list[dict[str, Any]] = []
    for idx, sc in enumerate(raw_scenes, 1):
        nar = re.sub(r"\s+", " ", str(sc.get("narration", "")).strip())
        if len(nar) < 12:
            continue
        sc["scene_id"] = f"{prefix}{idx:02d}"
        sc["narration"] = nar

        raw_mode = str(sc.get("scene_mode", "ai_scene")).lower().strip()
        if raw_mode not in valid_modes:
            if any(w in raw_mode for w in ("doc", "evidence", "news")):
                sc["scene_mode"] = "evidence_doc"
            elif any(w in raw_mode for w in ("kinetic", "black", "stat", "math")):
                sc["scene_mode"] = "kinetic_black"
            else:
                sc["scene_mode"] = "ai_scene"

        # Prevent 3 identical non-AI static cards in a row for dynamic visual pacing
        if len(cleaned) >= 2 and sc["scene_mode"] == cleaned[-1]["scene_mode"] == cleaned[-2]["scene_mode"]:
            sc["scene_mode"] = "ai_scene" if sc["scene_mode"] != "ai_scene" else "kinetic_black"

        raw_vs = str(sc.get("voice_style", "")).lower().strip()
        if raw_vs not in valid_voices:
            if idx == 1 or "hook" in raw_vs or "question" in raw_vs:
                sc["voice_style"] = "hook_question"
            elif any(w in raw_vs for w in ("satire", "punch", "irony", "roar")):
                sc["voice_style"] = "satire_punch"
            elif is_short or any(w in raw_vs for w in ("short", "fast", "energetic")):
                sc["voice_style"] = "short_energetic"
            else:
                sc["voice_style"] = "narrative_explain"

        # Rotate citations cleanly if missing or invalid so all crawled sources get featured
        cid = sc.get("citation_id")
        if cid not in valid_ids:
            cid = cite_keys[(idx - 1) % len(cite_keys)] if cite_keys else None
            sc["citation_id"] = cid

        if sc["scene_mode"] == "evidence_doc" and cid and cid in valid_ids:
            cite = valid_ids[cid]
            if not sc.get("document_heading"):
                sc["document_heading"] = f"{cite.publisher}｜{cite.title}"
            if not sc.get("document_excerpt") or sc["document_excerpt"].strip() == cite.title.strip():
                sc["document_excerpt"] = f"{cite.excerpt}（重點摘要：{nar[:65]}）"
            if not sc.get("red_underline_quote"):
                sc["red_underline_quote"] = nar[:20]

        if sc["scene_mode"] == "kinetic_black":
            kls = sc.get("kinetic_lines")
            if not isinstance(kls, list) or not kls:
                parts = [p.strip() for p in re.split(r"[，。！？!?]", nar) if p.strip()][:2]
                sc["kinetic_lines"] = [
                    {"text": pt[:18], "highlight": "", "highlight_color": "yellow"} for pt in parts
                ]
            else:
                for kl in kls:
                    if isinstance(kl, dict):
                        hc = str(kl.get("highlight_color", "yellow")).lower().strip()
                        kl["highlight_color"] = "red" if ("red" in hc or "crimson" in hc) else "yellow"

        if sc["scene_mode"] == "ai_scene" and not sc.get("image_prompt"):
            sc["image_prompt"] = (
                "Editorial documentary photograph of Taiwanese retail market and consumer products, "
                "dramatic studio lighting, shallow depth of field, photorealistic, no text"
            )

        cleaned.append(sc)
    return cleaned


def format_youtube_descriptions(
    blueprint: EpisodeBlueprint,
    dossier: ResearchDossier,
) -> tuple[str, str]:
    profile = load_channel_profile()
    cite_lines = ["【公開資訊與參考資料來源】："]
    for c in dossier.citations:
        cite_lines.append(f"• [{c.source_id}] {c.publisher}（{c.published_date}）：{c.title}\n  {c.url}")
    tags_line = " ".join(f"#{t.lstrip('#')}" for t in blueprint.long_video.seo_tags[:8])

    long_desc = (
        f"{blueprint.long_video.title}\n\n"
        + "\n".join(cite_lines)
        + f"\n\n{tags_line}\n\n"
        + profile["disclaimers"]["en"].strip()
        + "\n\n"
        + profile["disclaimers"]["zh_tw"].strip()
    )
    short_desc = (
        f"▶ 完整數據與法規拆解正片：《{blueprint.long_video.title}》\n\n"
        + "\n".join(cite_lines[:6])
        + "\n\n本影片內容僅為公共議題分析，所有資訊皆來自公開資料、政府法規與統計報導，僅供公共利益與教育用途。"
    )
    return long_desc, short_desc


def generate_episode_blueprint(
    candidate: TopicCandidate,
    dossier: ResearchDossier,
    run_dir: Path,
    scale_factor: float = 1.0,
) -> EpisodeBlueprint:
    """Generate a full 12-15 minute 6-Act Long script + 2-minute Traffic-Push Short script."""
    playbook = load_distilled_playbook()
    passes = _build_six_act_passes(candidate, dossier, playbook, scale_factor=scale_factor)

    cached_passes = run_dir / "temp" / "llm_passes_out.json"
    pass_results: list[dict[str, str]] | None = None
    if cached_passes.exists():
        try:
            pass_results = json.loads(cached_passes.read_text(encoding="utf-8")).get("passes")
        except Exception:
            pass_results = None

    if not pass_results:
        if os.environ.get("OPENAI_API_KEY"):
            pass_results = _execute_passes_local_or_api(passes, run_dir)
        else:
            try:
                pass_results = _execute_passes_rtx6000(passes, run_dir)
            except Exception:
                pass_results = _execute_passes_local_or_api(passes, run_dir)

    all_long_raw_scenes: list[dict[str, Any]] = []
    long_title = candidate.proposed_title
    seo_tags = ["獨角獸那麼叫", "台灣", "公開資訊", "法規拆解", "消費者權益"]
    short_parsed: dict[str, Any] = {}

    for pr in pass_results:
        pid = pr["id"]
        obj = _extract_json_dict(pr["content"])
        if pid == "short_promo_pass":
            short_parsed = obj
        else:
            if pid == "act_1_hook_reversal":
                if obj.get("title"):
                    long_title = str(obj["title"]).strip()
                if isinstance(obj.get("seo_tags"), list) and obj["seo_tags"]:
                    seo_tags = [str(t) for t in obj["seo_tags"]]
            all_long_raw_scenes.extend(obj.get("scenes", []))

    clean_long_scenes = _sanitize_scene_list(all_long_raw_scenes, "L", dossier, is_short=False)
    clean_short_scenes = _sanitize_scene_list(
        short_parsed.get("scenes", []), "S", dossier, is_short=True
    )

    long_cjk = len(re.findall(r"[\u4e00-\u9fff]", "".join(s["narration"] for s in clean_long_scenes)))
    short_cjk = len(re.findall(r"[\u4e00-\u9fff]", "".join(s["narration"] for s in clean_short_scenes)))

    hook_l1 = str(short_parsed.get("top_hook_line1") or candidate.short_hook_question[:11]).strip()
    hook_l2 = str(short_parsed.get("top_hook_line2") or "真相到底是什麼 回答我！").strip()
    s_title = str(short_parsed.get("title") or f"{hook_l1}，{hook_l2} #獨角獸那麼叫 #回答我").strip()
    if s_title.startswith("#"):
        s_title = f"{hook_l1}，{hook_l2} {s_title}"

    bp = EpisodeBlueprint(
        topic=candidate.proposed_title,
        pillar=candidate.pillar,
        long_cjk_chars=long_cjk,
        short_cjk_chars=short_cjk,
        long_video=LongVideoSpec.model_validate(
            {
                "title": long_title,
                "seo_tags": seo_tags,
                "scenes": clean_long_scenes,
            }
        ),
        short_video=ShortVideoSpec.model_validate(
            {
                "title": s_title,
                "top_hook_line1": hook_l1,
                "top_hook_line2": hook_l2,
                "hook_highlight_words": [
                    str(w) for w in (short_parsed.get("hook_highlight_words") or ["回答我!"])
                ],
                "push_cta_banner": str(
                    short_parsed.get("push_cta_banner") or "▶ 點擊下方連結觀看完整正片拆解"
                ),
                "scenes": clean_short_scenes,
            }
        ),
    )
    bp.long_description, bp.short_description = format_youtube_descriptions(bp, dossier)
    return bp
