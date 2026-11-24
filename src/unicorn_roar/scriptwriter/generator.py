"""Dual-Track AI Scriptwriter for 獨角獸那麼叫 (@unicorn-roar).

Generates a complete, source-grounded EpisodeBlueprint containing:
1. 16:9 Long-form analytical script (with `evidence_doc`, `kinetic_black`, and `ai_scene` visual cues + source IDs)
2. 9:16 Corresponding Traffic-Push Short script ("回答我!" hook -> core shock -> push to full Long video)
3. Full YouTube descriptions with cited public URLs and bilingual channel disclaimers.

Compute priority:
1. OpenAI-compatible / Gemini API if active key is configured
2. Colab RTX PRO 6000 PyTorch `Qwen/Qwen2.5-14B-Instruct` (bfloat16 CUDA)
3. Local Ollama (`qwen3:8b`) fallback
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from unicorn_roar.config import get_runtime_endpoint_path, load_channel_profile
from unicorn_roar.discovery.topic_radar import TopicCandidate
from unicorn_roar.discovery.web_crawler import ResearchDossier
from unicorn_roar.remote.rtx6000_client import (
    RTX6000WorkerClient,
    RemoteWorkerEndpoint,
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
    long_video: LongVideoSpec
    short_video: ShortVideoSpec
    long_description: str = ""
    short_description: str = ""


SYSTEM_PROMPT = """你是台灣知名百萬觀看級知識評論 YouTube 頻道「獨角獸那麼叫 (@unicorn-roar)」的首席總編劇。
你的任務是根據「爬蟲抓回來的真實公開新聞、政府法規與統計來源」，同時產出一支【16:9 深度長片腳本】與一支【9:16 對應導流 Short 腳本】。

### 頻道靈魂與語氣鐵律（必須100%還原）：
1. **繁體中文（台灣慣用語）**：使用「主計總處、食藥署、消基會、超商、盤子、話術、統計口徑、約定俗成」，自然口語且節奏犀利。
2. **長片開場前兩段（Hook）**：第一句直接拋出荒謬的數字反差或標示矛盾（參考真片原句風格：「白麵粉占90%竟也能叫全麥？」「126萬人算低薪，換個標準卻變成0人？你沒聽錯！難道是薛丁格的低薪嗎？」）。
3. **吐槽與硬核證據交替**：幽默反諷（「好樣的」、「你先別急著生氣啊」、「這標題是寫給誰看的啊」）之後，立刻切入「認真來說」，引用爬蟲提供的真實來源編號（如 S1、S2、S3）拆解法規門檻、計算公式或各國制度對比。
4. **三種視覺場景嚴格交替編排 (`scene_mode`)**：
   - `"ai_scene"`：寫實電影感或諷刺隱喻畫面，`image_prompt` 必須用英文撰寫具體場景（禁止在圖片 prompt 要求印出中文字，例如："Cinematic close-up of colorful fruit juice cartons on a modern Taiwanese convenience store shelf, dramatic moody lighting, photorealistic, 8k"）。
   - `"evidence_doc"`：白底公文/法規/新聞證據卡，必須綁定真實 `citation_id`（如 `"S1"`），提供 `document_heading`（公文或報導標題）、`document_excerpt`（60-110字內文摘要）以及 `red_underline_quote`（必須是 `document_excerpt` 中最關鍵的一句話，12-26字，系統會自動畫上紅線並於左上角標註來源與 accessed 日期！）。
   - `"kinetic_black"`：純黑底大字公式或對比圖卡，提供 2 到 3 行 `kinetic_lines`，每行含 `text`（如 `"標示門檻: 原汁 10%"`）與要變色的關鍵詞 `highlight`（如 `"10%"`）及 `highlight_color`（`"yellow"` 或 `"red"`）。
5. **對應推流 Short 鐵律**：
   - `top_hook_line1` 與 `top_hook_line2` 每行 7~13 個繁體中文字，極具視覺衝擊力（例如：「包裝畫滿整顆水果」「結果原汁只有10%？回答我！」）。
   - Short 共 5~6 個緊湊場景，最後一個場景 (`S05` 或 `S06`) 的旁白必須留下一個關鍵制度懸念並明確引導觀眾：「想知道背後完整的法規漏洞跟各國標準對比，直接點下方關聯正片，我們一次拆給你聽！」

請嚴格只輸出合法 JSON，不要加任何 markdown 註解或額外說明文字。"""


def _build_user_prompt(
    candidate: TopicCandidate,
    dossier: ResearchDossier,
    target_long_scenes: int = 10,
    target_short_scenes: int = 5,
) -> str:
    sources_block = []
    for c in dossier.citations:
        sources_block.append(
            f"- [{c.source_id}] 機構/媒體：{c.publisher}｜日期：{c.published_date}\n"
            f"  標題：{c.title}\n"
            f"  摘要：{c.excerpt}\n"
            f"  URL：{c.url}"
        )
    bg_block = "\n".join(dossier.background_notes) if dossier.background_notes else "（無額外百科條目）"

    schema_hint = {
        "long_video": {
            "title": "犀利雙問號標題（約30-55字，含核心關鍵字）",
            "seo_tags": ["關鍵字1", "關鍵字2", "關鍵字3", "關鍵字4", "關鍵字5"],
            "scenes": [
                {
                    "scene_id": "L01",
                    "scene_mode": "ai_scene",
                    "voice_style": "hook_question",
                    "narration": "約55-85字的繁體中文口語旁白，直接拋出反差痛點...",
                    "citation_id": "S1",
                    "document_heading": "",
                    "document_excerpt": "",
                    "red_underline_quote": "",
                    "kinetic_lines": [],
                    "image_prompt": "Detailed English photorealistic scene description without text...",
                }
            ],
        },
        "short_video": {
            "title": "Short 標題含 #標籤1 #標籤2 #回答我",
            "top_hook_line1": "第一行醒目大字(8-12字)",
            "top_hook_line2": "第二行靈魂拷問(8-12字)!",
            "hook_highlight_words": ["關鍵詞1", "回答我!"],
            "push_cta_banner": "▶ 點擊下方連結觀看完整正片拆解",
            "scenes": [
                {
                    "scene_id": "S01",
                    "scene_mode": "ai_scene",
                    "voice_style": "hook_question",
                    "narration": "約30-45字快節奏開場...",
                    "citation_id": "S1",
                    "document_heading": "",
                    "document_excerpt": "",
                    "red_underline_quote": "",
                    "kinetic_lines": [],
                    "image_prompt": "Detailed English vertical/square visual scene...",
                }
            ],
        },
    }

    return f"""請針對以下選定主題與爬蟲公開資料，撰寫「獨角獸那麼叫」雙軌腳本：

【選定主題】：{candidate.proposed_title}
【核心矛盾】：{candidate.core_contradiction}
【建議 Short 靈魂拷問】：{candidate.short_hook_question}

【爬蟲實時抓取之公開資訊與法規來源（請務必在腳本中引用至少 4 個不同 S 編號，且數字與事實須忠於來源）】：
{chr(10).join(sources_block)}

【背景知識補充】：
{bg_block}

【結構與場景數量要求】：
1. `long_video.scenes` 請精準產出 **{target_long_scenes} 個場景**（編號 `L01` 到 `L{target_long_scenes:02d}`），每段 `narration` 約 60~95 個繁體中文字（總字數約 700~950 字，語速自然飽滿）。
   - 至少包含 3 個 `"evidence_doc"` 場景（綁定真實 `citation_id` 並填寫 `document_heading`, `document_excerpt`, `red_underline_quote`）。
   - 至少包含 3 個 `"kinetic_black"` 場景（填寫 2~3 行 `kinetic_lines` 呈現數字換算或標準對比）。
   - 至少包含 4 個 `"ai_scene"` 場景（填寫高品質英文 `image_prompt`）。
   - 開場 `L01` 用 `voice_style: "hook_question"`，吐槽金句段用 `"satire_punch"`，分析段用 `"narrative_explain"`。
2. `short_video.scenes` 請精準產出 **{target_short_scenes} 個場景**（編號 `S01` 到 `S{target_short_scenes:02d}`），每段 `narration` 約 32~50 個繁體中文字（總長約 55~75 秒）。
   - 包含至少 1 個 `"evidence_doc"`、1 個 `"kinetic_black"`、以及 2~3 個 `"ai_scene"`。
   - 最後一段 `S{target_short_scenes:02d}` 必須明確推流引導點擊下方關聯長片！

【輸出 JSON 格式範例】：
{json.dumps(schema_hint, ensure_ascii=False, indent=2)}
"""


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


def _call_rtx6000_llm(system: str, user: str, run_dir: Path) -> str:
    ep = RemoteWorkerEndpoint.from_file(get_runtime_endpoint_path())
    if not ep:
        raise RuntimeError("No RTX PRO 6000 endpoint configured")
    client = RTX6000WorkerClient(ep)
    req_payload = {
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "system": system,
        "user": user,
        "max_new_tokens": 5200,
        "temperature": 0.5,
    }
    local_req = run_dir / "temp" / "llm_req.json"
    local_req.parent.mkdir(parents=True, exist_ok=True)
    local_req.write_text(json.dumps(req_payload, ensure_ascii=False), encoding="utf-8")

    remote_job_dir = f"runs/{run_dir.name}"
    client.upload_file(Path(__file__).resolve().parents[3] / "scripts/rtx6000_gpu_engine.py", "scripts/rtx6000_gpu_engine.py")
    client.upload_file(local_req, f"{remote_job_dir}/llm_req.json")
    jid = client.submit_job(
        f".venv/bin/python scripts/rtx6000_gpu_engine.py --task llm_generate "
        f"--req {remote_job_dir}/llm_req.json --out {remote_job_dir}/llm_out.json",
        job_id=f"llm_{run_dir.name}",
    )
    info = client.wait_job(jid, poll_interval=3.0, timeout=600.0)
    if info.get("rc") != 0:
        raise RuntimeError(f"RTX6000 LLM job failed: {info.get('log_tail', '')[-1200:]}")
    local_out = run_dir / "temp" / "llm_out.json"
    client.download_file(f"{remote_job_dir}/llm_out.json", local_out)
    return json.loads(local_out.read_text(encoding="utf-8"))["content"]


def _call_local_ollama(system: str, user: str) -> str:
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": "qwen3:8b",
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system + "\n/no_think"},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": 0.5, "num_ctx": 12288, "num_predict": 5000},
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


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
        + "\n".join(cite_lines[:5])
        + "\n\n本影片內容僅為公共議題分析，所有資訊皆來自公開資料、政府法規與統計報導，僅供公共利益與教育用途。"
    )
    return long_desc, short_desc


def generate_episode_blueprint(
    candidate: TopicCandidate,
    dossier: ResearchDossier,
    run_dir: Path,
    target_long_scenes: int = 10,
    target_short_scenes: int = 5,
) -> EpisodeBlueprint:
    """Generate and validate the complete Long + Short script blueprint."""
    user_prompt = _build_user_prompt(
        candidate,
        dossier,
        target_long_scenes=target_long_scenes,
        target_short_scenes=target_short_scenes,
    )

    raw_text: str | None = None
    last_err: Exception | None = None

    # 1. Try OpenAI-compatible env API if user configured OPENAI_API_KEY
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if api_key:
        try:
            with httpx.Client(timeout=120.0) as client:
                r = client.post(
                    f"{api_base.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                )
                r.raise_for_status()
                raw_text = r.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            last_err = exc

    # 2. Primary Heavy Node: Colab RTX PRO 6000 (Qwen2.5-14B-Instruct bfloat16)
    if raw_text is None:
        try:
            raw_text = _call_rtx6000_llm(SYSTEM_PROMPT, user_prompt, run_dir)
        except Exception as exc:
            last_err = exc

    # 3. Local Fallback: Ollama qwen3:8b
    if raw_text is None:
        try:
            raw_text = _call_local_ollama(SYSTEM_PROMPT, user_prompt)
        except Exception as exc:
            raise RuntimeError(f"All LLM backends failed; last={exc}, prev={last_err}") from exc

    parsed = _extract_json_dict(raw_text)
    valid_ids = {c.source_id: c for c in dossier.citations}
    default_cid = dossier.citations[0].source_id if dossier.citations else None

    # Sanitize & guarantee citation bindings on evidence_doc scenes
    for track_key in ("long_video", "short_video"):
        scenes = parsed.get(track_key, {}).get("scenes", [])
        for sc in scenes:
            cid = sc.get("citation_id")
            if cid not in valid_ids:
                sc["citation_id"] = default_cid
                cid = default_cid
            if sc.get("scene_mode") == "evidence_doc" and cid and cid in valid_ids:
                cite = valid_ids[cid]
                if not sc.get("document_heading"):
                    sc["document_heading"] = f"{cite.publisher}｜{cite.title}"
                if not sc.get("document_excerpt"):
                    sc["document_excerpt"] = cite.excerpt
                if not sc.get("red_underline_quote"):
                    sc["red_underline_quote"] = cite.excerpt[:22]
            if sc.get("scene_mode") == "kinetic_black" and not sc.get("kinetic_lines"):
                sc["kinetic_lines"] = [
                    {"text": sc.get("narration", "")[:14], "highlight": "", "highlight_color": "yellow"}
                ]
            if sc.get("scene_mode") == "ai_scene" and not sc.get("image_prompt"):
                sc["image_prompt"] = (
                    "Cinematic editorial photography in a modern Taiwanese supermarket aisle, "
                    "dramatic lighting, shallow depth of field, ultra detailed, photorealistic, no text"
                )

    bp = EpisodeBlueprint(
        topic=candidate.proposed_title,
        pillar=candidate.pillar,
        long_video=LongVideoSpec.model_validate(parsed["long_video"]),
        short_video=ShortVideoSpec.model_validate(parsed["short_video"]),
    )
    bp.long_description, bp.short_description = format_youtube_descriptions(bp, dossier)
    return bp
