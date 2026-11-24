"""V2 Distilled 6-Act + Showrunner-Polished Scriptwriter for 獨角獸那麼叫 (@unicorn-roar).

V2 Upgrades:
1. Powered by `unsloth/Qwen2.5-32B-Instruct-bnb-4bit` (32B parameters) on RTX PRO 6000.
2. Feeds full multi-paragraph crawled news & statutory text (`citation.paragraphs`) and strictly
   forbids inventing unverified numbers or contradictory regulations.
3. Choreographs 7 authentic @unicorn-roar visual scene modes:
   - `news_portal`        : Real Taiwanese news webpage + 3-beat punch-in zoom & red underline
   - `official_gov_doc`   : Formal ROC Government `函 / 公告` A4 document + 3-beat zoom & red underline
   - `gov_web_portal`     : Official `.gov.tw` ministry portal + 3-beat zoom & red underline
   - `product_comparison` : 2-beat split comparison card (`包裝宣稱 vs 實際真相` + red delta badge)
   - `netizen_comments`   : 3-beat progressive Taiwanese netizen comment wall
   - `kinetic_black`      : Multi-beat progressive `#000000` yellow/red math & scale-bar slam card
   - `ai_scene`           : Documentary photography / 24-step SDXL + 3-beat multi-crop fast cuts
4. Pass 8 (`showrunner_polish_pass`): Rewrites Long & Short titles and opening hooks to match
   @unicorn-roar's exact viral multi-question style and strips any textbook/AI phrasing.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import httpx
from opencc import OpenCC
from pydantic import BaseModel, Field

_CC_ZHTW = OpenCC("s2twp")


def _ensure_zhtw(val: Any) -> Any:
    if isinstance(val, str):
        return _CC_ZHTW.convert(val)
    if isinstance(val, list):
        return [_ensure_zhtw(x) for x in val]
    if isinstance(val, dict):
        return {k: _ensure_zhtw(v) for k, v in val.items()}
    return val

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

VisualMode = Literal[
    "news_portal",
    "official_gov_doc",
    "gov_web_portal",
    "product_comparison",
    "netizen_comments",
    "kinetic_black",
    "ai_scene",
]


class KineticLine(BaseModel):
    text: str
    highlight: str = ""
    highlight_color: Literal["yellow", "red"] = "yellow"


class ComparisonSpec(BaseModel):
    heading: str = ""
    left_title: str = "表面宣稱 / 調整前"
    left_points: list[str] = Field(default_factory=list)
    right_title: str = "實際真相 / 調整後"
    right_points: list[str] = Field(default_factory=list)
    delta_badge: str = "資訊落差"


class CommentItem(BaseModel):
    user: str = "@tw_consumer"
    time: str = "3 小時前"
    text: str = ""
    likes: int = 246


class SceneSpec(BaseModel):
    scene_id: str
    scene_mode: VisualMode
    voice_style: Literal["hook_question", "narrative_explain", "satire_punch", "short_energetic"] = "narrative_explain"
    narration: str
    citation_id: str | None = None
    document_heading: str = ""
    document_paragraphs: list[str] = Field(default_factory=list)
    red_underline_quote: str = ""
    kinetic_lines: list[KineticLine] = Field(default_factory=list)
    comparison: ComparisonSpec | None = None
    comments: list[CommentItem] = Field(default_factory=list)
    image_prompt: str = ""
    wikimedia_query: str = ""


class LongVideoSpec(BaseModel):
    title: str
    seo_tags: list[str] = Field(default_factory=list)
    scenes: list[SceneSpec]


class ShortVideoSpec(BaseModel):
    title: str
    top_hook_line1: str
    top_hook_line2: str
    hook_line2_color: Literal["red", "yellow", "white"] = "red"
    hook_highlight_words: list[str] = Field(default_factory=list)
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


AI_CLICHE_REPLACEMENTS = [
    ("你知道嗎？", ""),
    ("你知道嗎，", ""),
    ("你可能會問，", "問題來了，"),
    ("真是讓人頭疼", "這才是最荒謬的地方"),
    ("消費者權益岌岌可危", "最後買單的永遠是消費者"),
    ("總結來說，", "說穿了，"),
    ("綜上所述，", "講白了，"),
]

BASE_SYSTEM_PROMPT = """你是台灣百萬觀看級深度消費與制度拆解頻道「獨角獸那麼叫 (@unicorn-roar)」本尊總編劇。

### 絕對禁止違反之鐵律：
1. **100% 事實與數據忠誠度**：所有引用的百分比、法規條文、金額與日期，**必須完全來自下方提供的【爬蟲真實公開全文資料】**！若爬蟲資料未提及某個數字，絕對禁止自行編造任何「±10%」「七成商品」等假統計！
2. **純血「獨角獸那麼叫」口語氣口（拒絕教科書 AI 腔）**：
   - 嚴禁出現「你知道嗎？」「你可能會問」「總結來說」「讓人頭疼」「岌岌可危」等生硬翻譯腔或說教套話！
   - 開場第一句必須像原片一樣直接切入具體消費現場算帳或新聞反差（例如：「星巴克大杯拿鐵，去冰補滿以前免費，現在補滿要多收15元，不想多付錢結果拿到手只剩三分之二杯！」「蒲燒鯛魚漲5元，你生氣嗎？你先別急著生氣啊！」）。
   - 善用原片招牌節奏：「你沒聽錯！」「好樣的」「問題來了」「好吧認真來說」「這標題是寫給誰看的啊」「難道……嗎？」。
3. **豐富視覺場景編排（7 種原片實景模式交替使用，嚴禁單調重複）**：
   - `"news_portal"`：真實新聞網站截圖 + 放大紅線畫記（綁定 `citation_id`，提供 `red_underline_quote` 擷取來源真句 10-22 字）。
   - `"official_gov_doc"`：中華民國政府機關正式公文（函／令）A4 截圖 + 放大紅線畫記（適合食藥署、農業部、主計總處、公平會法規條文）。
   - `"gov_web_portal"`：政府官網公告專區頁面 + 放大紅線畫記。
   - `"product_comparison"`：雙欄對照圖卡（填寫 `comparison`: `{heading, left_title, left_points:[2-3點], right_title, right_points:[2-3點], delta_badge:"如 ↗ 變相漲價+18%"}`）。
   - `"netizen_comments"`：台灣網友社群真實反應留言牆（當旁白提到網友反應、輿論炸鍋或護航說法時使用，填寫 `comments`: 4 則道地台灣網友留言 `{user, time, text, likes}`）。
   - `"kinetic_black"`：純黑底黃紅高對比算術／排名圖卡（填寫 2-3 行 `kinetic_lines`: `{text, highlight, highlight_color: "yellow"|"red"}`，其中 `highlight` 必須是 `text` 裡要變色的子字串，例如 `"15元"`，不可填 boolean！）。
   - `"ai_scene"`：寫實情境攝影畫面（填寫具體英文 `image_prompt`，要求無文字乾淨構圖 `clean unbranded surface, no text, no letters`，並填 1-2 個英文實體名詞於 `wikimedia_query` 如 `"olive oil bottle"` 或 `"supermarket aisle"`）。
4. 只輸出合法 JSON，不加任何額外文字。"""


def _format_fulltext_sources(dossier: ResearchDossier) -> str:
    blocks = []
    for c in dossier.citations:
        body = "\n    ".join(c.paragraphs[:4]) if c.paragraphs else c.excerpt
        blocks.append(
            f"【{c.source_id}】機構/媒體：{c.publisher}｜日期：{c.published_date}｜標題：{c.title}\n"
            f"    內文實錄：{body}"
        )
    if dossier.background_notes:
        blocks.append("\n【制度與法規百科全文補充】：\n" + "\n".join(dossier.background_notes[:3]))
    return "\n\n".join(blocks)


def _build_six_act_passes(
    candidate: TopicCandidate,
    dossier: ResearchDossier,
    playbook: dict[str, Any],
    scale_factor: float = 0.82,
) -> list[dict[str, Any]]:
    sources_txt = _format_fulltext_sources(dossier)
    gems = playbook.get("rhetorical_gems", {})
    gems_txt = "\n".join(
        f"• {cat}：" + " ／ ".join(f"「{g[:58]}」" for g in items[:3])
        for cat, items in gems.items()
    )

    # Mode choreography recommendations per Act so all 7 visual modes appear authentically
    act_visual_mix = [
        '必須包含 `"ai_scene"`, `"news_portal"`, `"netizen_comments"`, `"product_comparison"`',
        '必須包含 `"official_gov_doc"`, `"kinetic_black"`, `"gov_web_portal"`, `"ai_scene"`',
        '必須包含 `"product_comparison"`, `"news_portal"`, `"kinetic_black"`, `"ai_scene"`',
        '必須包含 `"gov_web_portal"`, `"product_comparison"`, `"news_portal"`, `"ai_scene"`',
        '必須包含 `"netizen_comments"`, `"kinetic_black"`, `"official_gov_doc"`, `"ai_scene"`',
        '必須包含 `"product_comparison"`, `"kinetic_black"`, `"ai_scene"`',
    ]

    passes: list[dict[str, Any]] = []
    for act_idx, act in enumerate(SIX_ACT_SCHEMA, 1):
        act_id = act["act_id"]
        target_chars = max(340, int(round(act["target_cjk_chars"] * scale_factor)))
        target_scenes = max(4, int(round(act["target_scenes"] * scale_factor)))
        exemplars = retrieve_act_exemplars(playbook, candidate.pillar, act_id, max_exemplars=2, max_chars_each=420)
        ex_block = "\n\n".join(
            f"【原片同幕真實逐字範本 {i+1}：《{ex['title'][:34]}》】\n「{ex['excerpt']}」"
            for i, ex in enumerate(exemplars)
        )

        user_prompt = f"""請撰寫 13 分鐘正片的 **【{act['name']}】（第 {act_idx}/6 幕）**。

【本片總主題】：{candidate.proposed_title}
【核心制度矛盾】：{candidate.core_contradiction}
【本幕敘事任務】：{act['role_guide']}
【本幕視覺分鏡要求】：{act_visual_mix[act_idx - 1]}

【頻道原片蒸餾語氣參考】：
{gems_txt}

{ex_block}

【爬蟲抓取之真實公開新聞與法規內文（所有數據與條文必須嚴格取自此處，禁止捏造數字）】：
{sources_txt}

【輸出規格】：
產出 **{target_scenes} 個場景**，每段 `narration` 寫滿 **105～135 個繁體中文字**（合計不少於 {target_chars} 字）。
嚴格輸出以下合法 JSON：
{{
  "scenes": [
    {{
      "scene_id": "TEMP_1",
      "scene_mode": "news_portal",
      "voice_style": "{"hook_question" if act_idx == 1 else "narrative_explain"}",
      "narration": "105到135字道地口播稿，直接講具體事實與算術反差...",
      "citation_id": "S1",
      "document_heading": "新聞或公文標題",
      "red_underline_quote": "擷取來源內文中要畫紅線的關鍵原句(12-22字)",
      "kinetic_lines": [{{"text": "對比文字: 15元", "highlight": "15元", "highlight_color": "yellow"}}],
      "comparison": {{"heading": "對比標題", "left_title": "表面宣稱", "left_points": ["重點1", "重點2"], "right_title": "實際真相", "right_points": ["真相1", "真相2"], "delta_badge": "價差關鍵"}},
      "comments": [{{"user": "@tw_user1", "time": "2小時前", "text": "道地台灣網友吐槽留言", "likes": 320}}],
      "image_prompt": "Photorealistic editorial shot matching this scene, clean unbranded surface, no text, no letters",
      "wikimedia_query": "english noun"
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
                "temperature": 0.50,
                "inject_prior_summary": act_idx > 1,
            }
        )

    # Pass 7: Corresponding Traffic-Push Short (~120-150s, ~580 CJK chars)
    short_target_chars = min(620, max(420, int(round(580 * scale_factor))))
    short_exs = playbook.get("short_exemplars", [])[:2]
    short_ex_txt = "\n\n".join(
        f"【原片推流 Short 真實範本 {i+1}：《{sx['title'][:32]}》】\n「{sx['full_text'][:380]}…」"
        for i, sx in enumerate(short_exs)
    )
    short_user = f"""請撰寫對應導流至上述長片的 **【9:16 推流 YouTube Short 腳本】（目標約 115～145 秒，總字數不少於 {short_target_chars} 字）**。

【對應長片主題】：{candidate.proposed_title}
【靈魂拷問方向】：{candidate.short_hook_question}

{short_ex_txt}

【爬蟲真實公開來源】：
{sources_txt}

【輸出規格】：
1. 產出 **6 個快節奏場景**（包含 `"ai_scene"`, `"news_portal"`, `"product_comparison"`, `"netizen_comments"`, `"kinetic_black"`），每段 `narration` 90～115 字。
2. 開頭第一句直接發起犀利反問或「回答我！」拷問；最後一段留下更深層的法規或跨國對比懸念，自然引導觀眾點擊下方關聯正片觀看完整拆解。
3. `top_hook_line1`（7-10字，亮黃色）與 `top_hook_line2`（6-10字，鮮紅色震撼短句，如「結果只有10%？」「完全合法不用罰！」）。
嚴格輸出合法 JSON：
{{
  "title": "犀利雙問句加驚嘆號 Short 標題 #標籤1 #標籤2 #回答我",
  "top_hook_line1": "7到10字醒目上半句",
  "top_hook_line2": "6到10字震撼下半句!",
  "hook_line2_color": "red",
  "hook_highlight_words": ["關鍵字"],
  "scenes": [ ... ]
}}"""
    passes.append(
        {
            "id": "short_promo_pass",
            "system": BASE_SYSTEM_PROMPT,
            "user": short_user,
            "min_cjk_chars": short_target_chars,
            "max_new_tokens": 3200,
            "temperature": 0.52,
            "inject_prior_summary": False,
        }
    )

    # Pass 8: Showrunner Viral Title & Hook Critic Pass (referencing real 300k-700k view @unicorn-roar titles!)
    real_titles = [e["title"] for e in playbook.get("long_exemplars", [])[:8]]
    titles_ref = "\n".join(f"- {t}" for t in real_titles)
    critic_user = f"""請擔任「獨角獸那麼叫 (@unicorn-roar)」頻道主理人，為本集正片與推流 Short 敲定最終【高點閱靈魂標題】與【Short 頂部雙行大字】。

【頻道真實 30萬～70萬觀看代表作標題風格參考】：
{titles_ref}

【本集主題】：{candidate.proposed_title}
【核心矛盾】：{candidate.core_contradiction}

請注意原片標題特徵：
- 長度 38～62 字，由 **2～3 個連續犀利反問句** 組成，直接點出具體數字／制度矛盾／業界話術，絕不用「淺談」「價格遊戲」「大解密」這種平庸詞！
- Short 頂部大字 `top_hook_line1`（7～9字）+ `top_hook_line2`（6～9字），上下兩行形成強烈反差（參考原片：「台灣沒有麵包 / 只有狠活」「說吳郭魚是鯛魚 / 的人 回答我!」）。

請只輸出合法 JSON：
{{
  "long_title": "40-62字獨角獸那麼叫真跡級正片標題",
  "seo_tags": ["關鍵字1", "關鍵字2", "關鍵字3", "關鍵字4", "關鍵字5", "關鍵字6"],
  "short_title": "犀利問句 Short 標題 #標籤1 #標籤2 #獨角獸那麼叫",
  "top_hook_line1": "7-9字第一行亮黃大字",
  "top_hook_line2": "6-9字第二行鮮紅重擊大字",
  "hook_line2_color": "red"
}}"""
    passes.append(
        {
            "id": "showrunner_critic_pass",
            "system": BASE_SYSTEM_PROMPT,
            "user": critic_user,
            "min_cjk_chars": 0,
            "max_new_tokens": 800,
            "temperature": 0.55,
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
    req_payload = {"model": "unsloth/Qwen2.5-32B-Instruct-bnb-4bit", "passes": passes}
    local_req = run_dir / "temp" / "llm_passes_req.json"
    local_req.parent.mkdir(parents=True, exist_ok=True)
    local_req.write_text(json.dumps(req_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    remote_job_dir = f"runs/{run_dir.name}"
    client.upload_file(REPO_ROOT / "scripts/rtx6000_gpu_engine.py", "scripts/rtx6000_gpu_engine.py")
    client.upload_file(local_req, f"{remote_job_dir}/llm_passes_req.json")
    jid = client.submit_job(
        f".venv/bin/python scripts/rtx6000_gpu_engine.py --task llm_generate "
        f"--req {remote_job_dir}/llm_passes_req.json --out {remote_job_dir}/llm_passes_out.json",
        job_id=f"llm32b_{run_dir.name}",
    )
    info = client.wait_job(jid, poll_interval=4.0, timeout=1800.0)
    if info.get("rc") != 0:
        raise RuntimeError(f"RTX6000 32B LLM job failed: {info.get('log_tail', '')[-1800:]}")
    local_out = run_dir / "temp" / "llm_passes_out.json"
    client.download_file(f"{remote_job_dir}/llm_passes_out.json", local_out)
    return json.loads(local_out.read_text(encoding="utf-8"))["passes"]


def _execute_passes_local_or_api(passes: list[dict[str, Any]], run_dir: Path) -> list[dict[str, str]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    results: list[dict[str, str]] = []
    for p in passes:
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
                            {"role": "user", "content": p["user"]},
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
                        "think": False,
                        "format": "json",
                        "messages": [
                            {"role": "system", "content": p["system"]},
                            {"role": "user", "content": p["user"]},
                        ],
                        "options": {"temperature": 0.5, "num_ctx": 8192, "num_predict": 2800},
                    },
                )
                r.raise_for_status()
                raw = r.json()["message"]["content"]
        results.append({"id": p["id"], "content": raw})
    local_out = run_dir / "temp" / "llm_passes_out.json"
    local_out.parent.mkdir(parents=True, exist_ok=True)
    local_out.write_text(json.dumps({"ok": True, "passes": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def _strip_ai_cliches(text: str) -> str:
    out = _CC_ZHTW.convert(text)
    for bad, rep in AI_CLICHE_REPLACEMENTS:
        out = out.replace(bad, rep)
    return re.sub(r"\s+", " ", out).strip()


def _sanitize_scene_list(
    raw_scenes: list[dict[str, Any]],
    prefix: str,
    dossier: ResearchDossier,
    is_short: bool = False,
) -> list[dict[str, Any]]:
    valid_modes = {
        "news_portal",
        "official_gov_doc",
        "gov_web_portal",
        "product_comparison",
        "netizen_comments",
        "kinetic_black",
        "ai_scene",
    }
    valid_voices = {"hook_question", "narrative_explain", "satire_punch", "short_energetic"}
    valid_ids = {c.source_id: c for c in dossier.citations}
    cite_keys = list(valid_ids.keys())

    cleaned: list[dict[str, Any]] = []
    seen_ngrams_list: list[set[str]] = []
    for raw_sc in raw_scenes:
        sc = dict(raw_sc)
        nar = _strip_ai_cliches(str(sc.get("narration") or ""))
        if len(nar) < 14:
            continue
        cjk_only = "".join(re.findall(r"[\u4e00-\u9fff0-9]", nar))
        ngrams = {cjk_only[k : k + 5] for k in range(max(1, len(cjk_only) - 4))}
        if any(
            len(ngrams & prev) / max(1, min(len(ngrams), len(prev))) > 0.50
            for prev in seen_ngrams_list
        ):
            continue
        seen_ngrams_list.append(ngrams)
        idx = len(cleaned) + 1
        sc["scene_id"] = f"{prefix}{idx:02d}"
        sc["narration"] = nar

        raw_mode = str(sc.get("scene_mode") or "ai_scene").lower().strip()
        if raw_mode == "evidence_doc":
            raw_mode = "official_gov_doc" if idx % 2 == 0 else "news_portal"
        if raw_mode not in valid_modes:
            if "gov" in raw_mode or "doc" in raw_mode or "law" in raw_mode:
                raw_mode = "official_gov_doc"
            elif "portal" in raw_mode:
                raw_mode = "gov_web_portal"
            elif "news" in raw_mode or "evidence" in raw_mode:
                raw_mode = "news_portal"
            elif "comp" in raw_mode or "vs" in raw_mode:
                raw_mode = "product_comparison"
            elif "comment" in raw_mode or "netizen" in raw_mode:
                raw_mode = "netizen_comments"
            elif "kinetic" in raw_mode or "black" in raw_mode or "math" in raw_mode:
                raw_mode = "kinetic_black"
            else:
                raw_mode = "ai_scene"

        # Prevent repeating the exact same visual template back-to-back (and cap consecutive `ai_scene` at 2)
        if cleaned and raw_mode == cleaned[-1]["scene_mode"] and raw_mode != "ai_scene":
            raw_mode = "ai_scene" if idx % 2 == 1 else "product_comparison"
        elif (
            raw_mode == "ai_scene"
            and len(cleaned) >= (1 if is_short else 2)
            and all(prev_sc["scene_mode"] == "ai_scene" for prev_sc in cleaned[-(1 if is_short else 2) :])
        ):
            cycle_modes = ["news_portal", "product_comparison", "netizen_comments", "kinetic_black", "official_gov_doc"]
            raw_mode = cycle_modes[(idx - 1) % len(cycle_modes)]
        sc["scene_mode"] = raw_mode

        raw_vs = str(sc.get("voice_style") or "").lower().strip()
        if raw_vs not in valid_voices:
            if idx == 1 or "hook" in raw_vs or "question" in raw_vs:
                sc["voice_style"] = "hook_question"
            elif any(w in raw_vs for w in ("satire", "punch", "irony", "roar")):
                sc["voice_style"] = "satire_punch"
            elif is_short or any(w in raw_vs for w in ("short", "fast", "energetic")):
                sc["voice_style"] = "short_energetic"
            else:
                sc["voice_style"] = "narrative_explain"

        cid = sc.get("citation_id")
        if cid not in valid_ids:
            cid = cite_keys[(idx - 1) % len(cite_keys)] if cite_keys else None
        sc["citation_id"] = cid
        cite = valid_ids.get(cid) if cid else None

        for str_field in ("document_heading", "red_underline_quote", "image_prompt", "wikimedia_query"):
            val = sc.get(str_field)
            sc[str_field] = str(val).strip() if isinstance(val, (str, int, float)) else ""

        # Bind REAL crawled multi-paragraph text to news/gov cards (never AI-repeated titles!)
        if raw_mode in ("news_portal", "official_gov_doc", "gov_web_portal"):
            if cite:
                if not sc["document_heading"]:
                    sc["document_heading"] = cite.title
                real_paras = list(cite.paragraphs[:4]) if cite.paragraphs else [cite.excerpt]
                if len(real_paras) == 1 and len(real_paras[0]) < 80:
                    real_paras.append(nar)
                sc["document_paragraphs"] = real_paras
                if not sc["red_underline_quote"]:
                    sc["red_underline_quote"] = real_paras[0][:20]
            else:
                sc["document_paragraphs"] = [nar]
        else:
            sc["document_paragraphs"] = []

        # Normalize `kinetic_lines`
        if raw_mode != "kinetic_black":
            sc["kinetic_lines"] = []
        else:
            kls = sc.get("kinetic_lines")
            norm_kls: list[dict[str, str]] = []
            if isinstance(kls, list):
                for kl in kls:
                    if isinstance(kl, dict):
                        k_txt = str(kl.get("text") or "").strip()
                        if not k_txt:
                            continue
                        raw_hl = kl.get("highlight")
                        if isinstance(raw_hl, bool):
                            sep = "：" if "：" in k_txt else (":" if ":" in k_txt else "")
                            hl_str = k_txt.split(sep, 1)[1].strip() if (raw_hl and sep) else ""
                        elif raw_hl is None:
                            hl_str = ""
                        else:
                            hl_str = str(raw_hl).strip()
                        hc = str(kl.get("highlight_color") or "yellow").lower().strip()
                        norm_kls.append(
                            {
                                "text": k_txt,
                                "highlight": hl_str,
                                "highlight_color": "red" if ("red" in hc or "crimson" in hc) else "yellow",
                            }
                        )
            if not norm_kls:
                parts = [p.strip() for p in re.split(r"[，。！？!?]", nar) if p.strip()][:2]
                norm_kls = [{"text": pt[:18], "highlight": "", "highlight_color": "yellow"} for pt in parts]
            sc["kinetic_lines"] = norm_kls

        # Normalize `comparison`
        if raw_mode == "product_comparison":
            cmp_raw = sc.get("comparison") if isinstance(sc.get("comparison"), dict) else {}
            parts = [p.strip() for p in re.split(r"[，。！？!?]", nar) if len(p.strip()) >= 5]
            l_pts = [str(x) for x in (cmp_raw.get("left_points") or parts[:2]) if str(x).strip()] or [nar[:22]]
            r_pts = [str(x) for x in (cmp_raw.get("right_points") or parts[2:4] or parts[-2:]) if str(x).strip()] or [nar[-22:]]
            sc["comparison"] = {
                "heading": str(cmp_raw.get("heading") or sc["document_heading"] or "市場標示與實況對照")[:28],
                "left_title": str(cmp_raw.get("left_title") or "包裝宣稱 / 表面標準")[:16],
                "left_points": l_pts[:3],
                "right_title": str(cmp_raw.get("right_title") or "實際規定 / 換算真相")[:16],
                "right_points": r_pts[:3],
                "delta_badge": str(cmp_raw.get("delta_badge") or "關鍵落差")[:14],
            }
        else:
            sc["comparison"] = None

        # Normalize `comments`
        if raw_mode == "netizen_comments":
            cmts_raw = sc.get("comments") if isinstance(sc.get("comments"), list) else []
            norm_cmts = []
            for i_c, c_it in enumerate(cmts_raw[:4]):
                if isinstance(c_it, dict) and str(c_it.get("text") or "").strip():
                    norm_cmts.append(
                        {
                            "user": str(c_it.get("user") or f"@tw_citizen_{i_c+1}"),
                            "time": str(c_it.get("time") or f"{i_c+2} 小時前"),
                            "text": str(c_it["text"]).strip(),
                            "likes": int(c_it.get("likes") or (380 - i_c * 65)),
                        }
                    )
            if len(norm_cmts) < 3:
                parts = [p.strip() for p in re.split(r"[，。！？!?]", nar) if len(p.strip()) >= 6]
                while len(norm_cmts) < 4:
                    k_i = len(norm_cmts)
                    txt_fb = parts[k_i % len(parts)] if parts else nar[:28]
                    norm_cmts.append(
                        {
                            "user": f"@netizen_tw_{k_i+1}",
                            "time": f"{k_i+1} 小時前",
                            "text": txt_fb,
                            "likes": 420 - k_i * 73,
                        }
                    )
            sc["comments"] = norm_cmts
        else:
            sc["comments"] = []

        if raw_mode == "ai_scene" and not sc["image_prompt"]:
            sc["image_prompt"] = (
                "Documentary close-up photography inside a bright modern retail store, "
                "clean unbranded packaging, shallow depth of field, natural lighting, no text, no letters"
            )

        cleaned.append(sc)
    return cleaned


def format_youtube_descriptions(
    blueprint: EpisodeBlueprint,
    dossier: ResearchDossier,
) -> tuple[str, str]:
    profile = load_channel_profile()
    cite_lines = ["【公開資訊與法規參考來源】："]
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
    scale_factor: float = 0.82,
) -> EpisodeBlueprint:
    """Generate 32B 6-Act Long script + Showrunner Critic Title + Traffic-Push Short."""
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
    short_parsed: dict[str, Any] = {}
    critic_parsed: dict[str, Any] = {}

    for pr in pass_results:
        pid = pr["id"]
        try:
            obj = _ensure_zhtw(_extract_json_dict(pr["content"]))
        except Exception:
            continue
        if pid == "short_promo_pass":
            short_parsed = obj
        elif pid == "showrunner_critic_pass":
            critic_parsed = obj
        else:
            all_long_raw_scenes.extend(obj.get("scenes", []))

    clean_long_scenes = _sanitize_scene_list(all_long_raw_scenes, "L", dossier, is_short=False)
    clean_short_scenes = _sanitize_scene_list(
        short_parsed.get("scenes", []), "S", dossier, is_short=True
    )

    long_title = str(critic_parsed.get("long_title") or candidate.proposed_title).strip()
    seo_tags = [
        str(t)
        for t in (
            critic_parsed.get("seo_tags")
            or ["獨角獸那麼叫", "台灣", "公開資訊", "法規拆解", "消費者權益"]
        )
    ]

    hook_l1 = str(
        critic_parsed.get("top_hook_line1")
        or short_parsed.get("top_hook_line1")
        or candidate.short_hook_question[:9]
    ).strip()
    hook_l2 = str(
        critic_parsed.get("top_hook_line2")
        or short_parsed.get("top_hook_line2")
        or "真相是什麼 回答我!"
    ).strip()
    l2_color_raw = str(critic_parsed.get("hook_line2_color") or short_parsed.get("hook_line2_color") or "red").lower()
    l2_color: Literal["red", "yellow", "white"] = (
        "red" if "red" in l2_color_raw else ("white" if "white" in l2_color_raw else "yellow")
    )
    s_title = str(
        critic_parsed.get("short_title")
        or short_parsed.get("title")
        or f"{hook_l1}，{hook_l2} #獨角獸那麼叫 #回答我"
    ).strip()
    if s_title.startswith("#"):
        s_title = f"{hook_l1}，{hook_l2} {s_title}"

    long_cjk = len(re.findall(r"[\u4e00-\u9fff]", "".join(s["narration"] for s in clean_long_scenes)))
    short_cjk = len(re.findall(r"[\u4e00-\u9fff]", "".join(s["narration"] for s in clean_short_scenes)))

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
                "hook_line2_color": l2_color,
                "hook_highlight_words": [
                    str(w) for w in (short_parsed.get("hook_highlight_words") or ["回答我!"])
                ],
                "scenes": clean_short_scenes,
            }
        ),
    )
    bp.long_description, bp.short_description = format_youtube_descriptions(bp, dossier)
    return bp
