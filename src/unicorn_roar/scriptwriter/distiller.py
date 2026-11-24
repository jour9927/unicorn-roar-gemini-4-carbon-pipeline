"""Channel Script Distillation Engine for 獨角獸那麼叫 (@unicorn-roar).

Builds and loads the permanent Script Distillation Vault (`assets/script_distillation/`):
1. Parses creator-authored `.vtt` human subtitles + RTX PRO 6000 Whisper-large-v3-turbo full transcripts.
2. Computes empirical @unicorn-roar copywriting laws:
   - Exact CJK character count & speaking rate (CPM) for 12-15 min Long episodes & 2-3 min Shorts
   - 6-Act timeline segmentation (0-12%, 12-28%, 28-48%, 48-68%, 68-86%, 86-100%)
   - Verbatim rhetorical moves, transition phrases, satirical analogies, and regulatory breakdown exemplars
3. Provides Pillar-Aware Exemplar Retrieval (RAG) so each of the 6 Acts in new Long scripts is grounded
   in real @unicorn-roar full-length paragraph exemplars and meets real 12-15 minute character density.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from opencc import OpenCC

from unicorn_roar.config import ASSETS_DIR

DISTILL_DIR = ASSETS_DIR / "script_distillation"
CORPUS_DIR = DISTILL_DIR / "corpus"
PLAYBOOK_JSON = DISTILL_DIR / "distilled_playbook.json"
PLAYBOOK_MD = DISTILL_DIR / "distilled_playbook.md"

SIX_ACT_SCHEMA = [
    {
        "act_id": "act_1_hook_reversal",
        "name": "第一幕：荒謬反差開場與焦點翻轉",
        "time_ratio": [0.0, 0.13],
        "target_cjk_chars": 560,
        "target_scenes": 6,
        "role_guide": (
            "從消費者最熟悉的新聞事件、超商貨架或帳單痛點切入，第一句直接拋出違反常識的數據或標示矛盾；"
            "接著用「你先別急著生氣啊」「你沒聽錯！」翻轉表面爭點，點出背後真正荒謬的制度或標示核心。"
        ),
    },
    {
        "act_id": "act_2_regulation_anatomy",
        "name": "第二幕：官方公文、法規條文與統計定義逐字拆解",
        "time_ratio": [0.13, 0.30],
        "target_cjk_chars": 720,
        "target_scenes": 7,
        "role_guide": (
            "以「好吧認真來說」切入硬核查證，直接攤開食藥署、農業部、主計總處、CNS或OECD的公開函釋與定義條文，"
            "逐字解釋官方門檻怎麼訂、為什麼業者這樣標示竟然合法（或明明發函了卻沒人改），並用白話算術算給觀眾聽。"
        ),
    },
    {
        "act_id": "act_3_supply_chain_tacit_rules",
        "name": "第三幕：供應鏈真相、成本結構與「業界不說的默契」",
        "time_ratio": [0.30, 0.50],
        "target_cjk_chars": 780,
        "target_scenes": 8,
        "role_guide": (
            "深入拆解原料端、加工端到零售/餐飲菜單的命名變化與價差結構：上游報表寫什麼、配料表排第幾位、"
            "添加物或統計口徑怎麼把低成本包裝成高溢價，揭穿「約定俗成」背後的商業利益。"
        ),
    },
    {
        "act_id": "act_4_international_benchmark",
        "name": "第四幕：跨國法規與國際標準硬核對照",
        "time_ratio": [0.50, 0.70],
        "target_cjk_chars": 760,
        "target_scenes": 7,
        "role_guide": (
            "橫向對比美國FDA、日本JAS/景品表示法、歐盟、南韓或國際組織對同一件事的嚴格規範："
            "為什麼別國強制要求正面大字標示比例或直接開罰，到了台灣卻變成消費者自己要帶放大鏡看背面？"
        ),
    },
    {
        "act_id": "act_5_rebuttal_and_incentives",
        "name": "第五幕：預判護航話術打臉與劣幣驅逐良幣機制",
        "time_ratio": [0.70, 0.87],
        "target_cjk_chars": 680,
        "target_scenes": 6,
        "role_guide": (
            "正面迎擊常見護航說詞（例如「大家都知道那是行銷名稱啊」「管太嚴業者怎麼生存」），"
            "用犀利日常類比（如「能為了賣豬肉標成牛肉嗎？」）點破資訊不對稱如何懲罰老實標示的良心業者。"
        ),
    },
    {
        "act_id": "act_6_consumer_action_and_close",
        "name": "第六幕：實戰自保判讀法與結尾制度叩問",
        "time_ratio": [0.87, 1.0],
        "target_cjk_chars": 520,
        "target_scenes": 5,
        "role_guide": (
            "提供觀眾具體可執行的看標示技巧、成分表關鍵字或檢舉/查證管道，"
            "最後拉回制度層面留下一記冷峻而發人深省的反問收束全片。"
        ),
    },
]

CONNECTOR_PATTERNS = [
    ("反差開場 / 翻轉", r"(你沒聽錯|你先別急著生氣|難道是|結果呢|神奇的就來了|重點已經不是|卻變成)"),
    ("切入硬核法規 / 算術", r"(認真來說|問題來了|我們來看|根據.{2,10}規定|標準是|換句話說|算一下)"),
    ("冷幽默吐槽 / 類比", r"(好樣的|寫給誰看|回答我|行銷可以這樣嗎|混成一鍋粥|貫破地板|當成盤子|約定俗成)"),
    ("預判護航與反詰", r"(有人會說|護航|推給|難道.{2,15}嗎|為什麼.{2,18}卻)"),
]


def parse_vtt_to_segments(vtt_path: Path) -> list[dict[str, Any]]:
    cc = OpenCC("s2twp")
    raw = vtt_path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", raw.strip())
    ts_re = re.compile(r"(\d+):(\d+):(\d+\.\d+)\s*-->\s*(\d+):(\d+):(\d+\.\d+)")
    segs = []
    for b in blocks:
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        m = None
        idx_ln = 0
        for i, ln in enumerate(lines):
            m = ts_re.search(ln)
            if m:
                idx_ln = i
                break
        if not m:
            continue
        h1, m1, s1, h2, m2, s2 = m.groups()
        st = round(int(h1) * 3600 + int(m1) * 60 + float(s1), 2)
        et = round(int(h2) * 3600 + int(m2) * 60 + float(s2), 2)
        txt = cc.convert(" ".join(lines[idx_ln + 1 :]).strip())
        if txt:
            segs.append({"start": st, "end": et, "text": txt})
    return segs


def classify_episode_pillar(title: str) -> str:
    if any(k in title for k in ("薪", "缺工", "社宅", "無人化", "生小孩", "台灣病")):
        return "wage_labor_stats"
    if any(k in title for k in ("多收", "15元", "盤", "最貴", "降價", "零關稅")):
        return "consumer_pricing"
    if any(k in title for k in ("石斑", "公帑", "補血", "溫體", "收購")):
        return "industry_policy"
    return "food_labeling"


def slice_episode_into_six_acts(segments: list[dict[str, Any]], total_dur: float) -> dict[str, str]:
    acts_out: dict[str, list[str]] = {a["act_id"]: [] for a in SIX_ACT_SCHEMA}
    if total_dur <= 0 and segments:
        total_dur = max(float(segments[-1]["end"]), 1.0)
    for seg in segments:
        mid = (float(seg["start"]) + float(seg["end"])) * 0.5
        ratio = min(0.999, max(0.0, mid / max(total_dur, 1.0)))
        for act in SIX_ACT_SCHEMA:
            lo, hi = act["time_ratio"]
            if lo <= ratio < hi or (act == SIX_ACT_SCHEMA[-1] and ratio >= lo):
                acts_out[act["act_id"]].append(seg["text"])
                break
    return {k: "，".join(v) for k, v in acts_out.items()}


def extract_rhetorical_gems(episodes: list[dict[str, Any]]) -> dict[str, list[str]]:
    gems: dict[str, list[str]] = {label: [] for label, _ in CONNECTOR_PATTERNS}
    for ep in episodes:
        lines = [s["text"].strip() for s in ep.get("segments", []) if s.get("text")]
        for i, ln in enumerate(lines):
            window = "，".join(lines[max(0, i - 1) : min(len(lines), i + 3)])
            if not (22 <= len(window) <= 115):
                continue
            for label, pat in CONNECTOR_PATTERNS:
                if re.search(pat, ln) and window not in gems[label]:
                    if len(gems[label]) < 14:
                        gems[label].append(window)
    return gems


def load_distilled_playbook() -> dict[str, Any]:
    if not PLAYBOOK_JSON.exists():
        raise FileNotFoundError(
            f"Distilled playbook not found at {PLAYBOOK_JSON}. Run scripts/distill_channel_scripts.py first."
        )
    return json.loads(PLAYBOOK_JSON.read_text(encoding="utf-8"))


def retrieve_act_exemplars(
    playbook: dict[str, Any],
    pillar: str,
    act_id: str,
    max_exemplars: int = 2,
    max_chars_each: int = 420,
) -> list[dict[str, str]]:
    """Retrieve real distilled @unicorn-roar passages for a specific Act & Pillar."""
    all_eps = playbook.get("long_exemplars", [])
    same_pillar = [e for e in all_eps if e.get("pillar") == pillar]
    other_pillar = [e for e in all_eps if e.get("pillar") != pillar]
    ordered = same_pillar + other_pillar
    out = []
    for ep in ordered:
        passage = (ep.get("acts", {}).get(act_id) or "").strip()
        if len(passage) >= 120:
            out.append(
                {
                    "video_id": ep["id"],
                    "title": ep["title"],
                    "excerpt": passage[:max_chars_each] + ("…" if len(passage) > max_chars_each else ""),
                }
            )
        if len(out) >= max_exemplars:
            break
    return out
