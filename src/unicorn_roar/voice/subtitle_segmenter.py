"""Linguistic Word-Boundary & Spoken-Syllable Subtitle Segmenter (`subtitle_segmenter.py`).

Eliminates ALL unnatural mid-word, mid-number, mid-English, and orphan-enumeration subtitle breaks:
- NEVER splits numbers/dates/percentages/units (`2 | 026年`, `3 | 760萬元`, `96. | 7%`, `20 | 公升`)
- NEVER splits English words/codes (`Code | x`, `CXS | 33-1981`, `look at my eyes`)
- NEVER splits Mandarin disyllabic/compound words (`芥花 | 油`, `加 | 拿大`, `抽 | 查`, `自由健 | 康網`)
- NEVER creates 0.3s 2-char `、` enumeration flashes (`愛之味 / 泰山 / 寶素齋` stays grouped)
- NEVER leaves dangling half-quotation marks (`「蒙特樂`)
- Weights audio-to-subtitle alignment by **actual spoken Mandarin syllable count** (`3760萬元` = 8 spoken syllables)
"""
from __future__ import annotations

import re

import numpy as np
from opencc import OpenCC
import rjieba

from unicorn_roar.voice.tw_pronunciation import normalize_text_for_taiwan_tts

_CC_T2S = OpenCC("t2s")
_CC_S2TWP = OpenCC("s2twp")

_PROTECTED_COMPOUNDS: tuple[str, ...] = (
    "特級初榨橄欖油", "特級初榨", "橄欖粕油", "純橄欖油", "橄欖油", "芥花油", "調合油", "果渣油", "基礎油",
    "銅葉綠素", "食藥署", "消保處", "食安處", "自由健康網", "自由時報", "聯合新聞網", "環境資訊中心",
    "公視新聞", "Yahoo奇摩新聞", "Yahoo新聞", "卡優新聞", "上下游", "愛之味", "寶素齋", "蒙特樂", "大統假油",
    "二度預告", "標示新制", "分類標示", "標示草案", "強制標示", "不法獲利", "較經濟實惠", "經濟實惠",
    "大宗物資", "逆向推定", "資訊不對稱", "劣幣驅逐良幣", "專案稽查", "衛生標準", "國家標準",
    "國際規範", "國際標準", "業者雜音", "一路拖到", "拖到", "抓到", "分不清", "看不出",
    "加拿大", "義大利", "消費者", "計算機", "擋箭牌", "空窗期", "留言區",
)

_ATOMIC_RE = re.compile(
    r"look\s+at\s+my\s+eyes"
    r"|CXS\s*33-1981"
    r"|QR\s*Code"
    r"|Omega-?3"
    r"|(?:民國)?\d+(?:[.,~～\-/]\d+)*(?:\s*[%％]|年\d+月\d+日|年\d+月|月\d+日|年代|年|月|日|號|公升|毫升|公斤|公克|萬元|億元|千萬|百萬|萬|元|件|類|倍|份|成|歲|度|條|個|種|罐|瓶|家|款)?"
    r"|[A-Za-z][A-Za-z0-9\-._]*"
    r"|" + "|".join(re.escape(w) for w in sorted(_PROTECTED_COMPOUNDS, key=len, reverse=True)),
    re.IGNORECASE,
)

_QUANTITY_UNIT_RE = re.compile(
    r"^\d+(?:[.,]\d+)?(?:公升|毫升|公斤|公克|件|類|倍|份|成|個|種|罐|瓶|家|款)$"
)

_BREAK_BEFORE = {
    "但是", "結果", "因為", "所以", "其實", "還是", "卻是", "而是", "而且", "如果",
    "導致", "加上", "等於", "反觀", "偏偏", "難怪", "甚至", "尤其", "只要", "一旦",
    "卻", "而", "但", "還", "仍", "更", "就", "才", "都", "也", "只", "又", "竟",
    "把", "被", "讓", "將", "對", "從", "到", "在", "跟", "與", "或", "和", "及", "向", "依", "靠", "用",
    "加", "混", "配", "一路拖到",
}

_BREAK_AFTER_SUFFIX = ("年", "月", "日", "後", "前", "時", "裡", "上", "下", "內", "外", "、")
_NO_END_TOKENS = {"的", "之", "與", "跟", "和", "或", "及", "在", "從", "到", "把", "被", "讓", "對", "連", "較", "更", "最", "很", "太", "不", "沒"}
_NO_START_TOKENS = {"的", "了", "著", "過", "嗎", "呢", "吧", "啊", "喔", "嘛", "時", "後", "裡", "內", "上", "下"}


def visual_cjk_width(text: str) -> float:
    """Measure visual width in CJK character units (ASCII digits/letters take ~0.52 CJK width)."""
    w = 0.0
    for ch in text:
        if ch.isspace():
            w += 0.25
        elif ord(ch) < 128:
            w += 0.52
        else:
            w += 1.0
    return w


def _jieba_trad(sub: str) -> list[str]:
    out: list[str] = []
    for p in re.split(r"(、)", sub):
        if not p:
            continue
        if p == "、":
            if out:
                out[-1] += "、"
            continue
        simp_str = "".join(_CC_T2S.convert(ch)[0] for ch in p)
        idx = 0
        for w in rjieba.cut(simp_str):
            ln = len(w)
            out.append(p[idx : idx + ln])
            idx += ln
    return out


def tokenize_subtitle_atoms(clause: str) -> list[str]:
    """Segment Traditional Chinese clause into indivisible lexical/numeric/English atoms."""
    tokens: list[str] = []
    pos = 0
    for m in _ATOMIC_RE.finditer(clause):
        if m.start() > pos:
            tokens.extend(_jieba_trad(clause[pos : m.start()]))
        tokens.append(m.group(0))
        pos = m.end()
    if pos < len(clause):
        tokens.extend(_jieba_trad(clause[pos:]))
    return [t for t in tokens if t]


def _split_clause_recursive(clause: str, soft_max_vw: float, out: list[str]) -> None:
    vw = visual_cjk_width(clause)
    if vw <= soft_max_vw:
        cleaned = clause.rstrip("、").strip()
        if cleaned:
            out.append(cleaned)
        return

    tokens = tokenize_subtitle_atoms(clause)
    if len(tokens) <= 1:
        cleaned = clause.rstrip("、").strip()
        if cleaned:
            out.append(cleaned)
        return

    best_k: int | None = None
    best_score = -1e9
    target_left = vw * 0.50 if vw <= soft_max_vw * 1.85 else min(soft_max_vw - 1.0, vw * 0.38)

    cum_vw = 0.0
    for k in range(1, len(tokens)):
        cum_vw += visual_cjk_width(tokens[k - 1])
        rem_vw = vw - cum_vw
        if cum_vw < 4.5 or rem_vw < 4.5:
            continue
        if cum_vw > soft_max_vw + 0.8:
            continue

        left_tok = tokens[k - 1]
        right_tok = tokens[k]
        score = -abs(cum_vw - target_left) * 1.6
        if right_tok in _BREAK_BEFORE:
            score += 4.5
        if left_tok.endswith(_BREAK_AFTER_SUFFIX):
            score += 4.0
        if _QUANTITY_UNIT_RE.match(left_tok.rstrip("、")):
            score -= 4.5  # Keep `20公升` attached to `橄欖油`!
        if left_tok.rstrip("、") in _NO_END_TOKENS:
            score -= 9.0
        if right_tok.rstrip("、") in _NO_START_TOKENS:
            score -= 9.0

        if score > best_score:
            best_score = score
            best_k = k

    if best_k is None:
        cum = 0.0
        best_k = 1
        for k in range(1, len(tokens)):
            cum += visual_cjk_width(tokens[k - 1])
            if cum <= soft_max_vw:
                best_k = k

    left_str = "".join(tokens[:best_k]).rstrip("、").strip()
    right_str = "".join(tokens[best_k:]).strip()
    if left_str:
        out.append(left_str)
    if right_str:
        _split_clause_recursive(right_str, soft_max_vw, out)


def split_subtitle_phrases(text: str, max_chars: int = 15) -> list[str]:
    """Split Traditional Chinese narration into crisp, word-intact subtitle cues."""
    zhtw = _CC_S2TWP.convert(text)
    cleaned = re.sub(r"[「」『』《》【】]", "", zhtw).strip()
    raw_clauses = re.split(r"[，,。！？!?；;：:\n]+", cleaned)
    soft_max_vw = float(max_chars) + 2.5
    phrases: list[str] = []
    for rc in raw_clauses:
        rc = rc.strip(" —…")
        if not rc:
            continue
        _split_clause_recursive(rc, soft_max_vw, phrases)
    return phrases or [cleaned[:max_chars]]


def spoken_syllable_weight(phrase: str) -> float:
    """Count actual spoken syllables via Taiwan TTS normalizer so numbers (`3760` -> 6 syllables) align accurately."""
    norm = normalize_text_for_taiwan_tts(phrase)
    spoken = re.sub(r"[\s，,。！？!?；;：:、]", "", norm)
    # Count each English word as ~1.4 syllables and each CJK char as 1.0 syllable
    eng_words = re.findall(r"[A-Za-z]+", spoken)
    cjk_only = re.sub(r"[A-Za-z]+", "", spoken)
    return max(1.5, float(len(cjk_only)) + 1.4 * len(eng_words))


def align_phrases_to_audio(
    wav: np.ndarray,
    sr: int,
    phrases: list[str],
) -> list[dict]:
    """Snap word-intact subtitle phrases onto acoustic RMS micro-valleys using spoken syllable weights."""
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    total_dur = len(wav) / float(sr)
    if not phrases:
        return []
    if len(phrases) == 1:
        return [{"start": 0.0, "end": round(total_dur, 3), "text": phrases[0]}]

    hop = int(0.015 * sr)
    n_frames = max(1, len(wav) // hop)
    trimmed = wav[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(trimmed**2, axis=1) + 1e-12)

    weights = np.array([spoken_syllable_weight(p) for p in phrases], dtype=np.float64)
    cum_ratios = np.cumsum(weights) / np.sum(weights)

    boundaries = [0.0]
    for r in cum_ratios[:-1]:
        ideal_frame = int(r * n_frames)
        win_lo = max(1, ideal_frame - 11)
        win_hi = min(n_frames - 1, ideal_frame + 11)
        if win_hi > win_lo:
            valley_offset = int(np.argmin(rms[win_lo:win_hi]))
            snap_sec = (win_lo + valley_offset) * 0.015
        else:
            snap_sec = float(r) * total_dur
        snap_sec = max(boundaries[-1] + 0.32, min(total_dur - 0.28, snap_sec))
        boundaries.append(snap_sec)
    boundaries.append(total_dur)

    return [
        {"start": round(boundaries[i], 3), "end": round(boundaries[i + 1], 3), "text": p}
        for i, p in enumerate(phrases)
    ]
