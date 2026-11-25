"""Taiwanese Mandarin TTS Pronunciation & Text Normalization Engine (`tw_pronunciation.py`).

Eliminates ALL `F5-TTS` Chinglish digit-reading (`twenty 公升`, `one hundred percent`, `three point three percent`)
and Traditional-vs-Simplified polyphone errors (`急著 -> 急住`, `調合油 -> 掉合油`, `垃圾 -> lā jī`)
while keeping on-screen subtitles in 100% pure Taiwan Traditional Chinese (`zh-TW`)!
"""
from __future__ import annotations

import re

import cn2an
from opencc import OpenCC

_CC_TW2S = OpenCC("tw2s")

# Words where `著` is read `zhù` / `zhuó` (NOT particle `zhe`!)
_ZHU_WORDS = (
    "顯著", "著名", "著作", "著稱", "著手", "著重", "著陸", "著落",
    "執著", "附著", "衣著", "著想", "著迷", "著急", "著火", "著涼",
)

# Taiwan Mandarin pronunciation & polyphone lexicon (applied to TTS input ONLY, never subtitles!)
_TW_PRONUNCIATION_MAP: list[tuple[str, str]] = [
    ("調合油", "調和油"),
    ("調合", "調和"),
    ("過分", "過份"),
    ("垃圾", "樂色"),
    ("角色", "腳色"),
    ("曝光", "鋪光"),
    ("CP值", "西批值"),
    ("QR Code", "Q R 扣"),
    ("Omega-3", "歐米茄三"),
    ("Omega3", "歐米茄三"),
    ("omega-3", "歐米茄三"),
    ("CXS 33-1981標準", "三十三號標準"),
    ("CXS 33-1981規範", "三十三號規範"),
    ("CXS 33-1981", "三十三號規範"),
    ("CXS", "國際標準"),
    ("聯合國Codex", "聯合國食品法典"),
    ("國際Codex", "國際食品法典"),
    ("Codex標準", "國際法典標準"),
    ("Codex分級", "國際法典分級"),
    ("Codex規範", "國際法典規範"),
    ("Codex", "國際食品法典"),
    ("Yahoo奇摩", "雅虎奇摩"),
    ("Yahoo", "雅虎"),
    ("UDN", "聯合新聞網"),
    ("TVBS", "T V B S "),
    ("CNA", "中央社"),
    ("OECD", "經合組織"),
    ("IOC", "國際橄欖油理事會"),
    ("FDA", "食藥署"),
    ("100%", "百分之百"),
]


def _year_to_digits_cn(m: re.Match[str]) -> str:
    digits_map = str.maketrans("0123456789", "零一二三四五六七八九")
    return m.group(1).translate(digits_map) + "年"


def _fraction_to_cn(m: re.Match[str]) -> str:
    num = cn2an.an2cn(m.group(1))
    den = cn2an.an2cn(m.group(2))
    return f"{den}分之{num}"


def _percent_range_to_cn(m: re.Match[str]) -> str:
    a = cn2an.an2cn(m.group(1))
    b = cn2an.an2cn(m.group(2))
    return f"百分之{a}到{b}"


def _percent_to_cn(m: re.Match[str]) -> str:
    raw = m.group(1)
    if raw == "100":
        return "百分之百"
    return "百分之" + cn2an.an2cn(raw)


def normalize_text_for_taiwan_tts(text: str) -> str:
    """Convert Taiwan Traditional Chinese script into 100% pure spoken Mandarin for F5-TTS."""
    out = text.strip()

    # 1. Strip parenthetical Latin glosses like `(Extra Virgin Olive Oil)` or `(Olive Pomace Oil)`
    #    that cause sudden unnatural code-switching inside Mandarin sentences
    out = re.sub(r"[（(][A-Za-z0-9\s,\-\./&]+[)）]", "", out)

    # 2. Apply Taiwan pronunciation lexicon
    for src, dst in _TW_PRONUNCIATION_MAP:
        out = out.replace(src, dst)

    # 3. Protect `著` words pronounced `zhù`/`zhuó` with placeholders, then turn particle `著` -> `着` (`zhe`!)
    placeholders: dict[str, str] = {}
    for idx, zw in enumerate(_ZHU_WORDS):
        if zw in out:
            ph = f"__ZHU{idx}__"
            placeholders[ph] = zw
            out = out.replace(zw, ph)
    out = out.replace("著", "着")
    for ph, zw in placeholders.items():
        out = out.replace(ph, zw)

    # 4. Normalize symbols, currencies, ranges, fractions, years, and percentages BEFORE general numbers
    out = out.replace("NT$", "新台幣").replace("US$", "美金").replace("$", "元")
    out = out.replace("±", "正負").replace("≠", "不等於").replace("≥", "大於等於").replace("≤", "小於等於")

    # Percentage ranges: `3.7~4.35%` or `3.7-4.35%`
    out = re.sub(r"(\d+(?:\.\d+)?)\s*[~～\-－至到]\s*(\d+(?:\.\d+)?)\s*[%％]", _percent_range_to_cn, out)
    # Single percentages: `3.3%`, `51%`
    out = re.sub(r"(\d+(?:\.\d+)?)\s*[%％]", _percent_to_cn, out)
    # Fractions: `2/3` (using digit lookarounds instead of `\b` because CJK chars are `\w` in Python!)
    out = re.sub(r"(?<!\d)([1-9])\s*/\s*([2-9]|10|100)(?!\d)", _fraction_to_cn, out)
    # Number ranges: `5~6個月`, `3-4倍`
    out = re.sub(r"(\d+(?:\.\d+)?)\s*[~～\-－]\s*(\d+(?:\.\d+)?)", r"\1到\2", out)
    # 4-digit calendar years: `明明2010年` / `到2027年` -> `二零一零年` / `二零二七年` (NEVER `\b` before digits!)
    out = re.sub(r"(?<!\d)(19\d{2}|20\d{2})年", _year_to_digits_cn, out)

    # Remove thousands commas inside numbers (`3,760萬元` -> `3760萬元`)
    out = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", out)

    # 5. Convert remaining Arabic numbers (integers & decimals like `580`, `17年`, `102年`, `2.0`, `3760`)
    #    using `cn2an.an2cn` directly so durations (`17年 -> 十七年`) and Minguo years (`102年 -> 一百零二年`)
    #    are NEVER misread as `一七年` / `一零二年`!
    out = re.sub(r"\d+(?:\.\d+)?", lambda m: cn2an.an2cn(m.group(0)), out)

    # 6. Natural Taiwan Mandarin `二` -> `兩` before classifiers/units (`兩萬`, `兩千`, `兩百`, `兩個`, `兩倍`, `兩罐`)
    out = re.sub(
        r"(?<![一二三四五六七八九十百千萬億第零點])二([萬千百個件罐瓶種倍成次家款桶層幕條項年天週歲])",
        r"兩\1",
        out,
    )
    # Keep `二月`, `二十`, `十二` as `二` (handled by negative lookbehind & classifier list; fix `兩年` vs `二年級`)
    out = out.replace("兩年級", "二年級")

    # 7. Remove inline quotation marks cleanly (never insert artificial commas around quoted nouns!)
    out = re.sub(r"[「」『』《》【】]", "", out)
    out = re.sub(r"[—…]+", "，", out)
    out = re.sub(r"[，,、；;：:]{2,}", "，", out)

    # 8. Convert to Simplified Chinese strictly for F5-TTS's internal `rjieba` + `pypinyin` tokenizer!
    out = _CC_TW2S.convert(out)
    # Ensure `义大利` stays `义大利` (`yì dà lì` in Taiwan Mandarin) if user prefers, or natural `意大利` (`yì dà lì` - identical pinyin `yi4 da4 li4`!)
    out = re.sub(r"\s+", " ", out).strip()
    if out and out[-1] not in "。！？!?":
        out += "。"
    return out
