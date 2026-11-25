#!/usr/bin/env python3
"""V2 Unified PyTorch Heavy-Compute Engine for Colab RTX PRO 6000 (96GB) & CUDA Nodes.

V2 Station Upgrades:
1. `llm_generate`    — Upgraded to `unsloth/Qwen2.5-32B-Instruct-bnb-4bit` (32B NF4 on CUDA)
                       with multi-pass 6-Act + Showrunner Critic support & continuation length gate.
2. `synthesize_voice`— PyTorch F5-TTS owner voice cloning + Editor Breath-Cut Trimmer (`自動剪氣口`:
                       removes internal dead air > 95ms down to 35ms with 6ms cosine crossfades) +
                       `-13.2 LUFS` broadcast loudness normalization + 11-14 char crisp subtitle cues.
3. `generate_images` — Upgraded from 4-step Lightning to 24-step `stabilityai/stable-diffusion-xl-base-1.0`
                       (`EulerDiscreteScheduler`, `CFG=6.5`) for artifact-free photorealistic visuals.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning)


def _to_zhtw(text: str) -> str:
    try:
        from opencc import OpenCC
        return OpenCC("s2twp").convert(text)
    except Exception:
        return text


def split_subtitle_phrases(text: str, max_chars: int = 15) -> list[str]:
    """Split Traditional Chinese narration at natural phrase boundaries (zero mid-word breaks like `更多/錢`)."""
    text = _to_zhtw(text)
    raw_parts = re.split(r"[，。！？!?；;：:、\s]+", text)
    phrases: list[str] = []
    soft_max = max_chars + 2  # Keep <= 17-char clauses intact on one crisp line!
    for part in raw_parts:
        p = part.strip(" 「」『』")
        if not p:
            continue
        while len(p) > soft_max:
            cut: int | None = None
            for kw in ("但是", "結果", "因為", "所以", "其實", "還是", "卻是", "而是", "而且", "如果", "問題是", "也就是"):
                idx = p.find(kw, 5, max_chars + 2)
                if idx != -1 and (len(p) - idx) >= 5:
                    cut = idx
                    break
            if cut is None:
                mid = (len(p) + 1) // 2
                # Prefer cutting AFTER `了`, `過`, `完`, `後`, `時`, `裡`, `上`, `下` or BEFORE prepositions `在`, `把`, `被`, `讓`, `向`, `從`
                for after_ch in ("了", "過", "後", "時", "裡", "上", "下"):
                    idx = p.find(after_ch, max(5, mid - 4), min(max_chars, mid + 4))
                    if idx != -1 and (len(p) - (idx + 1)) >= 5:
                        cut = idx + 1
                        break
            if cut is None:
                mid = (len(p) + 1) // 2
                for before_ch in ("在", "把", "被", "讓", "跟", "從", "對", "比", "卻", "就", "都", "還"):
                    idx = p.find(before_ch, max(5, mid - 4), min(max_chars, mid + 4))
                    if idx != -1 and (len(p) - idx) >= 5:
                        cut = idx
                        break
            if cut is None:
                cut = min(max_chars, max(6, (len(p) + 1) // 2))
            phrases.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            phrases.append(p)
    return phrases or [text.strip()[:soft_max]]


def split_into_tts_clauses(text: str, target_max: int = 34) -> list[tuple[str, str]]:
    """Break scene narration into 16-36 char spoken clauses so F5-TTS preserves natural pitch & prosody!

    Returns list of `(clause_text_with_punct, ending_punct)`.
    """
    text = _to_zhtw(text).strip()
    tokens = re.split(r"([。！？!?；;，,：:])", text)
    raw_clauses: list[tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        seg = tokens[i].strip()
        punct = tokens[i + 1] if (i + 1) < len(tokens) else "。"
        i += 2
        if not seg:
            continue
        raw_clauses.append((seg, punct))

    merged: list[tuple[str, str]] = []
    buf_txt = ""
    for seg, punct in raw_clauses:
        cand = f"{buf_txt}，{seg}" if buf_txt else seg
        is_strong_end = punct in "。！？!?；;"
        if is_strong_end or len(cand) >= 20:
            if len(cand) > target_max and buf_txt:
                merged.append((buf_txt + "，", "，"))
                merged.append((seg + punct, punct))
            else:
                merged.append((cand + punct, punct))
            buf_txt = ""
        else:
            buf_txt = cand
    if buf_txt:
        merged.append((buf_txt + "。", "。"))
    return merged or [(text if text.endswith(("。", "！", "？")) else text + "。", "。")]


def tight_cut_breaths(
    wav: np.ndarray,
    sr: int,
    max_pause_sec: float = 0.095,
    kept_pause_sec: float = 0.035,
) -> np.ndarray:
    """Editor-grade breath & dead-air trimmer (`自動剪氣口`) matching @unicorn-roar's 89% zero-gap pacing."""
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    win = int(0.012 * sr)
    if len(wav) < win * 10:
        return wav

    n_wins = len(wav) // win
    frames = wav[: n_wins * win].reshape(n_wins, win)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    active_rms = float(np.percentile(rms, 75))
    silence_thresh = max(0.0045, active_rms * 0.065)
    is_silent = rms < silence_thresh

    # Trim leading/trailing silence down to 25ms
    active_indices = np.where(~is_silent)[0]
    if len(active_indices) == 0:
        return wav
    first_act = max(0, active_indices[0] - 2)
    last_act = min(n_wins, active_indices[-1] + 3)

    max_silent_wins = max(2, int(round(max_pause_sec / 0.012)))
    keep_silent_wins = max(1, int(round(kept_pause_sec / 0.012)))
    xfade = min(int(0.006 * sr), win // 2)

    pieces: list[np.ndarray] = []
    i = first_act
    while i < last_act:
        if not is_silent[i]:
            j = i + 1
            while j < last_act and not is_silent[j]:
                j += 1
            pieces.append(wav[i * win : j * win])
            i = j
        else:
            j = i + 1
            while j < last_act and is_silent[j]:
                j += 1
            run_len = j - i
            if run_len <= max_silent_wins:
                pieces.append(wav[i * win : j * win])
            else:
                pieces.append(wav[i * win : (i + keep_silent_wins) * win])
            i = j

    if not pieces:
        return wav

    # Smooth 6ms crossfade join to prevent any click artifacts
    out = pieces[0]
    for nxt in pieces[1:]:
        if len(out) > xfade and len(nxt) > xfade and xfade > 4:
            fade_out = np.linspace(1.0, 0.0, xfade, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, xfade, dtype=np.float32)
            overlap = out[-xfade:] * fade_out + nxt[:xfade] * fade_in
            out = np.concatenate([out[:-xfade], overlap, nxt[xfade:]])
        else:
            out = np.concatenate([out, nxt])
    return out


def align_phrases_to_audio(
    wav: np.ndarray,
    sr: int,
    phrases: list[str],
) -> list[dict]:
    """Snap subtitle transitions onto natural micro-valleys with zero gap between cues."""
    total_dur = len(wav) / float(sr)
    if not phrases:
        return []
    if len(phrases) == 1:
        return [{"start": 0.0, "end": round(total_dur, 3), "text": phrases[0]}]

    hop = int(0.015 * sr)
    n_frames = max(1, len(wav) // hop)
    trimmed = wav[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(trimmed**2, axis=1) + 1e-12)

    weights = np.array([max(1.0, float(len(re.sub(r"\s+", "", p)))) for p in phrases])
    cum_ratios = np.cumsum(weights) / np.sum(weights)

    boundaries = [0.0]
    for r in cum_ratios[:-1]:
        ideal_frame = int(r * n_frames)
        win_lo = max(1, ideal_frame - 10)
        win_hi = min(n_frames - 1, ideal_frame + 10)
        if win_hi > win_lo:
            valley_offset = int(np.argmin(rms[win_lo:win_hi]))
            snap_sec = (win_lo + valley_offset) * 0.015
        else:
            snap_sec = r * total_dur
        snap_sec = max(boundaries[-1] + 0.28, min(total_dur - 0.25, snap_sec))
        boundaries.append(snap_sec)
    boundaries.append(total_dur)

    return [
        {"start": round(boundaries[i], 3), "end": round(boundaries[i + 1], 3), "text": p}
        for i, p in enumerate(phrases)
    ]


def _normalize_for_tw_tts(text: str) -> str:
    """Self-contained Taiwan Mandarin TTS normalizer (digits/years/% -> Chinese words, `著->着`, `調合->調和`)."""
    import cn2an
    from opencc import OpenCC

    cc_tw2s = OpenCC("tw2s")
    zhu_words = (
        "顯著", "著名", "著作", "著稱", "著手", "著重", "著陸", "著落",
        "執著", "附著", "衣著", "著想", "著迷", "著急", "著火", "著涼",
    )
    lexicon = [
        ("調合油", "調和油"), ("調合", "調和"), ("過分", "過份"), ("垃圾", "樂色"),
        ("角色", "腳色"), ("曝光", "鋪光"), ("CP值", "西批值"), ("QR Code", "Q R 扣"),
        ("Omega-3", "歐米茄三"), ("Omega3", "歐米茄三"), ("omega-3", "歐米茄三"),
        ("CXS 33-1981標準", "三十三號標準"), ("CXS 33-1981規範", "三十三號規範"),
        ("CXS 33-1981", "三十三號規範"), ("CXS", "國際標準"),
        ("聯合國Codex", "聯合國食品法典"), ("國際Codex", "國際食品法典"),
        ("Codex標準", "國際法典標準"), ("Codex分級", "國際法典分級"),
        ("Codex規範", "國際法典規範"), ("Codex", "國際食品法典"),
        ("Yahoo奇摩", "雅虎奇摩"), ("Yahoo", "雅虎"),
        ("UDN", "聯合新聞網"), ("TVBS", "T V B S "), ("CNA", "中央社"),
        ("OECD", "經合組織"), ("IOC", "國際橄欖油理事會"), ("FDA", "食藥署"),
        ("100%", "百分之百"),
    ]
    out = re.sub(r"[（(][A-Za-z0-9\s,\-\./&]+[)）]", "", text.strip())
    for src, dst in lexicon:
        out = out.replace(src, dst)

    phs: dict[str, str] = {}
    for i, zw in enumerate(zhu_words):
        if zw in out:
            k = f"__ZHU{i}__"
            phs[k] = zw
            out = out.replace(zw, k)
    out = out.replace("著", "着")
    for k, zw in phs.items():
        out = out.replace(k, zw)

    out = out.replace("NT$", "新台幣").replace("US$", "美金").replace("$", "元")
    out = out.replace("±", "正負").replace("≠", "不等於").replace("≥", "大於等於").replace("≤", "小於等於")
    out = re.sub(
        r"(\d+(?:\.\d+)?)\s*[~～\-－至到]\s*(\d+(?:\.\d+)?)\s*[%％]",
        lambda m: f"百分之{cn2an.an2cn(m.group(1))}到{cn2an.an2cn(m.group(2))}",
        out,
    )
    out = re.sub(
        r"(\d+(?:\.\d+)?)\s*[%％]",
        lambda m: "百分之百" if m.group(1) == "100" else "百分之" + cn2an.an2cn(m.group(1)),
        out,
    )
    out = re.sub(r"(?<!\d)([1-9])\s*/\s*([2-9]|10|100)(?!\d)", lambda m: f"{cn2an.an2cn(m.group(2))}分之{cn2an.an2cn(m.group(1))}", out)
    out = re.sub(r"(\d+(?:\.\d+)?)\s*[~～\-－]\s*(\d+(?:\.\d+)?)", r"\1到\2", out)
    digits_map = str.maketrans("0123456789", "零一二三四五六七八九")
    out = re.sub(r"(?<!\d)(19\d{2}|20\d{2})年", lambda m: m.group(1).translate(digits_map) + "年", out)
    out = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", out)
    out = re.sub(r"\d+(?:\.\d+)?", lambda m: cn2an.an2cn(m.group(0)), out)
    out = re.sub(
        r"(?<![一二三四五六七八九十百千萬億第零點])二([萬千百個件罐瓶種倍成次家款桶層幕條項年天週歲])",
        r"兩\1",
        out,
    )
    out = out.replace("兩年級", "二年級")
    out = re.sub(r"[「」『』《》【】]", "", out)
    out = re.sub(r"[—…]+", "，", out)
    out = re.sub(r"[，,、；;：:]{2,}", "，", out)
    out = cc_tw2s.convert(out)
    out = re.sub(r"\s+", " ", out).strip()
    if out and out[-1] not in "。！？!?":
        out += "。"
    return out


def split_into_full_sentences(raw_zhtw: str) -> list[str]:
    """Split scene text ONLY on full sentence endings (`。！？!?`) when >= 22 chars for continuous natural prosody!"""
    raw_zhtw = _to_zhtw(raw_zhtw).strip()
    parts = re.split(r"(?<=[。！？!?])", raw_zhtw)
    merged: list[str] = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        cand = buf + p
        if len(cand) >= 22:
            merged.append(cand)
            buf = ""
        else:
            buf = cand
    if buf:
        if merged and len(buf) < 12:
            merged[-1] += buf
        else:
            merged.append(buf)
    return merged or [raw_zhtw]


def _trim_edge_silence_only(wav: np.ndarray, sr: int) -> np.ndarray:
    """Trim ONLY leading/trailing dead air (< -42 dBFS), NEVER chopping natural consonants inside sentences!"""
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    w = wav.astype(np.float32)
    win = int(0.015 * sr)
    n_w = len(w) // win
    if n_w < 6:
        return w
    rms = np.sqrt(np.mean(w[: n_w * win].reshape(n_w, win) ** 2, axis=1) + 1e-12)
    thresh = max(0.0028, float(np.percentile(rms, 75)) * 0.038)
    act = np.where(rms > thresh)[0]
    if len(act) == 0:
        return w
    s0 = max(0, (int(act[0]) - 2) * win)
    s1 = min(len(w), (int(act[-1]) + 3) * win)
    return w[s0:s1]


def master_narration_audio(wav: np.ndarray, sr: int, target_lufs: float = -12.9) -> np.ndarray:
    """Master natural unchopped cloned voice to `-13.0 LUFS` broadcast loudness."""
    wav = _trim_edge_silence_only(wav, sr)
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        if np.isfinite(loudness):
            gain_db = target_lufs - loudness
            wav = wav * float(10.0 ** (gain_db / 20.0))
    except Exception:
        rms = float(np.sqrt(np.mean(wav**2)) + 1e-9)
        wav = wav * (0.22 / rms)

    abs_w = np.abs(wav)
    knee = 0.78
    over = np.maximum(0.0, abs_w - knee)
    compressed = np.where(abs_w > knee, knee + 0.15 * np.tanh(over / 0.15), abs_w)
    wav = np.sign(wav) * compressed
    return wav.astype(np.float32)


def _extract_json_block(raw: str) -> dict:
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


def _chat_messages(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int = 3400,
    temperature: float = 0.52,
) -> str:
    import torch

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.10,
            do_sample=True,
        )
    out_ids = gen_ids[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(out_ids, skip_special_tokens=True)


def _chat_once(
    model,
    tokenizer,
    system: str,
    user: str,
    max_new_tokens: int = 3400,
    temperature: float = 0.52,
) -> str:
    return _chat_messages(
        model,
        tokenizer,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def _dedup_scenes_by_ngrams(
    scenes: list[dict],
    global_ngrams: list[set[str]],
    threshold: float = 0.55,
) -> list[dict]:
    kept: list[dict] = []
    local_seen = list(global_ngrams)
    for sc in scenes:
        if not isinstance(sc, dict):
            continue
        nar = str(sc.get("narration") or "").strip()
        cjk = "".join(re.findall(r"[\u4e00-\u9fff0-9]", nar))
        if len(cjk) < 16:
            continue
        ng = {cjk[k : k + 5] for k in range(max(1, len(cjk) - 4))}
        if any(
            len(ng & prev) / max(1, min(len(ng), len(prev))) > threshold
            for prev in local_seen
        ):
            continue
        local_seen.append(ng)
        kept.append(sc)
    return kept


def task_llm_generate(req_path: Path, out_path: Path) -> None:
    import os
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    req = json.loads(req_path.read_text(encoding="utf-8"))
    free_bytes, _ = torch.cuda.mem_get_info(0)
    free_gib = free_bytes / (1024**3)
    for _ in range(9):
        if free_gib >= 11.5:
            break
        print(f"[VRAM GUARD] {free_gib:.1f} GiB free; waiting 10s for sibling burst to clear...", flush=True)
        time.sleep(10.0)
        free_gib = torch.cuda.mem_get_info(0)[0] / (1024**3)

    if free_gib >= 23.5:
        model_id = req.get("model") or "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
    elif free_gib >= 11.5:
        print(f"[VRAM GUARD] {free_gib:.1f} GiB free; using cached 14B-bnb-4bit on GPU", flush=True)
        model_id = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
    else:
        print(f"[VRAM GUARD] {free_gib:.1f} GiB free; using compact 7B-bnb-4bit on GPU", flush=True)
        model_id = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    if "passes" in req:
        pass_outputs = []
        prior_points: list[str] = []
        global_ngrams: list[set[str]] = []
        for idx, p in enumerate(req["passes"]):
            pid = p["id"]
            min_cjk = int(p.get("min_cjk_chars", 0))
            is_short_pass = pid == "short_promo_pass"
            anti_repeat = ""
            if prior_points and p.get("inject_prior_summary", True):
                anti_repeat = (
                    "\n\n【前序幕次已講過之論點與案例（本幕嚴禁重複相同句子或同一數據，必須推進全新視角）】：\n"
                    + "\n".join(f"- {pt}" for pt in prior_points[-6:])
                )
            user_msg = p["user"] + anti_repeat
            raw = _chat_once(
                model,
                tokenizer,
                p["system"],
                user_msg,
                max_new_tokens=int(p.get("max_new_tokens", 3500)),
                temperature=float(p.get("temperature", 0.52)),
            )

            if min_cjk > 0:
                try:
                    parsed_obj = _extract_json_block(raw)
                except Exception:
                    raw = _chat_once(
                        model,
                        tokenizer,
                        p["system"],
                        user_msg + "\n請嚴格只輸出合法閉合的 JSON 物件。",
                        max_new_tokens=3600,
                        temperature=0.42,
                    )
                    parsed_obj = _extract_json_block(raw)

                ref_ngrams = [] if is_short_pass else global_ngrams
                parsed_obj["scenes"] = _dedup_scenes_by_ngrams(
                    parsed_obj.get("scenes", []), ref_ngrams
                )

                for attempt_idx in range(4):
                    scenes = parsed_obj.get("scenes", [])
                    nar_text = "".join(sc.get("narration", "") for sc in scenes)
                    cjk_len = len(re.findall(r"[\u4e00-\u9fff]", nar_text))
                    if cjk_len >= min_cjk:
                        break
                    deficit = min_cjk - cjk_len
                    extra_n = max(2, min(4, (deficit // 110) + 1))
                    assistant_so_far = json.dumps({"scenes": scenes}, ensure_ascii=False)
                    cont_prompt = (
                        f"目前本幕不重複有效字數為 {cjk_len} 字，距離目標 {min_cjk} 字還差 {deficit} 字。\n"
                        f"請緊接上述最後一段往深處推進（例如：拆解不同來源編號 S2..S14 的具體法條門檻、抽驗違規案例金額、跨國分級差異或消費者實測算術），"
                        f"嚴禁重寫或改寫上面已經出現過的任何句子！\n"
                        f"請只輸出**新增的 {extra_n} 個全新場景**（每段 `narration` 寫滿 115～140 字，視覺模式交替使用 `news_portal`, `official_gov_doc`, `product_comparison`, `netizen_comments`, `kinetic_black`, `ai_scene`），"
                        f"格式：`{{\"scenes\": [...]}}`。"
                    )
                    cont_raw = _chat_messages(
                        model,
                        tokenizer,
                        [
                            {"role": "system", "content": p["system"]},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": assistant_so_far},
                            {"role": "user", "content": cont_prompt},
                        ],
                        max_new_tokens=2600,
                        temperature=0.56 + 0.04 * attempt_idx,
                    )
                    try:
                        cont_obj = _extract_json_block(cont_raw)
                        if cont_obj.get("scenes"):
                            merged = scenes + cont_obj["scenes"]
                            parsed_obj["scenes"] = _dedup_scenes_by_ngrams(merged, ref_ngrams)
                    except Exception:
                        continue

                if not is_short_pass:
                    for sc in parsed_obj.get("scenes", []):
                        cjk_s = "".join(re.findall(r"[\u4e00-\u9fff0-9]", str(sc.get("narration") or "")))
                        if len(cjk_s) >= 16:
                            global_ngrams.append({cjk_s[k : k + 5] for k in range(max(1, len(cjk_s) - 4))})
                raw = json.dumps(parsed_obj, ensure_ascii=False)

            try:
                obj = _extract_json_block(raw)
                scs = obj.get("scenes", [])
                if scs:
                    prior_points.append(
                        f"{pid}: " + " / ".join(sc.get("narration", "")[:30] for sc in scs[:3])
                    )
            except Exception:
                pass
            pass_outputs.append({"id": pid, "content": raw})
            print(f"[32B PASS {idx+1}/{len(req['passes'])}] {pid} complete", flush=True)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"ok": True, "model": model_id, "passes": pass_outputs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        content = _chat_once(
            model,
            tokenizer,
            req["system"],
            req["user"],
            max_new_tokens=int(req.get("max_new_tokens", 4000)),
            temperature=float(req.get("temperature", 0.5)),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"ok": True, "model": model_id, "content": content}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def task_synthesize_voice(req_path: Path, out_dir: Path) -> None:
    import os
    import torch
    from f5_tts.api import F5TTS

    os.environ["PYTHONHASHSEED"] = "2026"
    req = json.loads(req_path.read_text(encoding="utf-8"))
    pack_dir = Path(req["voice_pack_dir"])
    manifest = json.loads((pack_dir / "voice_manifest.json").read_text(encoding="utf-8"))
    anchors_by_role = {a["role"]: a for a in manifest["anchors"]}
    default_anchor = manifest["anchors"][0]

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tts = F5TTS(model="F5TTS_v1_Base", device=device)
    os.environ["PYTHONHASHSEED"] = "2026"

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(req["segments"])
    for idx, item in enumerate(req["segments"], 1):
        seg_id = item["id"]
        final_wav_path = out_dir / f"{seg_id}.wav"
        meta_json_path = out_dir / f"{seg_id}.meta.json"
        if final_wav_path.exists() and meta_json_path.exists():
            try:
                results.append(json.loads(meta_json_path.read_text(encoding="utf-8")))
                continue
            except Exception:
                pass

        raw_text = _to_zhtw(item["text"].strip())
        role = item.get("voice_style", "narrative_explain")
        speed = float(item.get("speed", 1.01))
        max_chars = int(item.get("max_chars_per_line", 15))
        anchor = anchors_by_role.get(role, default_anchor)

        sentences_zhtw = split_into_full_sentences(raw_text)
        sent_wavs: list[np.ndarray] = []
        subs: list[dict] = []
        cursor_sec = 0.0
        sr = 24000
        nfe = 32 if device == "cuda" else 18

        for c_i, sent_zhtw in enumerate(sentences_zhtw):
            norm_tts_txt = _normalize_for_tw_tts(sent_zhtw)
            raw_wav_path = out_dir / f"{seg_id}_s{c_i}_raw.wav"
            wav_c, sr_c, _ = tts.infer(
                ref_file=str(pack_dir / anchor["file"]),
                ref_text=anchor["text"],
                gen_text=norm_tts_txt,
                file_wave=str(raw_wav_path),
                nfe_step=nfe,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
                speed=1.0,
                cross_fade_duration=0.12,
                seed=2026 + idx * 10 + c_i,
                remove_silence=False,
            )
            os.environ["PYTHONHASHSEED"] = "2026"
            raw_wav_path.unlink(missing_ok=True)
            sr = int(sr_c)
            trimmed_c = _trim_edge_silence_only(np.asarray(wav_c, dtype=np.float32), sr)

            gap_sec = 0.14 if c_i < len(sentences_zhtw) - 1 else 0.04
            full_c = np.concatenate([trimmed_c, np.zeros(int(gap_sec * sr), dtype=np.float32)])
            c_dur = len(full_c) / float(sr)

            # Display original clean zh-TW text (with crisp digits like 100%, 580公升) on screen!
            c_phrases = split_subtitle_phrases(sent_zhtw, max_chars=max_chars)
            c_subs = align_phrases_to_audio(full_c, sr, c_phrases)
            for s_item in c_subs:
                subs.append(
                    {
                        "start": round(cursor_sec + float(s_item["start"]), 3),
                        "end": round(cursor_sec + float(s_item["end"]), 3),
                        "text": s_item["text"],
                    }
                )
            sent_wavs.append(full_c)
            cursor_sec += c_dur

        combined_wav = np.concatenate(sent_wavs) if sent_wavs else np.zeros(sr, dtype=np.float32)
        mastered = master_narration_audio(combined_wav, sr, target_lufs=-12.9)

        # Ensure 0.000s visual gap between consecutive subtitle cues so pills never flicker!
        total_dur = round(len(mastered) / float(sr), 3)
        for s_idx in range(len(subs) - 1):
            subs[s_idx]["end"] = subs[s_idx + 1]["start"]
        if subs:
            subs[-1]["end"] = total_dur

        tmp_24k = out_dir / f"{seg_id}_24k.wav"
        sf.write(str(tmp_24k), mastered, sr, subtype="PCM_16")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_24k), "-ar", "44100", "-ac", "2", str(final_wav_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tmp_24k.unlink(missing_ok=True)

        seg_meta = {
            "id": seg_id,
            "file": final_wav_path.name,
            "duration": total_dur,
            "voice_style": anchor["role"],
            "subtitles": subs,
        }
        meta_json_path.write_text(json.dumps(seg_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(seg_meta)
        if idx % 5 == 0 or idx == total:
            print(
                f"[TTS {idx}/{total}] {seg_id} dur={total_dur}s sents={len(sentences_zhtw)} cues={len(subs)}",
                flush=True,
            )

    res_out_file = Path(req.get("results_filename", "tts_results.json"))
    if not res_out_file.is_absolute():
        res_out_file = out_dir / res_out_file
    res_out_file.write_text(
        json.dumps({"ok": True, "device": device, "segments": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    del tts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def task_generate_images(req_path: Path, out_dir: Path) -> None:
    """24-step High-Definition SDXL Base 1.0 (EulerDiscreteScheduler, CFG=6.5) — zero 4-step melting."""
    import torch
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline

    req = json.loads(req_path.read_text(encoding="utf-8"))
    base_id = "stabilityai/stable-diffusion-xl-base-1.0"
    free_gib = torch.cuda.mem_get_info(0)[0] / (1024**3)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_id, torch_dtype=torch.float16, variant="fp16"
    )
    if free_gib >= 9.0:
        pipe = pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    total = len(req["images"])
    for idx, item in enumerate(req["images"], 1):
        img_id = item["id"]
        prompt = item["prompt"]
        neg_prompt = item.get(
            "negative_prompt",
            "text, letters, typography, watermark, logo, gibberish writing, blurry, melted, deformed hands, lowres, cartoon",
        )
        w = int(item.get("width", 1344))
        h = int(item.get("height", 768))
        seed = int(item.get("seed", 42))
        gen = torch.Generator(device="cpu").manual_seed(seed)
        img = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            width=w,
            height=h,
            num_inference_steps=24,
            guidance_scale=6.5,
            generator=gen,
        ).images[0]
        out_file = out_dir / f"{img_id}.png"
        img.save(out_file)
        generated.append({"id": img_id, "file": out_file.name, "width": w, "height": h})
        if idx % 5 == 0 or idx == total:
            print(f"[SDXL-24STEP {idx}/{total}] {img_id} ({w}x{h}) saved", flush=True)

    (out_dir / "image_results.json").write_text(
        json.dumps({"ok": True, "images": generated}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["llm_generate", "synthesize_voice", "generate_images"])
    ap.add_argument("--req", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.task == "llm_generate":
        task_llm_generate(args.req, args.out)
    elif args.task == "synthesize_voice":
        task_synthesize_voice(args.req, args.out)
    elif args.task == "generate_images":
        task_generate_images(args.req, args.out)


if __name__ == "__main__":
    main()
