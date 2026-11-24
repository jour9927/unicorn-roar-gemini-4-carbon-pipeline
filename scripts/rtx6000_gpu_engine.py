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
import torch

warnings.filterwarnings("ignore", category=UserWarning)


def split_subtitle_phrases(text: str, max_chars: int = 14) -> list[str]:
    """Split Traditional Chinese narration into @unicorn-roar 11-14 char rapid subtitle cues."""
    raw_parts = re.split(r"[，。！？!?；;：:、\n]+", text)
    phrases: list[str] = []
    for part in raw_parts:
        p = part.strip(" 「」『』")
        if not p:
            continue
        while len(p) > max_chars:
            cut = max_chars
            for kw in ("但是", "結果", "因為", "所以", "其實", "還是", "卻", "而且", "如果", "問題是", "也就是"):
                idx = p.find(kw, 5, max_chars + 1)
                if idx != -1:
                    cut = idx
                    break
            phrases.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            phrases.append(p)
    return phrases or [text.strip()[:max_chars]]


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


def master_narration_audio(wav: np.ndarray, sr: int, target_lufs: float = -13.2) -> np.ndarray:
    """Tight-cut breaths & master cloned voice to @unicorn-roar's exact `-13.2 LUFS` punch."""
    wav = tight_cut_breaths(wav, sr, max_pause_sec=0.095, kept_pause_sec=0.035)
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        if np.isfinite(loudness):
            # Pre-scale gently then soft-limit to -1.0 dBTP (0.891) so zero harsh clipping
            gain_db = target_lufs - loudness
            wav = wav * float(10.0 ** (gain_db / 20.0))
    except Exception:
        rms = float(np.sqrt(np.mean(wav**2)) + 1e-9)
        wav = wav * (0.21 / rms)

    # Smooth broadcast limiter ceiling at 0.89 (-1.0 dBFS)
    wav = np.tanh(wav * 1.08) * 0.89
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


def _chat_once(
    model,
    tokenizer,
    system: str,
    user: str,
    max_new_tokens: int = 3400,
    temperature: float = 0.52,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.08,
            do_sample=True,
        )
    out_ids = gen_ids[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(out_ids, skip_special_tokens=True)


def task_llm_generate(req_path: Path, out_path: Path) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    req = json.loads(req_path.read_text(encoding="utf-8"))
    model_id = req.get("model", "unsloth/Qwen2.5-32B-Instruct-bnb-4bit")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
    )

    if "passes" in req:
        pass_outputs = []
        prior_points: list[str] = []
        for idx, p in enumerate(req["passes"]):
            pid = p["id"]
            min_cjk = int(p.get("min_cjk_chars", 0))
            anti_repeat = ""
            if prior_points and p.get("inject_prior_summary", True):
                anti_repeat = (
                    "\n\n【前序幕次已講過之論點與案例（本幕嚴禁重複相同句子或同一數據，必須推進全新視角）】：\n"
                    + "\n".join(f"- {pt}" for pt in prior_points[-5:])
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

                for _ in range(3):
                    scenes = parsed_obj.get("scenes", [])
                    nar_text = "".join(sc.get("narration", "") for sc in scenes)
                    cjk_len = len(re.findall(r"[\u4e00-\u9fff]", nar_text))
                    if cjk_len >= min_cjk:
                        break
                    deficit = min_cjk - cjk_len
                    extra_n = max(2, min(4, (deficit // 110) + 1))
                    last_nar = scenes[-1].get("narration", "") if scenes else ""
                    cont_prompt = (
                        f"目前本幕已寫 {cjk_len} 字，距離目標 {min_cjk} 字還差 {deficit} 字。\n"
                        f"上一句結尾：「{last_nar}」\n"
                        f"請緊接其後，引用爬蟲真實段落再深化拆解具體價差算術、條文細節或通路實況（嚴禁重複前文、嚴禁編造不存在的數據），"
                        f"新增 **{extra_n} 個場景**（每段 `narration` 110～140 字），只輸出 `{{\"scenes\": [...]}}` JSON。"
                    )
                    cont_raw = _chat_once(
                        model,
                        tokenizer,
                        p["system"],
                        user_msg + "\n\n" + cont_prompt,
                        max_new_tokens=2600,
                        temperature=0.54,
                    )
                    try:
                        cont_obj = _extract_json_block(cont_raw)
                        if cont_obj.get("scenes"):
                            parsed_obj["scenes"] = scenes + cont_obj["scenes"]
                    except Exception:
                        continue
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
    from f5_tts.api import F5TTS

    req = json.loads(req_path.read_text(encoding="utf-8"))
    pack_dir = Path(req["voice_pack_dir"])
    manifest = json.loads((pack_dir / "voice_manifest.json").read_text(encoding="utf-8"))
    anchors_by_role = {a["role"]: a for a in manifest["anchors"]}
    default_anchor = manifest["anchors"][0]

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tts = F5TTS(model="F5TTS_v1_Base", device=device)

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(req["segments"])
    for idx, item in enumerate(req["segments"], 1):
        seg_id = item["id"]
        text = re.sub(r"\s+", "，", item["text"].strip())
        if text and text[-1] not in "。！？!?":
            text += "。"
        role = item.get("voice_style", "narrative_explain")
        speed = float(item.get("speed", 1.06))
        max_chars = int(item.get("max_chars_per_line", 14))
        anchor = anchors_by_role.get(role, default_anchor)

        raw_wav_path = out_dir / f"{seg_id}_raw.wav"
        final_wav_path = out_dir / f"{seg_id}.wav"

        wav, sr, _ = tts.infer(
            ref_file=str(pack_dir / anchor["file"]),
            ref_text=anchor["text"],
            gen_text=text,
            file_wave=str(raw_wav_path),
            speed=speed,
            remove_silence=False,
        )
        raw_wav_path.unlink(missing_ok=True)
        mastered = master_narration_audio(np.asarray(wav), int(sr), target_lufs=-13.2)

        tmp_24k = out_dir / f"{seg_id}_24k.wav"
        sf.write(str(tmp_24k), mastered, int(sr), subtype="PCM_16")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_24k), "-ar", "44100", "-ac", "2", str(final_wav_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tmp_24k.unlink(missing_ok=True)

        phrases = split_subtitle_phrases(item["text"].strip(), max_chars=max_chars)
        subs = align_phrases_to_audio(mastered, int(sr), phrases)
        dur = round(len(mastered) / float(sr), 3)
        results.append(
            {
                "id": seg_id,
                "file": final_wav_path.name,
                "duration": dur,
                "voice_style": anchor["role"],
                "subtitles": subs,
            }
        )
        if idx % 10 == 0 or idx == total:
            print(f"[TTS {idx}/{total}] {seg_id} dur={dur}s cues={len(subs)}", flush=True)

    (out_dir / "tts_results.json").write_text(
        json.dumps({"ok": True, "device": device, "segments": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    del tts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def task_generate_images(req_path: Path, out_dir: Path) -> None:
    """24-step High-Definition SDXL Base 1.0 (EulerDiscreteScheduler, CFG=6.5) — zero 4-step melting."""
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline

    req = json.loads(req_path.read_text(encoding="utf-8"))
    base_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_id, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

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
        gen = torch.Generator(device="cuda").manual_seed(seed)
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
