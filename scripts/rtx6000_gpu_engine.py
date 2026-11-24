#!/usr/bin/env python3
"""Unified PyTorch Heavy-Compute Engine for Colab RTX PRO 6000 (96GB) & CUDA Nodes.

Master copy lives locally in `scripts/rtx6000_gpu_engine.py` and is synced on demand.
Provides 3 heavy GPU tasks in pure PyTorch:
1. `llm_generate`    — Qwen/Qwen2.5-14B-Instruct (bfloat16) for @unicorn-roar script generation
2. `synthesize_voice`— PyTorch F5-TTS (F5TTS_v1_Base) zero-shot owner voice cloning + LUFS mastering + phrase timing
3. `generate_images` — PyTorch SDXL-Lightning 4-step (float16) high-speed 16:9 & 1:1 AI visual generation
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def split_subtitle_phrases(text: str, max_chars: int = 17) -> list[str]:
    """Split Traditional Chinese narration into crisp @unicorn-roar single-line subtitle phrases."""
    raw_parts = re.split(r"[，。！？!?；;：:\n]+", text)
    phrases: list[str] = []
    for part in raw_parts:
        p = part.strip()
        if not p:
            continue
        while len(p) > max_chars:
            # Prefer splitting near natural conjunctions if present around middle
            cut = max_chars
            for kw in ("但是", "結果", "因為", "所以", "其實", "還是", "卻", "而且", "如果"):
                idx = p.find(kw, 5, max_chars + 2)
                if idx != -1:
                    cut = idx
                    break
            phrases.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            phrases.append(p)
    return phrases or [text.strip()]


def align_phrases_to_audio(
    wav: np.ndarray,
    sr: int,
    phrases: list[str],
) -> list[dict]:
    """Compute energy-aware natural phrase timestamps bounded accurately to spoken waveform."""
    total_dur = len(wav) / float(sr)
    if not phrases:
        return []
    if len(phrases) == 1:
        return [{"start": 0.0, "end": round(total_dur, 3), "text": phrases[0]}]

    # Find low-energy pause valleys to snap subtitle transitions onto natural breath pauses
    hop = int(0.02 * sr)
    n_frames = max(1, len(wav) // hop)
    trimmed = wav[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(trimmed**2, axis=1) + 1e-12)

    weights = np.array([max(1.0, float(len(re.sub(r"\s+", "", p)))) for p in phrases])
    cum_ratios = np.cumsum(weights) / np.sum(weights)

    boundaries = [0.0]
    for r in cum_ratios[:-1]:
        ideal_frame = int(r * n_frames)
        win_lo = max(1, ideal_frame - 12)  # +/- 240ms search window for speech valley
        win_hi = min(n_frames - 1, ideal_frame + 12)
        if win_hi > win_lo:
            valley_offset = int(np.argmin(rms[win_lo:win_hi]))
            snap_sec = (win_lo + valley_offset) * 0.02
        else:
            snap_sec = r * total_dur
        snap_sec = max(boundaries[-1] + 0.35, min(total_dur - 0.30, snap_sec))
        boundaries.append(snap_sec)
    boundaries.append(total_dur)

    out = []
    for i, p in enumerate(phrases):
        out.append(
            {
                "start": round(boundaries[i], 3),
                "end": round(boundaries[i + 1], 3),
                "text": p,
            }
        )
    return out


def master_narration_audio(wav: np.ndarray, sr: int, target_lufs: float = -16.0) -> np.ndarray:
    """Broadcast-normalize cloned voice to -16 LUFS with -1.5 dBTP soft peak limiter."""
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    # Trim extreme leading/trailing near-silence (> -48 dBFS threshold) while preserving 80ms pad
    thresh = 0.004
    idx = np.where(np.abs(wav) > thresh)[0]
    if len(idx) > 10:
        pad = int(0.08 * sr)
        s = max(0, int(idx[0]) - pad)
        e = min(len(wav), int(idx[-1]) + int(0.16 * sr))
        wav = wav[s:e]

    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        if np.isfinite(loudness):
            wav = pyln.normalize.loudness(wav, loudness, target_lufs)
    except Exception:
        rms = float(np.sqrt(np.mean(wav**2)) + 1e-9)
        wav = wav * (0.14 / rms)

    # Soft peak ceiling at 0.84 (-1.5 dBFS)
    peak = float(np.max(np.abs(wav)) + 1e-9)
    if peak > 0.84:
        wav = np.tanh(wav / peak * 1.15) * 0.84
    return wav


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


def _chat_once(model, tokenizer, system: str, user: str, max_new_tokens: int = 3200, temperature: float = 0.55) -> str:
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
            repetition_penalty=1.06,
            do_sample=True,
        )
    out_ids = gen_ids[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(out_ids, skip_special_tokens=True)


def task_llm_generate(req_path: Path, out_path: Path) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    req = json.loads(req_path.read_text(encoding="utf-8"))
    model_id = req.get("model", "Qwen/Qwen2.5-14B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

    # Multi-pass 6-Act Long + Short mode when `passes` list is provided
    if "passes" in req:
        pass_outputs = []
        prior_titles_and_points: list[str] = []
        for idx, p in enumerate(req["passes"]):
            pid = p["id"]
            min_cjk = int(p.get("min_cjk_chars", 0))
            anti_repeat = ""
            if prior_titles_and_points and p.get("inject_prior_summary", True):
                anti_repeat = (
                    "\n\n【前序章節已講過之論點（本幕嚴禁重複相同句子，必須推進全新細節與數據）】：\n"
                    + "\n".join(f"- {pt}" for pt in prior_titles_and_points[-5:])
                )
            user_msg = p["user"] + anti_repeat
            raw = _chat_once(
                model,
                tokenizer,
                p["system"],
                user_msg,
                max_new_tokens=int(p.get("max_new_tokens", 3400)),
                temperature=float(p.get("temperature", 0.55)),
            )
            # Enforce minimum CJK character count via continuation scenes if needed
            if min_cjk > 0:
                try:
                    parsed_obj = _extract_json_block(raw)
                except Exception:
                    # Retry once if malformed JSON
                    raw = _chat_once(
                        model,
                        tokenizer,
                        p["system"],
                        user_msg + "\n請務必只輸出合法 JSON 物件，確保雙引號與括號完整閉合。",
                        max_new_tokens=int(p.get("max_new_tokens", 3600)),
                        temperature=0.45,
                    )
                    parsed_obj = _extract_json_block(raw)

                for cont_round in range(3):
                    scenes = parsed_obj.get("scenes", [])
                    nar_text = "".join(sc.get("narration", "") for sc in scenes)
                    cjk_len = len(re.findall(r"[\u4e00-\u9fff]", nar_text))
                    if cjk_len >= min_cjk:
                        break
                    deficit = min_cjk - cjk_len
                    extra_n = max(2, min(4, (deficit // 105) + 1))
                    last_nar = scenes[-1].get("narration", "") if scenes else ""
                    cont_prompt = (
                        f"目前本幕已撰寫 {cjk_len} 字，距離目標門檻 {min_cjk} 字還差 {deficit} 字。\n"
                        f"上一句結尾為：「{last_nar}」\n"
                        f"請緊接在上一句之後，針對本幕同一主題繼續深入展開（補充具體數字換算、條文細節、供應鏈實況或犀利生活類比，嚴禁重複前面講過的話），"
                        f"再新增 **{extra_n} 個場景**，每個場景 `narration` 必須寫滿 110～140 個繁體中文字！\n"
                        f"請只輸出 `{{\"scenes\": [ ...新增的 {extra_n} 個場景... ]}}` 合法 JSON。"
                    )
                    cont_raw = _chat_once(
                        model,
                        tokenizer,
                        p["system"],
                        user_msg + "\n\n" + cont_prompt,
                        max_new_tokens=2600,
                        temperature=0.58,
                    )
                    try:
                        cont_obj = _extract_json_block(cont_raw)
                        new_scenes = cont_obj.get("scenes", [])
                        if new_scenes:
                            parsed_obj["scenes"] = scenes + new_scenes
                    except Exception:
                        continue
                raw = json.dumps(parsed_obj, ensure_ascii=False)
            try:
                obj = _extract_json_block(raw)
                scs = obj.get("scenes", [])
                if scs:
                    summary_line = f"{pid}: " + " / ".join(sc.get("narration", "")[:28] for sc in scs[:3])
                    prior_titles_and_points.append(summary_line)
            except Exception:
                pass
            pass_outputs.append({"id": pid, "content": raw})
            print(f"[LLM PASS {idx+1}/{len(req['passes'])}] {pid} complete", flush=True)

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
            max_new_tokens=int(req.get("max_new_tokens", 5500)),
            temperature=float(req.get("temperature", 0.55)),
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
    for item in req["segments"]:
        seg_id = item["id"]
        text = re.sub(r"\s+", "，", item["text"].strip())
        if text and text[-1] not in "。！？!?":
            text += "。"
        role = item.get("voice_style", "narrative_explain")
        speed = float(item.get("speed", 1.04))
        max_chars = int(item.get("max_chars_per_line", 17))
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
        mastered = master_narration_audio(np.asarray(wav), int(sr), target_lufs=-16.0)
        # Write 44.1kHz stereo-ready clean PCM_16 WAV for frame-accurate FFmpeg muxing
        tmp_24k = out_dir / f"{seg_id}_24k.wav"
        sf.write(str(tmp_24k), mastered, int(sr), subtype="PCM_16")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_24k), "-ar", "44100", "-ac", "2", str(final_wav_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tmp_24k.unlink(missing_ok=True)

        phrases = split_subtitle_phrases(text, max_chars=max_chars)
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

    (out_dir / "tts_results.json").write_text(
        json.dumps({"ok": True, "device": device, "segments": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    del tts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def task_generate_images(req_path: Path, out_dir: Path) -> None:
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    req = json.loads(req_path.read_text(encoding="utf-8"))
    base_id = "stabilityai/stable-diffusion-xl-base-1.0"
    lightning_repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"

    unet_cfg = UNet2DConditionModel.load_config(base_id, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_cfg).to("cuda", torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(lightning_repo, ckpt), device="cuda"))
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_id, unet=unet, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for item in req["images"]:
        img_id = item["id"]
        prompt = item["prompt"]
        neg_prompt = item.get(
            "negative_prompt",
            "text, watermark, blurry, deformed, ugly, low quality, distorted hands, gibberish letters",
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
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=gen,
        ).images[0]
        out_file = out_dir / f"{img_id}.png"
        img.save(out_file)
        generated.append({"id": img_id, "file": out_file.name, "width": w, "height": h})

    (out_dir / "image_results.json").write_text(
        json.dumps({"ok": True, "images": generated}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    del pipe, unet
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
