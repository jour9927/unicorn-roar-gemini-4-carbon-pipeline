#!/usr/bin/env python3
"""RTX PRO 6000 Full-Length Channel Corpus Transcriber for 獨角獸那麼叫 (@unicorn-roar).

Transcribes full 12-15 min Long episodes and 1-3 min Shorts start-to-finish using
PyTorch `openai/whisper-large-v3-turbo` on CUDA, converts all text to Taiwan Traditional Chinese
via OpenCC (`s2twp`), and outputs structured timestamped transcripts + character/pacing stats.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import soundfile as sf
import torch
from opencc import OpenCC
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

WORK = Path("/content/unicorn_roar_g4c").resolve()
AUDIO_DIR = WORK / "corpus_audio"
OUT_FILE = WORK / "corpus_transcripts.json"

DOMAIN_FIXES = {
    "魚業署": "漁業署",
    "魚業鼠": "漁業署",
    "台灣鳥": "台灣鯛",
    "臺灣鳥": "臺灣鯛",
    "台灣釣": "台灣鯛",
    "鳥魚": "鯛魚",
    "釣魚便當": "鯛魚便當",
    "釣魚排": "鯛魚排",
    "蒲燒釣魚": "蒲燒鯛魚",
    "無鍋魚": "吳郭魚",
    "勿濤": "悟饕",
    "誤濤": "悟饕",
    "悟濤": "悟饕",
    "主計處": "主計總處",
}


def clean_tw_text(text: str, cc: OpenCC) -> str:
    t = cc.convert(text or "").strip()
    for wrong, right in DOMAIN_FIXES.items():
        t = t.replace(wrong, right)
    return t


def main() -> None:
    cc = OpenCC("s2twp")
    model_id = "openai/whisper-large-v3-turbo"
    dtype = torch.float16
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True
        ).to("cuda")
    except TypeError:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to("cuda")
    processor = AutoProcessor.from_pretrained(model_id)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device="cuda",
    )

    files = sorted(AUDIO_DIR.glob("*.mp3")) + sorted(AUDIO_DIR.glob("*.wav"))
    results: dict[str, dict] = {}
    for idx, f in enumerate(files, 1):
        info = sf.info(str(f))
        dur = round(float(info.duration), 2)
        out = asr(
            str(f),
            chunk_length_s=28,
            batch_size=16,
            return_timestamps=True,
            generate_kwargs={"language": "chinese", "task": "transcribe"},
        )
        segments = []
        for ch in out.get("chunks", []):
            ts = ch.get("timestamp") or (0.0, dur)
            s_t = round(float(ts[0] or 0.0), 2)
            e_t = round(float(ts[1] if ts[1] is not None else dur), 2)
            txt = clean_tw_text(ch.get("text", ""), cc)
            if txt:
                segments.append({"start": s_t, "end": e_t, "text": txt})

        full_text = "\n".join(s["text"] for s in segments)
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", full_text))
        cpm = round(cjk_chars / max(dur / 60.0, 0.1), 1)
        results[f.stem] = {
            "id": f.stem,
            "source": "whisper-large-v3-turbo+s2twp",
            "duration_sec": dur,
            "cjk_char_count": cjk_chars,
            "chars_per_min": cpm,
            "segment_count": len(segments),
            "segments": segments,
            "full_text": full_text,
        }
        print(f"[{idx}/{len(files)}] {f.stem}: dur={dur}s chars={cjk_chars} cpm={cpm}", flush=True)

    OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("CORPUS_TRANSCRIPTION_COMPLETE", len(results))


if __name__ == "__main__":
    main()
