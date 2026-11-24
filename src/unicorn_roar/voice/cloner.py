"""V2 PyTorch Owner Voice Cloning Orchestrator (Tight Breath-Cut + `-13.2 LUFS` Mastering)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from unicorn_roar.config import (
    REPO_ROOT,
    get_runtime_endpoint_path,
    get_voice_pack_dir,
)
from unicorn_roar.remote.rtx6000_client import (
    RTX6000WorkerClient,
    RemoteWorkerEndpoint,
)
from unicorn_roar.scriptwriter.generator import EpisodeBlueprint


def _ensure_remote_voice_pack(client: RTX6000WorkerClient, local_pack: Path) -> str:
    remote_pack = "voice_pack_out"
    chk = client.exec_sync(f"test -f {remote_pack}/voice_manifest.json && echo READY || echo MISSING")
    if "READY" in chk.get("stdout", ""):
        return remote_pack
    for f in local_pack.glob("*"):
        if f.is_file():
            client.upload_file(f, f"{remote_pack}/{f.name}")
    return remote_pack


def synthesize_episode_narration(
    blueprint: EpisodeBlueprint,
    run_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Synthesize all Long & Short segments in owner's voice with tight breath-cutting & -13.2 LUFS."""
    local_pack = get_voice_pack_dir()
    if not (local_pack / "voice_manifest.json").exists():
        raise FileNotFoundError(f"Owner voice reference pack missing at {local_pack}/voice_manifest.json")

    cached_res = run_dir / "audio" / "tts_results.json"
    if cached_res.exists():
        try:
            res_data = json.loads(cached_res.read_text(encoding="utf-8"))
            by_id = {}
            for seg in res_data["segments"]:
                lp = run_dir / "audio" / seg["file"]
                if lp.exists():
                    seg["local_path"] = str(lp)
                    by_id[seg["id"]] = seg
            expected_ids = {s.scene_id for s in blueprint.long_video.scenes} | {
                s.scene_id for s in blueprint.short_video.scenes
            }
            if expected_ids.issubset(by_id.keys()):
                return by_id
        except Exception:
            pass

    segments_req = []
    for sc in blueprint.long_video.scenes:
        segments_req.append(
            {
                "id": sc.scene_id,
                "text": sc.narration,
                "voice_style": sc.voice_style,
                "speed": 1.06,
                "max_chars_per_line": 14,
            }
        )
    for sc in blueprint.short_video.scenes:
        segments_req.append(
            {
                "id": sc.scene_id,
                "text": sc.narration,
                "voice_style": sc.voice_style if sc.voice_style != "narrative_explain" else "short_energetic",
                "speed": 1.10,
                "max_chars_per_line": 13,
            }
        )

    audio_out_dir = run_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    local_req_file = run_dir / "temp" / "tts_req.json"
    local_req_file.parent.mkdir(parents=True, exist_ok=True)

    ep = RemoteWorkerEndpoint.from_file(get_runtime_endpoint_path())
    if ep is not None:
        try:
            client = RTX6000WorkerClient(ep)
            remote_pack = _ensure_remote_voice_pack(client, local_pack)
            remote_job_dir = f"runs/{run_dir.name}/tts"
            req_data = {"voice_pack_dir": remote_pack, "segments": segments_req}
            local_req_file.write_text(json.dumps(req_data, ensure_ascii=False, indent=2), encoding="utf-8")

            client.upload_file(REPO_ROOT / "scripts/rtx6000_gpu_engine.py", "scripts/rtx6000_gpu_engine.py")
            client.upload_file(local_req_file, f"{remote_job_dir}/tts_req.json")
            jid = client.submit_job(
                f".venv/bin/python scripts/rtx6000_gpu_engine.py --task synthesize_voice "
                f"--req {remote_job_dir}/tts_req.json --out {remote_job_dir}/out",
                job_id=f"tts_{run_dir.name}",
            )
            info = client.wait_job(jid, poll_interval=3.0, timeout=1200.0)
            if info.get("rc") != 0:
                raise RuntimeError(f"RTX6000 TTS failed: {info.get('log_tail', '')[-1500:]}")

            local_res_json = audio_out_dir / "tts_results.json"
            client.download_file(f"{remote_job_dir}/out/tts_results.json", local_res_json)
            res_data = json.loads(local_res_json.read_text(encoding="utf-8"))
            by_id = {}
            for seg in res_data["segments"]:
                wav_name = seg["file"]
                local_wav = audio_out_dir / wav_name
                client.download_file(f"{remote_job_dir}/out/{wav_name}", local_wav)
                seg["local_path"] = str(local_wav)
                by_id[seg["id"]] = seg
            client.exec_sync(f"rm -rf {remote_job_dir}")
            return by_id
        except Exception as exc:
            print(f"[WARN] RTX PRO 6000 TTS unavailable ({exc}); falling back to local PyTorch.", file=sys.stderr)

    req_data = {"voice_pack_dir": str(local_pack), "segments": segments_req}
    local_req_file.write_text(json.dumps(req_data, ensure_ascii=False, indent=2), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/rtx6000_gpu_engine.py"),
            "--task",
            "synthesize_voice",
            "--req",
            str(local_req_file),
            "--out",
            str(audio_out_dir),
        ],
        check=True,
    )
    res_data = json.loads((audio_out_dir / "tts_results.json").read_text(encoding="utf-8"))
    return {seg["id"]: {**seg, "local_path": str(audio_out_dir / seg["file"])} for seg in res_data["segments"]}
