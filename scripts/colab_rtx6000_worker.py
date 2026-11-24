#!/usr/bin/env python3
"""Isolated Colab RTX PRO 6000 Heavy Worker Bootstrap & Daemon for 獨角獸那麼叫 (@unicorn-roar).

Designed to coexist peacefully with other sessions on shared high-VRAM Colab instances:
- Uses dedicated workspace: /local-scratch/unicorn_roar_g4c (or /content/unicorn_roar_g4c)
- Uses dedicated port: 19777
- Never touches processes outside its own daemon/tunnel binary names.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

PORT = int(os.environ.get("UNICORN_WORKER_PORT", "19777"))
SCRATCH = Path("/local-scratch")
WORK = (
    (SCRATCH / "unicorn_roar_g4c")
    if (SCRATCH.exists() and os.access(SCRATCH, os.W_OK))
    else Path("/content/unicorn_roar_g4c")
)
WORK.mkdir(parents=True, exist_ok=True)
LINK = Path("/content/unicorn_roar_g4c")
if LINK != WORK and not LINK.exists():
    try:
        LINK.symlink_to(WORK)
    except OSError:
        pass

(WORK / "bin").mkdir(parents=True, exist_ok=True)
(WORK / "jobs").mkdir(parents=True, exist_ok=True)

TOKEN_FILE = WORK / "token.txt"
if TOKEN_FILE.exists() and TOKEN_FILE.read_text().strip():
    TOKEN = TOKEN_FILE.read_text().strip()
else:
    TOKEN = f"ur_rtx6000_{secrets.token_urlsafe(24)}"
    TOKEN_FILE.write_text(TOKEN)

DAEMON_CODE = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WORK = Path(os.environ.get("UNICORN_WORK_DIR", "/content/unicorn_roar_g4c")).resolve()
TOKEN = ""
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _env() -> dict[str, str]:
    e = dict(os.environ)
    ps = [
        str(WORK / ".venv/bin"),
        str(WORK / "bin"),
        "/usr/local/cuda/bin",
        "/usr/local/bin",
        e.get("PATH", ""),
    ]
    e["PATH"] = ":".join(p for p in ps if p)
    e["PYTHONUNBUFFERED"] = "1"
    e["HF_HOME"] = str(WORK / "hf_cache")
    return e


def _safe(rel: str) -> Path:
    p = (WORK / rel.lstrip("/")).resolve()
    if p != WORK and WORK not in p.parents:
        raise ValueError("path outside workspace")
    return p


def _run_async(jid: str, cmd: str, cwd: Path, log_path: Path) -> None:
    t0 = time.time()
    with log_path.open("wb") as f:
        pr = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=_env(),
            start_new_session=True,
        )
        with LOCK:
            JOBS[jid]["pid"] = pr.pid
        rc = pr.wait()
    with LOCK:
        JOBS[jid].update(
            status="done" if rc == 0 else "failed",
            rc=rc,
            elapsed=round(time.time() - t0, 2),
        )


class WorkerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def _auth(self) -> bool:
        if self.headers.get("Authorization") == f"Bearer {TOKEN}":
            return True
        self._json(401, {"error": "unauthorized"})
        return False

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if not self._auth():
            return
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/health":
            with LOCK:
                running = [k for k, v in JOBS.items() if v.get("status") == "running"]
            try:
                gpu = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total,memory.free,utilization.gpu",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).strip()
            except Exception as exc:
                gpu = f"unavailable: {exc}"
            du = shutil.disk_usage(str(WORK))
            return self._json(
                200,
                {
                    "ok": True,
                    "service": "unicorn-roar-rtx6000-worker",
                    "work": str(WORK),
                    "gpu": gpu,
                    "cpus": os.cpu_count(),
                    "disk_free_gb": round(du.free / (1024**3), 2),
                    "running_jobs": running,
                },
            )
        if u.path == "/job":
            jid = (q.get("id") or [""])[0]
            tail = int((q.get("tail") or ["8000"])[0])
            with LOCK:
                info = dict(JOBS.get(jid) or {})
            if not info:
                return self._json(404, {"error": "job_not_found", "id": jid})
            lp = Path(info["log_path"])
            if lp.exists():
                raw = lp.read_bytes()
                info["log_tail"] = raw[-tail:].decode("utf-8", errors="replace")
                info["log_bytes"] = len(raw)
            else:
                info["log_tail"] = ""
                info["log_bytes"] = 0
            return self._json(200, info)
        if u.path == "/download":
            rel = (q.get("path") or [""])[0]
            try:
                target = _safe(rel)
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            if not target.is_file():
                return self._json(404, {"error": "file_not_found", "path": rel})
            size = target.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with target.open("rb") as f:
                shutil.copyfileobj(f, self.wfile, length=1024 * 1024)
            return
        return self._json(404, {"error": "unknown_route"})

    def do_POST(self) -> None:
        if not self._auth():
            return
        u = urlparse(self.path)
        q = parse_qs(u.query)
        length = int(self.headers.get("Content-Length") or "0")

        if u.path == "/upload":
            rel = (q.get("path") or [""])[0]
            if not rel:
                return self._json(400, {"error": "missing_path"})
            try:
                target = _safe(rel)
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            target.parent.mkdir(parents=True, exist_ok=True)
            remaining = length
            written = 0
            with target.open("wb") as out:
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
                    remaining -= len(chunk)
            return self._json(200, {"ok": True, "path": str(target), "bytes": written})

        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        payload = json.loads(raw_body.decode("utf-8"))

        if u.path == "/exec_sync":
            cmd = payload["cmd"]
            cwd = _safe(payload.get("cwd", ""))
            timeout = float(payload.get("timeout", 90))
            t0 = time.time()
            try:
                cp = subprocess.run(
                    ["bash", "-lc", cmd],
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    env=_env(),
                    timeout=timeout,
                )
                return self._json(
                    200,
                    {
                        "ok": cp.returncode == 0,
                        "rc": cp.returncode,
                        "stdout": cp.stdout[-32000:],
                        "stderr": cp.stderr[-32000:],
                        "elapsed": round(time.time() - t0, 2),
                    },
                )
            except subprocess.TimeoutExpired as exc:
                return self._json(
                    504,
                    {
                        "ok": False,
                        "error": "timeout",
                        "stdout": (exc.stdout or "")[-16000:] if isinstance(exc.stdout, str) else "",
                        "stderr": (exc.stderr or "")[-16000:] if isinstance(exc.stderr, str) else "",
                    },
                )

        if u.path == "/job":
            cmd = payload["cmd"]
            cwd = _safe(payload.get("cwd", ""))
            cwd.mkdir(parents=True, exist_ok=True)
            jid = payload.get("id") or f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            lp = WORK / "jobs" / f"{jid}.log"
            meta = {
                "id": jid,
                "cmd": cmd,
                "cwd": str(cwd),
                "status": "running",
                "started_at": time.time(),
                "log_path": str(lp),
            }
            with LOCK:
                JOBS[jid] = meta
            t = threading.Thread(target=_run_async, args=(jid, cmd, cwd, lp), daemon=True)
            t.start()
            return self._json(200, {"ok": True, "id": jid, "status": "running"})

        return self._json(404, {"error": "unknown_route"})


def main() -> None:
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=19777)
    ap.add_argument("--token", required=True)
    args = ap.parse_args()
    TOKEN = args.token
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), WorkerHandler)
    srv.serve_forever()


if __name__ == "__main__":
    main()
'''


def bootstrap() -> None:
    daemon_py = WORK / f"unicorn_roar_daemon_{PORT}.py"
    daemon_py.write_text(DAEMON_CODE, encoding="utf-8")
    daemon_py.chmod(0o755)

    subprocess.run(f"pkill -f '[u]nicorn_roar_daemon_{PORT}.py' || true", shell=True)
    subprocess.run(f"pkill -f '[u]nicorn_roar_cf_{PORT}' || true", shell=True)
    time.sleep(0.5)

    env = dict(os.environ)
    env["UNICORN_WORK_DIR"] = str(WORK)
    daemon_log = WORK / "daemon.log"
    subprocess.Popen(
        ["python3", str(daemon_py), "--port", str(PORT), "--token", TOKEN],
        cwd=str(WORK),
        stdin=subprocess.DEVNULL,
        stdout=daemon_log.open("w"),
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    time.sleep(0.8)

    cf_bin = WORK / "bin" / f"unicorn_roar_cf_{PORT}"
    if not cf_bin.exists():
        urllib.request.urlretrieve(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            str(cf_bin),
        )
        cf_bin.chmod(0o755)

    cf_log = WORK / "tunnel.log"
    cf_log.write_text("")
    subprocess.Popen(
        [
            str(cf_bin),
            "tunnel",
            "--protocol",
            "http2",
            "--no-autoupdate",
            "--url",
            f"http://127.0.0.1:{PORT}",
        ],
        cwd=str(WORK),
        stdin=subprocess.DEVNULL,
        stdout=cf_log.open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    pub_url = ""
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    for _ in range(50):
        time.sleep(0.5)
        txt = cf_log.read_text(errors="ignore") if cf_log.exists() else ""
        m = pat.search(txt)
        if m and "Registered" in txt:
            pub_url = m.group(0)
            break
        if m:
            pub_url = m.group(0)

    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:
        gpu = f"unavailable: {exc}"

    du = shutil.disk_usage(str(WORK))
    info = {
        "url": pub_url,
        "token": TOKEN,
        "port": PORT,
        "work": str(WORK),
        "gpu": gpu,
        "cpus": os.cpu_count(),
        "disk_free_gb": round(du.free / (1024**3), 2),
    }
    (WORK / "endpoint.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("UNICORN_RTX6000_READY=" + json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    bootstrap()
