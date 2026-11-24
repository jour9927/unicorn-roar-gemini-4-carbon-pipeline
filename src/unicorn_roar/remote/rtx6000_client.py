"""High-speed Remote GPU Worker Client for Colab RTX PRO 6000 & Tailscale B850 Fallback."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass
class RemoteWorkerEndpoint:
    url: str
    token: str
    port: int = 19777
    work: str = "/content/unicorn_roar_g4c"
    gpu: str = ""

    @classmethod
    def from_file(cls, path: Path) -> "RemoteWorkerEndpoint | None":
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("url") or not data.get("token"):
            return None
        return cls(
            url=data["url"].rstrip("/"),
            token=data["token"],
            port=int(data.get("port", 19777)),
            work=data.get("work", "/content/unicorn_roar_g4c"),
            gpu=data.get("gpu", ""),
        )


class RTX6000WorkerClient:
    """Authenticated HTTP client for the isolated Colab RTX PRO 6000 worker."""

    def __init__(self, endpoint: RemoteWorkerEndpoint, timeout: float = 120.0) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {endpoint.token}"}

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=15.0, headers=self._headers) as client:
            resp = client.get(f"{self.endpoint.url}/health")
            resp.raise_for_status()
            return resp.json()

    def exec_sync(self, cmd: str, cwd: str = "", timeout: float = 90.0) -> dict[str, Any]:
        with httpx.Client(timeout=timeout + 15.0, headers=self._headers) as client:
            resp = client.post(
                f"{self.endpoint.url}/exec_sync",
                json={"cmd": cmd, "cwd": cwd, "timeout": timeout},
            )
            resp.raise_for_status()
            return resp.json()

    def submit_job(self, cmd: str, cwd: str = "", job_id: str | None = None) -> str:
        payload: dict[str, Any] = {"cmd": cmd, "cwd": cwd}
        if job_id:
            payload["id"] = job_id
        with httpx.Client(timeout=30.0, headers=self._headers) as client:
            resp = client.post(f"{self.endpoint.url}/job", json=payload)
            resp.raise_for_status()
            return resp.json()["id"]

    def get_job(self, job_id: str, tail: int = 8000) -> dict[str, Any]:
        with httpx.Client(timeout=30.0, headers=self._headers) as client:
            resp = client.get(
                f"{self.endpoint.url}/job",
                params={"id": job_id, "tail": str(tail)},
            )
            resp.raise_for_status()
            return resp.json()

    def wait_job(
        self,
        job_id: str,
        poll_interval: float = 3.0,
        timeout: float = 1800.0,
    ) -> dict[str, Any]:
        deadline = time.time() + timeout
        last_info: dict[str, Any] = {}
        while time.time() < deadline:
            last_info = self.get_job(job_id)
            if last_info.get("status") in ("done", "failed"):
                return last_info
            time.sleep(poll_interval)
        raise TimeoutError(f"Remote GPU job {job_id} timed out after {timeout}s: {last_info}")

    def upload_file(self, local_path: Path, remote_rel_path: str) -> dict[str, Any]:
        data = local_path.read_bytes()
        with httpx.Client(timeout=300.0, headers=self._headers) as client:
            resp = client.post(
                f"{self.endpoint.url}/upload",
                params={"path": remote_rel_path},
                content=data,
                headers={**self._headers, "Content-Type": "application/octet-stream"},
            )
            resp.raise_for_status()
            return resp.json()

    def upload_bytes(self, data: bytes, remote_rel_path: str) -> dict[str, Any]:
        with httpx.Client(timeout=120.0, headers=self._headers) as client:
            resp = client.post(
                f"{self.endpoint.url}/upload",
                params={"path": remote_rel_path},
                content=data,
                headers={**self._headers, "Content-Type": "application/octet-stream"},
            )
            resp.raise_for_status()
            return resp.json()

    def download_file(self, remote_rel_path: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=300.0, headers=self._headers) as client:
            with client.stream(
                "GET",
                f"{self.endpoint.url}/download",
                params={"path": remote_rel_path},
            ) as resp:
                resp.raise_for_status()
                with local_path.open("wb") as out:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                        out.write(chunk)
        return local_path
