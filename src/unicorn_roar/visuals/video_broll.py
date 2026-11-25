"""V3 Live Commercial HD Video B-Roll & Signature Reaction Meme Engine for 獨角獸那麼叫 (@unicorn-roar).

Replaces frozen static AI stills with real 30fps moving live-action HD `.mp4` footage
(sourced from Mixkit's royalty-free commercial HD CDN) plus the channel's signature
reaction meme cutaways (`Huh?` white cat, fighting cats, shelf cat, red F mark, confused chart guy).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import httpx

from unicorn_roar.config import ASSETS_DIR

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

PILLAR_VIDEO_SLUGS: dict[str, list[str]] = {
    "food_labeling": [
        "olive-oil",
        "cooking-oil",
        "supermarket",
        "grocery-store",
        "cooking",
        "salad",
        "kitchen",
        "food-factory",
        "laboratory",
        "bottles",
        "shopping-cart",
        "chef",
    ],
    "consumer_pricing": [
        "supermarket",
        "coffee-shop",
        "cashier",
        "money",
        "shopping",
        "restaurant",
        "bakery",
        "receipt",
        "credit-card",
        "cafe",
    ],
    "wage_labor_stats": [
        "office",
        "money",
        "working",
        "factory",
        "computer",
        "city-traffic",
        "calculator",
        "business-meeting",
        "commute",
        "semiconductor",
    ],
    "industry_policy": [
        "cargo-ship",
        "factory",
        "city-skyline",
        "money",
        "office",
        "traffic",
        "construction",
        "port",
        "technology",
        "stock-market",
    ],
}


def get_broll_cache_dir() -> Path:
    d = ASSETS_DIR / "broll_video_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_meme_pack_dir() -> Path:
    return ASSETS_DIR / "meme_pack"


def load_meme_clips() -> tuple[list[Path], Path | None]:
    """Return list of clean 1080p@30fps reaction meme `.mp4` clips and `huh_sfx.wav`."""
    mp = get_meme_pack_dir()
    manifest_p = mp / "meme_manifest.json"
    if not manifest_p.exists():
        return [], None
    data = json.loads(manifest_p.read_text(encoding="utf-8"))
    clips = [mp / item["file"] for item in data.get("clips", []) if (mp / item["file"]).exists()]
    sfx_name = data.get("sfx", {}).get("huh")
    sfx_p = (mp / sfx_name) if (sfx_name and (mp / sfx_name).exists()) else None
    return clips, sfx_p


def _discover_mixkit_video_ids(slug: str, client: httpx.Client, max_ids: int = 8) -> list[str]:
    slug_clean = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-")
    if not slug_clean:
        return []
    try:
        r = client.get(f"https://mixkit.co/free-stock-video/{slug_clean}/")
        if r.status_code != 200:
            return []
        vids = list(
            dict.fromkeys(
                re.findall(r"https://assets\.mixkit\.co/videos/([0-9]+)/[0-9]+-[a-z0-9-]+\.mp4", r.text)
            )
        )
        return vids[:max_ids]
    except Exception:
        return []


def _download_mixkit_hd_mp4(vid_id: str, client: httpx.Client, out_dir: Path) -> Path | None:
    cached = out_dir / f"mixkit_{vid_id}.mp4"
    if cached.exists() and cached.stat().st_size > 150_000:
        return cached

    for suffix in ("720", "1080", "large", "360"):
        url = f"https://assets.mixkit.co/videos/{vid_id}/{vid_id}-{suffix}.mp4"
        try:
            with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    continue
                tmp = out_dir / f"mixkit_{vid_id}.part"
                total = 0
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=512 * 1024):
                        fh.write(chunk)
                        total += len(chunk)
                        if total > 32 * 1024 * 1024:  # Cap huge raw files
                            break
                if total > 150_000:
                    tmp.replace(cached)
                    return cached
                tmp.unlink(missing_ok=True)
        except Exception:
            continue
    return None


def fetch_episode_video_broll_pool(
    pillar: str,
    extra_keywords: list[str] | None = None,
    target_count: int = 20,
) -> list[Path]:
    """Fetch a diverse pool of 20+ real commercial HD live-action `.mp4` B-roll clips."""
    cache_dir = get_broll_cache_dir()
    slugs: list[str] = []
    if extra_keywords:
        for kw in extra_keywords:
            s = re.sub(r"[^a-z0-9]+", "-", kw.lower()).strip("-")
            if len(s) >= 3 and s not in slugs:
                slugs.append(s)
    for default_s in PILLAR_VIDEO_SLUGS.get(pillar, PILLAR_VIDEO_SLUGS["food_labeling"]):
        if default_s not in slugs:
            slugs.append(default_s)

    collected: list[Path] = []
    seen_ids: set[str] = set()

    with httpx.Client(
        timeout=15.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        # Interleave 2-3 videos per slug first for maximum topical variety!
        per_slug_ids: list[list[str]] = []
        for s in slugs[:12]:
            ids = _discover_mixkit_video_ids(s, client, max_ids=5)
            if ids:
                per_slug_ids.append(ids)

        for round_idx in range(5):
            for id_list in per_slug_ids:
                if len(collected) >= target_count:
                    return collected
                if round_idx < len(id_list):
                    vid_id = id_list[round_idx]
                    if vid_id in seen_ids:
                        continue
                    seen_ids.add(vid_id)
                    dl = _download_mixkit_hd_mp4(vid_id, client, cache_dir)
                    if dl is not None:
                        collected.append(dl)

    # Fallback to any cached MP4s if network is restricted
    if len(collected) < 6:
        for existing in sorted(cache_dir.glob("mixkit_*.mp4")):
            if existing not in collected and existing.stat().st_size > 150_000:
                collected.append(existing)
            if len(collected) >= target_count:
                break
    return collected


def pick_broll_clip(pool: list[Path], key: str, avoid_prev: Path | None = None) -> Path | None:
    if not pool:
        return None
    if len(pool) == 1:
        return pool[0]
    idx = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % len(pool)
    chosen = pool[idx]
    if avoid_prev is not None and chosen == avoid_prev:
        chosen = pool[(idx + 1) % len(pool)]
    return chosen
