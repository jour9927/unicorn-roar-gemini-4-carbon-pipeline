"""Real Documentary Photography & CC0/Wikimedia Media Fetcher for 獨角獸那麼叫 (@unicorn-roar).

Provides real-world product, food, retail, and industrial photographs from Wikimedia Commons
so episodes blend genuine real-world photography with high-step RTX PRO 6000 renders and
Chrome webpage/document captures.
"""
from __future__ import annotations

import io
from pathlib import Path

import httpx
from PIL import Image

WIKI_API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "UnicornRoarDocumentaryPipeline/2.0 (Educational Public Analysis)"}


def fetch_wikimedia_photo(query: str, out_path: Path) -> Path | None:
    """Search Wikimedia Commons for a clean real-world photograph (>= 900px wide) matching `query`."""
    try:
        with httpx.Client(timeout=12.0, follow_redirects=True, headers=HEADERS) as client:
            resp = client.get(
                WIKI_API,
                params={
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": f"filetype:bitmap {query}",
                    "gsrnamespace": "6",
                    "gsrlimit": "6",
                    "prop": "imageinfo",
                    "iiprop": "url|size|mime",
                    "iiurlwidth": "1600",
                    "format": "json",
                },
            )
            if resp.status_code != 200:
                return None
            pages = resp.json().get("query", {}).get("pages", {})
            for p in pages.values():
                infos = p.get("imageinfo") or []
                if not infos:
                    continue
                ii = infos[0]
                mime = ii.get("mime", "")
                w = int(ii.get("width") or 0)
                h = int(ii.get("height") or 0)
                if mime not in ("image/jpeg", "image/png") or w < 900 or h < 550:
                    continue
                img_url = ii.get("thumburl") or ii.get("url")
                if not img_url:
                    continue
                ir = client.get(img_url, timeout=12.0)
                if ir.status_code == 200 and len(ir.content) > 25_000:
                    im = Image.open(io.BytesIO(ir.content)).convert("RGB")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    im.save(out_path)
                    return out_path
    except Exception:
        return None
    return None
