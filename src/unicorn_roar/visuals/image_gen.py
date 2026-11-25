"""V2 Visual Asset Generator: Real Wikimedia Documentary Photos + 24-Step SDXL + Progressive Kinetic Cards."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from unicorn_roar.config import REPO_ROOT, get_runtime_endpoint_path
from unicorn_roar.discovery.web_crawler import SourceCitation
from unicorn_roar.remote.rtx6000_client import (
    RTX6000WorkerClient,
    RemoteWorkerEndpoint,
)
from unicorn_roar.scriptwriter.generator import EpisodeBlueprint, SceneSpec
from unicorn_roar.visuals.broll_fetcher import fetch_wikimedia_photo

FONT_CANDIDATES_BOLD = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
]

FONT_CANDIDATES_SANS = [
    "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
]


def load_cjk_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for fp in FONT_CANDIDATES_BOLD:
        p = Path(fp)
        if p.exists():
            try:
                # In macOS PingFang.ttc: index 2 is PingFang TC Semibold/Medium, index 0 is Regular
                idx = 2 if ("PingFang" in p.name and bold) else 0
                return ImageFont.truetype(str(p), size=size, index=idx)
            except Exception:
                try:
                    return ImageFont.truetype(str(p), size=size, index=0)
                except Exception:
                    continue
    return ImageFont.load_default()


def load_latin_italic_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for fp in FONT_CANDIDATES_SANS:
        p = Path(fp)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    return load_cjk_font(size, bold=False)


def _seed_from_id(scene_id: str) -> int:
    return int(hashlib.sha256(scene_id.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)


def _fallback_editorial_canvas(width: int, height: int, seed: int) -> Image.Image:
    img = Image.new("RGB", (width, height), (18, 22, 30))
    draw = ImageDraw.Draw(img)
    accents = [(42, 68, 94), (78, 52, 38), (32, 74, 68), (68, 44, 76)]
    c = accents[seed % len(accents)]
    for i in range(height):
        t = i / max(height - 1, 1)
        r = int(16 * (1 - t) + c[0] * t)
        g = int(20 * (1 - t) + c[1] * t)
        b = int(28 * (1 - t) + c[2] * t)
        draw.line([(0, i), (width, i)], fill=(r, g, b))
    return img.filter(ImageFilter.GaussianBlur(radius=2))


def generate_episode_ai_images(
    blueprint: EpisodeBlueprint,
    run_dir: Path,
) -> dict[str, Path]:
    """Blend real Wikimedia Commons documentary photos + 24-step SDXL Base 1.0 renders."""
    visuals_dir = run_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    out_map: dict[str, Path] = {}
    sdxl_requests: list[dict[str, Any]] = []

    all_ai_scenes: list[tuple[int, SceneSpec, int, int]] = []
    for i, sc in enumerate(blueprint.long_video.scenes):
        if sc.scene_mode == "ai_scene":
            all_ai_scenes.append((i, sc, 1344, 768))
    for i, sc in enumerate(blueprint.short_video.scenes):
        if sc.scene_mode == "ai_scene":
            all_ai_scenes.append((i, sc, 1024, 1024))

    for idx, sc, w, h in all_ai_scenes:
        cached = visuals_dir / f"{sc.scene_id}.png"
        if cached.exists() and cached.stat().st_size > 15_000:
            out_map[sc.scene_id] = cached
            continue
        # Try real documentary photography from Wikimedia on alternating scenes when query is provided
        if sc.wikimedia_query and idx % 2 == 0:
            fetched = fetch_wikimedia_photo(sc.wikimedia_query, cached)
            if fetched is not None:
                out_map[sc.scene_id] = fetched
                continue
        sdxl_requests.append(
            {
                "id": sc.scene_id,
                "prompt": (
                    f"{sc.image_prompt}, documentary editorial photography, Sony A7R V 35mm lens, "
                    "natural realistic lighting, ultra sharp focus, clean unbranded surface, no text, no letters"
                ),
                "width": w,
                "height": h,
                "seed": _seed_from_id(sc.scene_id),
            }
        )

    if not sdxl_requests:
        return out_map

    ep = RemoteWorkerEndpoint.from_file(get_runtime_endpoint_path())
    if ep is not None:
        try:
            client = RTX6000WorkerClient(ep)
            local_req = run_dir / "temp" / "img_req.json"
            local_req.parent.mkdir(parents=True, exist_ok=True)
            local_req.write_text(
                json.dumps({"images": sdxl_requests}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            remote_job_dir = f"runs/{run_dir.name}/images"
            client.upload_file(REPO_ROOT / "scripts/rtx6000_gpu_engine.py", "scripts/rtx6000_gpu_engine.py")
            client.upload_file(local_req, f"{remote_job_dir}/img_req.json")
            jid = client.submit_job(
                f".venv/bin/python scripts/rtx6000_gpu_engine.py --task generate_images "
                f"--req {remote_job_dir}/img_req.json --out {remote_job_dir}/out",
                job_id=f"img24_{run_dir.name}",
            )
            info = client.wait_job(jid, poll_interval=3.0, timeout=900.0)
            if info.get("rc") != 0:
                raise RuntimeError(f"RTX6000 24-step SDXL failed: {info.get('log_tail', '')[-1500:]}")
            res_json = visuals_dir / "image_results.json"
            client.download_file(f"{remote_job_dir}/out/image_results.json", res_json)
            res = json.loads(res_json.read_text(encoding="utf-8"))
            for item in res["images"]:
                local_png = visuals_dir / item["file"]
                client.download_file(f"{remote_job_dir}/out/{item['file']}", local_png)
                out_map[item["id"]] = local_png
            client.exec_sync(f"rm -rf {remote_job_dir}")
            return out_map
        except Exception as exc:
            print(f"[WARN] RTX PRO 6000 SDXL unavailable ({exc}); using studio backdrop.", file=sys.stderr)

    for req in sdxl_requests:
        p = visuals_dir / f"{req['id']}.png"
        _fallback_editorial_canvas(req["width"], req["height"], req["seed"]).save(p)
        out_map[req["id"]] = p
    return out_map


def draw_citation_watermark(
    draw: ImageDraw.ImageDraw,
    citation: SourceCitation | None,
    x: int = 8,
    y: int = 5,
    font_size: int = 16,
) -> None:
    if citation is None:
        return
    font = load_cjk_font(font_size, bold=True)
    draw.text(
        (x, y),
        citation.onscreen_watermark,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0, 255),
    )


def render_kinetic_black_beats(
    scene: SceneSpec,
    width: int,
    height: int,
    out_prefix: Path,
) -> list[Path]:
    """Render authentic @unicorn-roar incremental equation/ratio `#000000` boards (`JZg7GtBEKFM` & `I1qvoKvTyJU`).

    - Pure `#000000` black canvas, ZERO fake blue ruler bar!
    - Left-aligned structured formula grid (`x = 9%` width), `54px` crisp bold typography.
    - Numbers & key deltas highlighted in `#FFD600` (gold), `#00E640` (green), or `#FF2E2E` (red).
    """
    lines = scene.kinetic_lines[:4]
    is_vertical = height > width
    font_size = max(40, int(width * (0.048 if is_vertical else 0.033)))
    font = load_cjk_font(font_size, bold=True)
    line_gap = int(font_size * 1.68)
    total_h = max(1, len(lines)) * line_gap
    start_y = max(int(height * 0.18), (height - total_h) // 2 - int(height * 0.06))
    left_x = int(width * (0.07 if is_vertical else 0.085))
    palette = {
        "yellow": (255, 214, 0),
        "red": (255, 46, 46),
        "green": (0, 230, 64),
    }

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    total_beats = max(2, len(lines))
    beats: list[Path] = []

    for beat_idx in range(1, total_beats + 1):
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        vis_lines = lines[: min(len(lines), beat_idx)]

        for idx, kl in enumerate(vis_lines):
            full_txt = kl.text.strip()
            hl = (kl.highlight or "").strip()
            hl_color = palette.get(kl.highlight_color, (255, 214, 0))
            cur_x = left_x
            cur_y = start_y + idx * line_gap

            if hl and hl in full_txt:
                pre, post = full_txt.split(hl, 1)
                if pre:
                    draw.text((cur_x, cur_y), pre, font=font, fill=(255, 255, 255))
                    cur_x += draw.textbbox((0, 0), pre, font=font)[2]
                # Draw subtle green/red highlight badge on final line's punch number if short
                if idx == len(lines) - 1 and len(hl) <= 9 and kl.highlight_color == "red":
                    hb = draw.textbbox((0, 0), hl, font=font)
                    hw, hh = hb[2] - hb[0], hb[3] - hb[1]
                    draw.rounded_rectangle(
                        [cur_x - 6, cur_y + 2, cur_x + hw + 10, cur_y + hh + 14],
                        radius=6,
                        fill=(200, 18, 24),
                    )
                    draw.text((cur_x + 2, cur_y), hl, font=font, fill=(255, 255, 255))
                    cur_x += hw + 16
                else:
                    draw.text((cur_x, cur_y), hl, font=font, fill=hl_color)
                    cur_x += draw.textbbox((0, 0), hl, font=font)[2]
                if post:
                    draw.text((cur_x, cur_y), post, font=font, fill=(255, 255, 255))
            else:
                tokens = re.split(r"(\d[\d,\.]*%?萬?元?倍?公升?克?人?|[xX×÷=+\-∼~])", full_txt)
                for tok in tokens:
                    if not tok:
                        continue
                    if re.fullmatch(r"\d[\d,\.]*%?萬?元?倍?公升?克?人?", tok):
                        color = (0, 230, 64) if idx == 1 else (255, 214, 0)
                    else:
                        color = (255, 255, 255)
                    draw.text((cur_x, cur_y), tok, font=font, fill=color)
                    cur_x += draw.textbbox((0, 0), tok, font=font)[2]

        bp = out_prefix.parent / f"{out_prefix.name}_kbeat{beat_idx}.png"
        canvas.save(bp)
        beats.append(bp)

    if len(beats) < 3 and beats:
        p_extra = out_prefix.parent / f"{out_prefix.name}_kbeat3.png"
        Image.open(beats[-1]).save(p_extra)
        beats.append(p_extra)

    return beats


# Backwards-compatible helpers for unit tests
def render_kinetic_black_card(
    scene: SceneSpec,
    citation: SourceCitation | None,
    width: int,
    height: int,
    out_path: Path,
    include_watermark: bool = False,
) -> Path:
    beats = render_kinetic_black_beats(scene, width, height, out_path.with_suffix(""))
    final = Image.open(beats[-1])
    if include_watermark and citation:
        draw_citation_watermark(ImageDraw.Draw(final), citation)
    final.save(out_path)
    return out_path
