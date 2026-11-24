"""AI Image Generation (RTX PRO 6000 PyTorch SDXL-Lightning) & Forensic Slide Card Renderer.

Replicates all 3 visual modes of 獨角獸那麼叫 (@unicorn-roar):
1. `ai_scene`     — Photorealistic / satirical editorial visuals generated via PyTorch on RTX PRO 6000
2. `evidence_doc` — Official regulation / news clipping sheet with crimson `#CC0000` marker underline
                    and top-left stroked `accessed:YYYYMMDD` source citation watermark
3. `kinetic_black`— Pure `#000000` contrast data/formula card (`#FFFFFF` + `#FFE600` + `#FF2424` + scale bar)
"""
from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from unicorn_roar.config import REPO_ROOT, get_runtime_endpoint_path
from unicorn_roar.discovery.web_crawler import ResearchDossier, SourceCitation
from unicorn_roar.remote.rtx6000_client import (
    RTX6000WorkerClient,
    RemoteWorkerEndpoint,
)
from unicorn_roar.scriptwriter.generator import EpisodeBlueprint, SceneSpec

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
                # In PingFang.ttc, index 2 or 0 provides crisp TC glyphs; fallback to index 0
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
    """Clean moody studio backdrop fallback if GPU node is offline."""
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
    """Batch-generate all `ai_scene` images on Colab RTX PRO 6000 (PyTorch SDXL-Lightning)."""
    visuals_dir = run_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    requests_list: list[dict[str, Any]] = []
    for sc in blueprint.long_video.scenes:
        if sc.scene_mode == "ai_scene":
            requests_list.append(
                {
                    "id": sc.scene_id,
                    "prompt": sc.image_prompt
                    + ", editorial documentary photography, ultra sharp, natural lighting, no text, no watermark",
                    "width": 1344,
                    "height": 768,
                    "seed": _seed_from_id(sc.scene_id),
                }
            )
    for sc in blueprint.short_video.scenes:
        if sc.scene_mode == "ai_scene":
            requests_list.append(
                {
                    "id": sc.scene_id,
                    "prompt": sc.image_prompt
                    + ", high contrast editorial close-up, dramatic lighting, photorealistic, no text, no watermark",
                    "width": 1024,
                    "height": 1024,
                    "seed": _seed_from_id(sc.scene_id),
                }
            )

    out_map: dict[str, Path] = {}
    if not requests_list:
        return out_map

    ep = RemoteWorkerEndpoint.from_file(get_runtime_endpoint_path())
    if ep is not None:
        try:
            client = RTX6000WorkerClient(ep)
            local_req = run_dir / "temp" / "img_req.json"
            local_req.parent.mkdir(parents=True, exist_ok=True)
            local_req.write_text(
                json.dumps({"images": requests_list}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            remote_job_dir = f"runs/{run_dir.name}/images"
            client.upload_file(REPO_ROOT / "scripts/rtx6000_gpu_engine.py", "scripts/rtx6000_gpu_engine.py")
            client.upload_file(local_req, f"{remote_job_dir}/img_req.json")
            jid = client.submit_job(
                f".venv/bin/python scripts/rtx6000_gpu_engine.py --task generate_images "
                f"--req {remote_job_dir}/img_req.json --out {remote_job_dir}/out",
                job_id=f"img_{run_dir.name}",
            )
            info = client.wait_job(jid, poll_interval=3.0, timeout=900.0)
            if info.get("rc") != 0:
                raise RuntimeError(f"RTX6000 image gen failed: {info.get('log_tail', '')[-1500:]}")
            res_json = visuals_dir / "image_results.json"
            client.download_file(f"{remote_job_dir}/out/image_results.json", res_json)
            res = json.loads(res_json.read_text(encoding="utf-8"))
            for item in res["images"]:
                local_png = visuals_dir / item["file"]
                client.download_file(f"{remote_job_dir}/out/{item['file']}", local_png)
                out_map[item["id"]] = local_png
            return out_map
        except Exception as exc:
            print(f"[WARN] RTX PRO 6000 image gen unavailable ({exc}); using fallback studio backdrop.", file=sys.stderr)

    for req in requests_list:
        p = visuals_dir / f"{req['id']}.png"
        _fallback_editorial_canvas(req["width"], req["height"], req["seed"]).save(p)
        out_map[req["id"]] = p
    return out_map


def draw_citation_watermark(
    draw: ImageDraw.ImageDraw,
    citation: SourceCitation | None,
    x: int = 14,
    y: int = 12,
    font_size: int = 24,
) -> None:
    """Render top-left white-with-black-stroke source citation (`... accessed:YYYYMMDD`)."""
    if citation is None:
        return
    font = load_cjk_font(font_size, bold=True)
    draw.text(
        (x, y),
        citation.onscreen_watermark,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=3,
        stroke_fill=(0, 0, 0, 255),
    )


def render_evidence_doc_card(
    scene: SceneSpec,
    citation: SourceCitation | None,
    width: int,
    height: int,
    out_path: Path,
    include_watermark: bool = True,
) -> Path:
    """Render white official regulation / news clipping sheet with crimson `#CC0000` marker underline."""
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    pad_x = int(width * 0.08)
    pad_y = int(height * 0.09)
    sheet_box = [pad_x, pad_y, width - pad_x, height - pad_y]
    draw.rounded_rectangle(sheet_box, radius=8, fill=(252, 252, 250), outline=(210, 210, 210), width=2)

    # Top metadata strip on the paper sheet
    meta_font = load_cjk_font(max(20, int(width * 0.017)), bold=False)
    head_font = load_cjk_font(max(28, int(width * 0.025)), bold=True)
    body_font = load_cjk_font(max(26, int(width * 0.023)), bold=False)

    inner_x = pad_x + int(width * 0.05)
    inner_w = (width - 2 * pad_x) - int(width * 0.10)
    cur_y = pad_y + int(height * 0.055)

    pub_label = (
        f"公開資料來源：{citation.publisher}｜發布日期：{citation.published_date}"
        if citation
        else "公開法規與統計資料摘錄"
    )
    draw.text((inner_x, cur_y), pub_label, font=meta_font, fill=(95, 95, 95))
    cur_y += int(height * 0.055)
    draw.line([(inner_x, cur_y), (inner_x + inner_w, cur_y)], fill=(205, 205, 205), width=2)
    cur_y += int(height * 0.035)

    # Heading wrap
    heading = scene.document_heading or (citation.title if citation else scene.narration[:28])
    chars_head = max(14, int(inner_w / max(28, int(width * 0.026))))
    for hline in textwrap.wrap(heading, width=chars_head)[:2]:
        draw.text((inner_x, cur_y), hline, font=head_font, fill=(22, 22, 22))
        cur_y += int(height * 0.062)

    cur_y += int(height * 0.025)

    # Body excerpt + red marker underline on target quote
    excerpt = scene.document_excerpt or (citation.excerpt if citation else scene.narration)
    quote = (scene.red_underline_quote or "").strip()
    if quote and quote not in excerpt:
        excerpt = f"{excerpt}「{quote}」"

    chars_body = max(16, int(inner_w / max(26, int(width * 0.024))))
    wrapped_lines = textwrap.wrap(excerpt, width=chars_body)[:6]
    line_step = int(height * 0.072)

    # Identify which wrapped line(s) overlap with `quote` (or default to middle high-impact line)
    underlined_any = False
    quote_fragments = [quote[i : i + 6] for i in range(0, max(1, len(quote) - 4), 4)] if len(quote) >= 6 else ([quote] if quote else [])

    for idx, bline in enumerate(wrapped_lines):
        draw.text((inner_x, cur_y), bline, font=body_font, fill=(32, 32, 32))
        bbox = draw.textbbox((inner_x, cur_y), bline, font=body_font)
        should_underline = any(frag and frag in bline for frag in quote_fragments)
        if should_underline:
            underlined_any = True
            uy = bbox[3] + 6
            draw.line([(bbox[0], uy), (bbox[2], uy)], fill=(208, 12, 18), width=7)
        cur_y += line_step

    if not underlined_any and wrapped_lines:
        target_idx = min(1, len(wrapped_lines) - 1)
        ty = cur_y - (len(wrapped_lines) - target_idx) * line_step
        bbox = draw.textbbox((inner_x, ty), wrapped_lines[target_idx], font=body_font)
        uy = bbox[3] + 6
        draw.line([(bbox[0], uy), (bbox[2], uy)], fill=(208, 12, 18), width=7)

    # Footer URL note inside sheet
    if citation and citation.url:
        url_short = citation.url if len(citation.url) <= 68 else citation.url[:65] + "..."
        foot_font = load_cjk_font(max(16, int(width * 0.013)), bold=False)
        draw.text(
            (inner_x, height - pad_y - int(height * 0.055)),
            f"查證連結：{url_short}",
            font=foot_font,
            fill=(120, 120, 120),
        )

    if include_watermark:
        draw_citation_watermark(draw, citation, x=14, y=10, font_size=max(20, int(width * 0.013)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def render_kinetic_black_card(
    scene: SceneSpec,
    citation: SourceCitation | None,
    width: int,
    height: int,
    out_path: Path,
    include_watermark: bool = False,
) -> Path:
    """Render pure `#000000` kinetic comparison / formula slide (`#FFFFFF` + `#FFE600` + `#FF2424`)."""
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    lines = scene.kinetic_lines
    font_size = max(46, int(width * (0.048 if len(lines) <= 2 else 0.040)))
    font = load_cjk_font(font_size, bold=True)

    line_gap = int(font_size * 1.55)
    total_h = len(lines) * line_gap
    start_y = max(int(height * 0.16), (height - total_h) // 2 - int(height * 0.06))

    palette = {
        "yellow": (255, 234, 0),
        "red": (255, 38, 38),
    }

    for idx, kl in enumerate(lines[:3]):
        full_txt = kl.text.strip()
        hl = (kl.highlight or "").strip()
        hl_color = palette.get(kl.highlight_color, (255, 234, 0))

        full_bbox = draw.textbbox((0, 0), full_txt, font=font)
        tw = full_bbox[2] - full_bbox[0]
        cur_x = max(40, (width - tw) // 2)
        cur_y = start_y + idx * line_gap

        if hl and hl in full_txt:
            pre, post = full_txt.split(hl, 1)
            if pre:
                draw.text((cur_x, cur_y), pre, font=font, fill=(255, 255, 255))
                cur_x += draw.textbbox((0, 0), pre, font=font)[2]
            draw.text((cur_x, cur_y), hl, font=font, fill=hl_color)
            cur_x += draw.textbbox((0, 0), hl, font=font)[2]
            if post:
                draw.text((cur_x, cur_y), post, font=font, fill=(255, 255, 255))
        else:
            # Auto-highlight numbers/percentages in bright yellow like I1qvoKvTyJU/frame_480s.jpg
            import re

            tokens = re.split(r"(\d+(?:\.\d+)?%?萬?元?)", full_txt)
            for tok in tokens:
                if not tok:
                    continue
                is_num = bool(re.fullmatch(r"\d+(?:\.\d+)?%?萬?元?", tok))
                color = (255, 234, 0) if is_num else (255, 255, 255)
                draw.text((cur_x, cur_y), tok, font=font, fill=color)
                cur_x += draw.textbbox((0, 0), tok, font=font)[2]

    # Signature subtle blue-cyan comparison ruler bar below kinetic comparison (matching frame_90s.jpg!)
    ruler_y = min(height - int(height * 0.22), start_y + len(lines[:3]) * line_gap + int(height * 0.04))
    rx0, rx1 = int(width * 0.08), int(width * 0.92)
    rh = max(28, int(height * 0.055))
    draw.rounded_rectangle([rx0, ruler_y, rx1, ruler_y + rh], radius=6, fill=(115, 192, 214))
    draw.line([(rx0, ruler_y + rh // 2), (rx1, ruler_y + rh // 2)], fill=(228, 246, 252), width=3)
    ticks = 12
    for t_i in range(1, ticks):
        tx = rx0 + (rx1 - rx0) * t_i // ticks
        draw.line([(tx, ruler_y), (tx, ruler_y + rh // 3)], fill=(35, 65, 78), width=2)
        draw.line([(tx, ruler_y + (2 * rh) // 3), (tx, ruler_y + rh)], fill=(35, 65, 78), width=2)

    if include_watermark and citation:
        draw_citation_watermark(draw, citation, x=14, y=10, font_size=max(20, int(width * 0.013)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path
