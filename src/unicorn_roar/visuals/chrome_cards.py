"""V2 Real Browser DOM Renderer (Google Chrome 158 + Playwright) for 獨角獸那麼叫 (@unicorn-roar).

Eliminates fake Pillow white boxes by rendering authentic HTML5/CSS3 layouts in headless Chrome 158
and returning exact DOM `getClientRects()` coordinates of `#hl-target` so the video composer
executes @unicorn-roar's signature 2-beat camera move:
- Beat 1: Full Webpage / Official 公文 / Portal Overview (`1.0x`)
- Beat 2: Crisp 1.42x Punch-In Zoom centered on `#hl-target` + Progressive Red Marker Underline!

Supported authentic templates:
1. `news_portal`        — Taiwanese news site (CNA / UDN / LTN / PTS / NOWnews / foodNEXT)
2. `official_gov_doc`   — Formal ROC Government `函 / 公告` (受文者 / 發文字號 / 主旨 / 說明一二三)
3. `gov_web_portal`     — Official `.gov.tw` Ministry Portal (MOHW / FDA / MOA / DGBAS)
4. `product_comparison` — Split comparison / price-jump delta card (`包裝宣稱 vs 實際成分` / `↗ +18%`)
5. `netizen_comments`   — Progressive YouTube / Threads Taiwanese netizen comment wall
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from playwright.sync_api import Browser, sync_playwright

from unicorn_roar.discovery.web_crawler import SourceCitation
from unicorn_roar.visuals.image_gen import draw_citation_watermark

CHROME_EXEC = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

BRAND_COLORS = [
    ("中央社", "#003B73", "CNA 中央通訊社"),
    ("公視", "#9E1B32", "PNN 公視新聞網"),
    ("聯合", "#E65500", "UDN 聯合新聞網"),
    ("自由", "#005596", "LTN 自由時報"),
    ("食力", "#2E7D32", "foodNEXT 食力"),
    ("上下游", "#388E3C", "上下游 News&Market"),
    ("TVBS", "#0047AB", "TVBS 新聞網"),
    ("三立", "#0B5394", "SETN 三立新聞網"),
    ("Yahoo", "#5F01D1", "Yahoo奇摩新聞"),
]


def _pick_brand(publisher: str) -> tuple[str, str]:
    for kw, color, label in BRAND_COLORS:
        if kw.lower() in publisher.lower():
            return color, label
    return "#1A4B84", publisher or "台灣焦點新聞"


def _inject_highlight_span(paragraphs: list[str], quote: str) -> list[str]:
    q = (quote or "").strip(" 「」『』。，！？")
    out = []
    marked = False
    for p in paragraphs:
        esc_p = html.escape(p)
        if not marked and q and len(q) >= 4:
            esc_q = html.escape(q)
            if esc_q in esc_p:
                esc_p = esc_p.replace(esc_q, f'<span id="hl-target">{esc_q}</span>', 1)
                marked = True
            else:
                # Try 6-char prefix match
                sub_q = html.escape(q[: min(len(q), 10)])
                pos = esc_p.find(sub_q)
                if pos != -1:
                    end_p = min(len(esc_p), pos + max(len(esc_q), 18))
                    esc_p = (
                        esc_p[:pos]
                        + f'<span id="hl-target">{esc_p[pos:end_p]}</span>'
                        + esc_p[end_p:]
                    )
                    marked = True
        out.append(esc_p)

    if not marked and out:
        # Wrap first key clause of the richest paragraph
        idx = min(1, len(out) - 1)
        clauses = re.split(r"([，。；])", out[idx])
        if len(clauses) >= 3:
            clauses[0] = f'<span id="hl-target">{clauses[0]}</span>'
            out[idx] = "".join(clauses)
        else:
            out[idx] = f'<span id="hl-target">{out[idx][:32]}</span>{out[idx][32:]}'
    return out


def _build_news_portal_html(
    heading: str,
    paragraphs: list[str],
    quote: str,
    citation: SourceCitation | None,
    w: int,
    h: int,
) -> str:
    pub = citation.publisher if citation else "台灣消費新聞"
    pub_date = citation.published_date if citation else "2026-11-25"
    color, brand_name = _pick_brand(pub)
    marked_paras = _inject_highlight_span(paragraphs[:4], quote)
    p_html = "\n".join(f"<p>{p}</p>" for p in marked_paras)
    is_vertical = h > w
    card_w = "94%" if is_vertical else "1460px"
    title_sz = "36px" if is_vertical else "40px"
    body_sz = "27px" if is_vertical else "26px"

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      width: {w}px; height: {h}px; background: #000;
      display: flex; align-items: center; justify-content: center;
      font-family: "PingFang TC", "Noto Sans CJK TC", "Heiti TC", sans-serif;
      overflow: hidden;
    }}
    .browser {{
      width: {card_w}; height: 86%; background: #ffffff;
      border-radius: 6px; overflow: hidden; box-shadow: 0 12px 40px rgba(0,0,0,0.65);
    }}
    .topbar {{
      background: {color}; color: #fff; padding: 14px 36px;
      display: flex; align-items: center; justify-content: space-between;
    }}
    .brand {{ font-size: 26px; font-weight: 800; letter-spacing: 1px; }}
    .nav {{ font-size: 17px; opacity: 0.9; }}
    .content {{ padding: 28px 48px; }}
    .crumb {{ color: {color}; font-size: 17px; font-weight: 600; margin-bottom: 12px; }}
    h1 {{ font-size: {title_sz}; line-height: 1.32; color: #141414; font-weight: 800; margin-bottom: 16px; }}
    .meta-row {{
      display: flex; align-items: center; justify-content: space-between;
      padding-bottom: 14px; margin-bottom: 20px; border-bottom: 1px solid #e3e3e3;
      color: #666; font-size: 17px;
    }}
    .shares span {{
      display: inline-block; padding: 4px 12px; border-radius: 14px;
      font-size: 14px; font-weight: 700; color: #fff; margin-left: 6px;
    }}
    .s-fb {{ background: #1877F2; }} .s-ln {{ background: #06C755; }} .s-th {{ background: #111; }}
    p {{ font-size: {body_sz}; line-height: 1.72; color: #222; margin-bottom: 18px; text-align: justify; }}
    </style></head><body>
      <div class="browser">
        <div class="topbar">
          <div class="brand">{html.escape(brand_name)}</div>
          <div class="nav">即時　｜　生活消費　｜　財經政策　｜　國際法規</div>
        </div>
        <div class="content">
          <div class="crumb">首頁 〉 生活消費 〉 民生數據追蹤</div>
          <h1>{html.escape(heading)}</h1>
          <div class="meta-row">
            <div>{html.escape(pub)} 綜合報導　｜　發布時間：{html.escape(pub_date)}</div>
            <div class="shares"><span class="s-fb">f 分享</span><span class="s-ln">LINE</span><span class="s-th">@Threads</span></div>
          </div>
          {p_html}
        </div>
      </div>
    </body></html>"""


def _build_official_gov_doc_html(
    heading: str,
    paragraphs: list[str],
    quote: str,
    citation: SourceCitation | None,
    w: int,
    h: int,
) -> str:
    agency = citation.publisher if (citation and any(k in citation.publisher for k in ("署", "部", "處", "會"))) else "衛生福利部食品藥物管理署"
    pub_date = citation.published_date if citation else "2026-11-25"
    roc_year = max(112, int(pub_date[:4]) - 1911) if pub_date[:4].isdigit() else 115
    marked_paras = _inject_highlight_span(paragraphs[:3], quote)
    numerals = ["一、", "二、", "三、", "四、"]
    items_html = "\n".join(
        f'<div class="clause"><span class="num">{numerals[i]}</span><div class="ctxt">{p}</div></div>'
        for i, p in enumerate(marked_paras)
    )
    is_vertical = h > w
    sheet_w = "94%" if is_vertical else "1280px"
    font_sz = "25px" if is_vertical else "25px"

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      width: {w}px; height: {h}px; background: #000;
      display: flex; align-items: center; justify-content: center;
      font-family: "BiauKai", "Kaiti TC", "Songti TC", "PingFang TC", serif;
      overflow: hidden;
    }}
    .a4 {{
      width: {sheet_w}; height: 88%; background: #fcfcfa;
      padding: 36px 58px; box-shadow: 0 10px 36px rgba(0,0,0,0.7);
      color: #111; position: relative;
    }}
    .doc-header {{ text-align: center; font-size: 36px; font-weight: 700; letter-spacing: 5px; margin-bottom: 18px; }}
    .meta-right {{ text-align: right; font-size: 16px; color: #333; line-height: 1.5; margin-bottom: 10px; font-family: "PingFang TC", sans-serif; }}
    .field {{ font-size: 21px; line-height: 1.65; margin-bottom: 4px; }}
    .subject {{ font-size: 24px; line-height: 1.62; margin: 14px 0; font-weight: 700; }}
    .sec-title {{ font-size: 24px; font-weight: 700; margin: 10px 0 8px 0; }}
    .clause {{ display: flex; font-size: {font_sz}; line-height: 1.72; margin-bottom: 12px; }}
    .num {{ min-width: 52px; font-weight: 700; }}
    .ctxt {{ flex: 1; text-align: justify; }}
    .stamp {{
      position: absolute; right: 64px; bottom: 28px; border: 3px solid rgba(190,20,20,0.55);
      color: rgba(190,20,20,0.65); padding: 6px 14px; font-size: 18px; font-weight: 700;
      transform: rotate(-4deg); letter-spacing: 2px;
    }}
    </style></head><body>
      <div class="a4">
        <div class="doc-header">{html.escape(agency)}　函</div>
        <div class="meta-right">檔　　號：公開法規彙整<br>保存年限：永久</div>
        <div class="field">受文者：各直轄市及縣市政府、相關產業公協會及消費者保護團體</div>
        <div class="field">發文日期：中華民國 {roc_year} 年 {pub_date[5:7]} 月 {pub_date[8:10]} 日　｜　速別：普通件</div>
        <div class="subject">主旨：有關「{html.escape(heading[:38])}」之標示與資訊揭露查核原則，請查照並轉知所屬依規定辦理。</div>
        <div class="sec-title">說明：</div>
        {items_html}
        <div class="stamp">公開法規資料查證聯</div>
      </div>
    </body></html>"""


def _build_gov_portal_html(
    heading: str,
    paragraphs: list[str],
    quote: str,
    citation: SourceCitation | None,
    w: int,
    h: int,
) -> str:
    agency = citation.publisher if citation else "衛生福利部食品藥物管理署"
    marked_paras = _inject_highlight_span(paragraphs[:3], quote)
    p_html = "\n".join(f"<p>{p}</p>" for p in marked_paras)
    is_vertical = h > w
    card_w = "94%" if is_vertical else "1420px"

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      width: {w}px; height: {h}px; background: #000;
      display: flex; align-items: center; justify-content: center;
      font-family: "PingFang TC", "Noto Sans CJK TC", sans-serif; overflow: hidden;
    }}
    .portal {{ width: {card_w}; height: 86%; background: #fff; border-radius: 6px; overflow: hidden; }}
    .govbar {{
      background: #00796B; color: #fff; padding: 18px 36px;
      display: flex; align-items: center; justify-content: space-between;
    }}
    .gtitle {{ font-size: 28px; font-weight: 800; display: flex; align-items: center; gap: 14px; }}
    .badge {{ background: #fff; color: #00796B; border-radius: 50%; width: 40px; height: 40px; display: inline-flex; align-items: center; justify-content: center; font-weight: 900; }}
    .search-icon {{ background: #004D40; padding: 8px 18px; border-radius: 20px; font-size: 16px; }}
    .banner {{ background: #E0F2F1; color: #004D40; padding: 14px 44px; font-size: 20px; font-weight: 700; border-bottom: 3px solid #00796B; }}
    .body {{ padding: 28px 44px; }}
    h2 {{ font-size: 34px; color: #004D40; margin-bottom: 18px; line-height: 1.35; }}
    p {{ font-size: 26px; line-height: 1.75; color: #222; margin-bottom: 18px; }}
    </style></head><body>
      <div class="portal">
        <div class="govbar">
          <div class="gtitle"><span class="badge">政</span> {html.escape(agency)} 公開資訊整合平台</div>
          <div class="search-icon">🔍 站內法規與公告檢索</div>
        </div>
        <div class="banner">公告專區 〉 消費資訊透明化與標示規範說明</div>
        <div class="body">
          <h2>{html.escape(heading)}</h2>
          {p_html}
        </div>
      </div>
    </body></html>"""


def _build_product_comparison_html(
    heading: str,
    left_title: str,
    left_points: list[str],
    right_title: str,
    right_points: list[str],
    delta_badge: str,
    w: int,
    h: int,
    reveal_right: bool = True,
) -> str:
    l_items = "".join(f"<li>{html.escape(x)}</li>" for x in left_points[:4])
    r_items = "".join(f"<li>{html.escape(x)}</li>" for x in right_points[:4])
    right_opacity = "1.0" if reveal_right else "0.08"
    badge_opacity = "1.0" if reveal_right else "0.0"

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      width: {w}px; height: {h}px; background: #0B0E14;
      font-family: "PingFang TC", "Noto Sans CJK TC", sans-serif;
      color: #fff; padding: 48px 64px; display: flex; flex-direction: column; justify-content: center;
    }}
    .top-head {{
      display: flex; align-items: center; justify-content: space-between; margin-bottom: 28px;
    }}
    h1 {{ font-size: 42px; font-weight: 900; color: #FFE600; }}
    .delta {{
      opacity: {badge_opacity}; background: #D90416; color: #fff;
      padding: 10px 26px; border-radius: 8px; font-size: 32px; font-weight: 900;
      box-shadow: 0 6px 20px rgba(217,4,22,0.5);
    }}
    .grid {{ display: flex; gap: 32px; flex: 1; max-height: 72%; }}
    .col {{
      flex: 1; border-radius: 12px; padding: 32px 36px;
      display: flex; flex-direction: column;
    }}
    .col-l {{ background: #162238; border: 2px solid #3B82F6; }}
    .col-r {{ background: #2A1215; border: 3px solid #FF2E2E; opacity: {right_opacity}; }}
    .ctitle {{ font-size: 34px; font-weight: 800; margin-bottom: 22px; padding-bottom: 14px; border-bottom: 2px solid rgba(255,255,255,0.18); }}
    .col-l .ctitle {{ color: #60A5FA; }}
    .col-r .ctitle {{ color: #FF5252; }}
    ul {{ list-style: none; }}
    li {{ font-size: 28px; line-height: 1.65; margin-bottom: 16px; padding-left: 34px; position: relative; }}
    .col-l li::before {{ content: "●"; color: #60A5FA; position: absolute; left: 0; }}
    .col-r li::before {{ content: "▲"; color: #FFE600; position: absolute; left: 0; }}
    </style></head><body>
      <div class="top-head">
        <h1>{html.escape(heading[:28])}</h1>
        <div class="delta">{html.escape(delta_badge)}</div>
      </div>
      <div class="grid">
        <div class="col col-l"><div class="ctitle">{html.escape(left_title)}</div><ul>{l_items}</ul></div>
        <div class="col col-r"><div class="ctitle">{html.escape(right_title)}</div><ul>{r_items}</ul></div>
      </div>
    </body></html>"""


def _build_netizen_comments_html(
    comments: list[dict[str, str]],
    visible_count: int,
    w: int,
    h: int,
) -> str:
    avatar_colors = ["#E53935", "#1E88E5", "#43A047", "#8E24AA", "#FB8C00"]
    rows = []
    for i, c in enumerate(comments[:4]):
        vis = "1.0" if i < visible_count else "0.0"
        handle = html.escape(c.get("user", f"@tw_citizen_{i+1}"))
        ago = html.escape(c.get("time", f"{i+2} 小時前"))
        txt = html.escape(c.get("text", ""))
        likes = html.escape(str(c.get("likes", 180 + i * 97)))
        initial = handle.lstrip("@")[:1].upper() or "網"
        bg = avatar_colors[i % len(avatar_colors)]
        rows.append(
            f"""<div class="cmt" style="opacity:{vis}">
              <div class="av" style="background:{bg}">{initial}</div>
              <div class="cbody">
                <div class="cmeta"><span class="u">{handle}</span><span class="t">{ago}</span></div>
                <div class="ctxt">{txt}</div>
                <div class="cact">👍 {likes}　　👎　　回覆</div>
              </div>
            </div>"""
        )

    is_vertical = h > w
    box_w = "94%" if is_vertical else "1380px"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      width: {w}px; height: {h}px; background: #000;
      display: flex; align-items: center; justify-content: center;
      font-family: "PingFang TC", "Noto Sans CJK TC", sans-serif; overflow: hidden;
    }}
    .feed {{ width: {box_w}; height: 86%; background: #ffffff; border-radius: 8px; padding: 28px 44px; }}
    .hdr {{ font-size: 24px; font-weight: 800; color: #111; padding-bottom: 14px; margin-bottom: 18px; border-bottom: 2px solid #eee; }}
    .cmt {{ display: flex; gap: 20px; margin-bottom: 22px; padding-bottom: 16px; border-bottom: 1px solid #f2f2f2; }}
    .av {{ width: 52px; height: 52px; border-radius: 50%; color: #fff; font-size: 24px; font-weight: 800; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }}
    .cmeta {{ margin-bottom: 6px; }}
    .u {{ font-size: 19px; font-weight: 700; color: #111; margin-right: 12px; }}
    .t {{ font-size: 16px; color: #777; }}
    .ctxt {{ font-size: 26px; color: #181818; line-height: 1.48; font-weight: 500; margin-bottom: 8px; }}
    .cact {{ font-size: 16px; color: #666; font-weight: 600; }}
    </style></head><body>
      <div class="feed">
        <div class="hdr">熱門社群留言討論串　｜　依熱度排序</div>
        {''.join(rows)}
      </div>
    </body></html>"""


def _crop_and_draw_red_underline(
    full_img: Image.Image,
    rects: list[dict[str, float]],
    zoom: float = 1.42,
    underline_progress: float = 1.0,
) -> Image.Image:
    """Perform @unicorn-roar signature punch-in zoom onto `#hl-target` + thick crimson underline."""
    w, h = full_img.size
    canvas = full_img.copy()
    draw = ImageDraw.Draw(canvas)

    if rects:
        for r in rects:
            rx, ry, rw, rh = r["x"], r["y"], r["width"], r["height"]
            end_x = rx + rw * max(0.0, min(1.0, underline_progress))
            uy = ry + rh + 3
            draw.line([(rx, uy), (end_x, uy)], fill=(216, 6, 20), width=6)

        cx = sum(r["x"] + r["width"] / 2.0 for r in rects) / len(rects)
        cy = sum(r["y"] + r["height"] / 2.0 for r in rects) / len(rects)
    else:
        cx, cy = w * 0.5, h * 0.48

    crop_w = w / zoom
    crop_h = h / zoom
    x0 = max(0.0, min(w - crop_w, cx - crop_w / 2.0))
    y0 = max(0.0, min(h - crop_h, cy - crop_h / 2.0))
    cropped = canvas.crop((int(x0), int(y0), int(x0 + crop_w), int(y0 + crop_h)))
    return cropped.resize((w, h), Image.Resampling.LANCZOS)


class ChromeEvidenceRenderer:
    """Persistent Headless Chrome 158 session for blazingly fast multi-beat card rendering."""

    def __init__(self) -> None:
        self._pw = None
        self._browser: Browser | None = None

    def __enter__(self) -> "ChromeEvidenceRenderer":
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            executable_path=CHROME_EXEC,
            headless=True,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def render_two_beat_evidence(
        self,
        template_kind: str,
        heading: str,
        paragraphs: list[str],
        quote: str,
        citation: SourceCitation | None,
        width: int,
        height: int,
        out_prefix: Path,
        include_watermark: bool = True,
    ) -> list[Path]:
        """Render 3 progressive visual beats: Overview -> Zoom-in 50% underline -> Zoom-in 100% underline."""
        assert self._browser is not None
        page = self._browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1.0)
        try:
            if template_kind == "official_gov_doc":
                html_doc = _build_official_gov_doc_html(heading, paragraphs, quote, citation, width, height)
            elif template_kind == "gov_web_portal":
                html_doc = _build_gov_portal_html(heading, paragraphs, quote, citation, width, height)
            else:
                html_doc = _build_news_portal_html(heading, paragraphs, quote, citation, width, height)

            page.set_content(html_doc, wait_until="domcontentloaded")
            raw_png = out_prefix.parent / f"{out_prefix.name}_raw.png"
            raw_png.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(raw_png))

            rects: list[dict[str, float]] = page.evaluate(
                """() => {
                    const el = document.getElementById('hl-target');
                    if (!el) return [];
                    return Array.from(el.getClientRects()).map(r => ({
                        x: r.x, y: r.y, width: r.width, height: r.height
                    }));
                }"""
            )
        finally:
            page.close()

        base_im = Image.open(raw_png).convert("RGB")
        raw_png.unlink(missing_ok=True)

        beats: list[Path] = []
        # Beat 1: Full page overview (`1.0x`)
        b1 = base_im.copy()
        if include_watermark and citation:
            draw_citation_watermark(ImageDraw.Draw(b1), citation, x=14, y=10, font_size=max(20, int(width * 0.0125)))
        p1 = out_prefix.parent / f"{out_prefix.name}_beat1.png"
        b1.save(p1)
        beats.append(p1)

        # Beat 2: 1.36x Punch-In Zoom + 55% Red Underline Wipe
        b2 = _crop_and_draw_red_underline(base_im, rects, zoom=1.36, underline_progress=0.55)
        if include_watermark and citation:
            draw_citation_watermark(ImageDraw.Draw(b2), citation, x=14, y=10, font_size=max(20, int(width * 0.0125)))
        p2 = out_prefix.parent / f"{out_prefix.name}_beat2.png"
        b2.save(p2)
        beats.append(p2)

        # Beat 3: 1.44x Closer Zoom + 100% Full Red Underline
        b3 = _crop_and_draw_red_underline(base_im, rects, zoom=1.44, underline_progress=1.0)
        if include_watermark and citation:
            draw_citation_watermark(ImageDraw.Draw(b3), citation, x=14, y=10, font_size=max(20, int(width * 0.0125)))
        p3 = out_prefix.parent / f"{out_prefix.name}_beat3.png"
        b3.save(p3)
        beats.append(p3)

        return beats

    def render_comparison_beats(
        self,
        heading: str,
        left_title: str,
        left_points: list[str],
        right_title: str,
        right_points: list[str],
        delta_badge: str,
        citation: SourceCitation | None,
        width: int,
        height: int,
        out_prefix: Path,
    ) -> list[Path]:
        assert self._browser is not None
        page = self._browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1.0)
        beats: list[Path] = []
        try:
            for b_idx, rev in enumerate([False, True], 1):
                html_doc = _build_product_comparison_html(
                    heading, left_title, left_points, right_title, right_points, delta_badge, width, height, reveal_right=rev
                )
                page.set_content(html_doc, wait_until="domcontentloaded")
                bp = out_prefix.parent / f"{out_prefix.name}_cmp{b_idx}.png"
                bp.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(bp))
                if citation:
                    im = Image.open(bp).convert("RGB")
                    draw_citation_watermark(ImageDraw.Draw(im), citation, x=14, y=10, font_size=max(20, int(width * 0.0125)))
                    im.save(bp)
                beats.append(bp)
        finally:
            page.close()
        return beats

    def render_comment_wall_beats(
        self,
        comments: list[dict[str, str]],
        width: int,
        height: int,
        out_prefix: Path,
    ) -> list[Path]:
        assert self._browser is not None
        page = self._browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1.0)
        beats: list[Path] = []
        try:
            for b_idx, vis_n in enumerate([1, 2, 4], 1):
                html_doc = _build_netizen_comments_html(comments, vis_n, width, height)
                page.set_content(html_doc, wait_until="domcontentloaded")
                bp = out_prefix.parent / f"{out_prefix.name}_cmt{b_idx}.png"
                bp.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(bp))
                beats.append(bp)
        finally:
            page.close()
        return beats
