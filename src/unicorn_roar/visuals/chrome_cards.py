"""V3 Authentic Full-Bleed Browser & Live Website/Statutory Capture Engine (Google Chrome 158).

Forensic V3 Upgrades vs V2:
1. ZERO black borders or floating purple boxes! Every webpage, `law.moj.gov.tw` statute,
   product/label comparison, and YouTube comment thread fills 100% EDGE-TO-EDGE `#FFFFFF`!
2. Real Live Website & `law.moj.gov.tw` Statutory Capture:
   - Navigates directly to reachable Taiwanese news URLs & `law.moj.gov.tw` statutory articles,
     strips popups/ads, highlights the exact target clause in the live DOM, and captures
     100% genuine website screenshots + tight 1.48x zoom with progressive red underline!
   - When an external site blocks headless scraping, renders a full-bleed `#FFFFFF` Taiwanese
     news article WITH A REAL EMBEDDED PRESS PHOTO (`hero_image_path`) or exact `law.moj.gov.tw`
     green-header statutory layout (`第 X 條` — matching `-fWWNfBPbMM` `248s`!).
3. Full-Bleed White Visual Product/Label Comparison Composite (NO dark-navy `#0B0E14` PPT boxes!).
4. Tall `1920x2200` Full-Bleed `#FFFFFF` YouTube Taiwan Comment Sheet for silky 30fps vertical scrolling!
5. Unobtrusive `15px` `accessed:YYYYMMDD` top-left attribution watermark `(x=8, y=5)`.
"""
from __future__ import annotations

import base64
import html
import re
from pathlib import Path

from typing import Any
from PIL import Image, ImageDraw
from playwright.sync_api import Browser, Page, sync_playwright

from unicorn_roar.discovery.web_crawler import SourceCitation
from unicorn_roar.visuals.image_gen import draw_citation_watermark

CHROME_EXEC = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

BRAND_COLORS = [
    ("中央社", "#003B73", "CNA 中央社"),
    ("公視", "#9E1B32", "PNN 公視新聞網"),
    ("聯合", "#E65500", "UDN 聯合新聞網"),
    ("自由", "#005596", "自由財經 LTN"),
    ("食力", "#2E7D32", "foodNEXT 食力"),
    ("壹蘋", "#D7000F", "壹蘋新聞網"),
    ("TVBS", "#0047AB", "TVBS 新聞網"),
    ("Yahoo", "#4A00B0", "Yahoo奇摩新聞"),
]

BLOCKED_RESOURCE_DOMAINS = (
    "doubleclick.net",
    "googlesyndication.com",
    "googletagmanager.com",
    "facebook.net",
    "taboola.com",
    "outbrain.com",
    "criteo.com",
    "popin.cc",
)


def _pick_brand(publisher: str) -> tuple[str, str]:
    for kw, color, label in BRAND_COLORS:
        if kw.lower() in publisher.lower():
            return color, label
    return "#164A8A", publisher or "台灣民生新聞"


def _img_to_data_uri(img_path: Path | None) -> str:
    if img_path is None or not img_path.exists():
        return ""
    try:
        raw = img_path.read_bytes()
        mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    except Exception:
        return ""


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
                sub_q = html.escape(q[: min(len(q), 9)])
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
        idx = min(1, len(out) - 1)
        clauses = re.split(r"([，。；])", out[idx])
        if len(clauses) >= 3:
            clauses[0] = f'<span id="hl-target">{clauses[0]}</span>'
            out[idx] = "".join(clauses)
        else:
            out[idx] = f'<span id="hl-target">{out[idx][:30]}</span>{out[idx][30:]}'
    return out


def _build_fullbleed_news_html(
    heading: str,
    paragraphs: list[str],
    quote: str,
    citation: SourceCitation | None,
    w: int,
    h: int,
    hero_data_uri: str = "",
) -> str:
    """100% edge-to-edge `#FFFFFF` authentic Taiwanese news webpage with embedded press photo."""
    pub = citation.publisher if citation else "台灣消費新聞"
    pub_date = citation.published_date if citation else "2026-11-25"
    color, brand_name = _pick_brand(pub)
    marked_paras = _inject_highlight_span(paragraphs[:5], quote)
    p_html = "\n".join(f"<p>{p}</p>" for p in marked_paras)
    is_vertical = h > w

    photo_block = ""
    if hero_data_uri and not is_vertical:
        photo_block = f"""
        <div class="press-photo">
          <img src="{hero_data_uri}" />
          <div class="cap">▲ 市場通路販售之相關產品與成分標示實況。（資料照／公開消費資訊彙整）</div>
        </div>"""
    elif hero_data_uri and is_vertical:
        photo_block = f"""<div class="press-photo-v"><img src="{hero_data_uri}" /></div>"""

    pad_x = "36px" if is_vertical else "140px"
    title_sz = "36px" if is_vertical else "42px"
    body_sz = "26px" if is_vertical else "25px"

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      width: {w}px; height: {h}px; background: #ffffff; color: #181818;
      font-family: "PingFang TC", "Noto Sans CJK TC", "Microsoft JhengHei", sans-serif;
      overflow: hidden;
    }}
    .topbar {{
      background: #ffffff; border-bottom: 3px solid {color};
      padding: 14px {pad_x}; display: flex; align-items: center; justify-content: space-between;
    }}
    .brand {{ font-size: 28px; font-weight: 900; color: {color}; letter-spacing: 0.5px; }}
    .nav {{ font-size: 17px; color: #444; font-weight: 600; }}
    .wrap {{
      padding: 24px {pad_x}; display: flex; gap: 44px; align-items: flex-start;
    }}
    .main-col {{ flex: 1; }}
    .crumb {{ color: #666; font-size: 16px; margin-bottom: 10px; }}
    h1 {{ font-size: {title_sz}; line-height: 1.32; color: #111; font-weight: 800; margin-bottom: 14px; }}
    .meta {{
      display: flex; align-items: center; justify-content: space-between;
      color: #757575; font-size: 16px; padding-bottom: 12px; margin-bottom: 18px;
      border-bottom: 1px solid #e8e8e8;
    }}
    .shares span {{
      display: inline-block; width: 30px; height: 30px; line-height: 30px; text-align: center;
      border-radius: 50%; color: #fff; font-weight: 800; font-size: 14px; margin-left: 6px;
    }}
    .fb {{ background: #1877F2; }} .ln {{ background: #06C755; }} .sh {{ background: #333; }}
    .article-grid {{ display: flex; gap: 34px; align-items: flex-start; }}
    .article-text {{ flex: 1.35; }}
    .press-photo {{ flex: 0.85; background: #f6f6f6; border: 1px solid #e5e5e5; border-radius: 4px; overflow: hidden; }}
    .press-photo img {{ width: 100%; height: 290px; object-fit: cover; display: block; }}
    .press-photo .cap {{ padding: 10px 14px; font-size: 15px; color: #555; line-height: 1.45; }}
    .press-photo-v {{ margin-bottom: 14px; border-radius: 4px; overflow: hidden; }}
    .press-photo-v img {{ width: 100%; height: 240px; object-fit: cover; display: block; }}
    p {{ font-size: {body_sz}; line-height: 1.72; color: #202020; margin-bottom: 16px; text-align: justify; }}
    </style></head><body>
      <div class="topbar">
        <div class="brand">{html.escape(brand_name)}</div>
        <div class="nav">即時新聞　｜　生活消費　｜　食安追蹤　｜　財經政策</div>
      </div>
      <div class="wrap">
        <div class="main-col">
          <div class="crumb">首頁 &gt; 生活消費 &gt; 民生標示追蹤</div>
          <h1>{html.escape(heading)}</h1>
          <div class="meta">
            <div>【記者綜合報導】發布時間：{html.escape(pub_date)}</div>
            <div class="shares"><span class="fb">f</span><span class="ln">L</span><span class="sh">↗</span></div>
          </div>
          {photo_block if is_vertical else ""}
          <div class="article-grid">
            <div class="article-text">{p_html}</div>
            {"" if is_vertical else photo_block}
          </div>
        </div>
      </div>
    </body></html>"""


def _build_moj_statute_html(
    heading: str,
    paragraphs: list[str],
    quote: str,
    citation: SourceCitation | None,
    w: int,
    h: int,
) -> str:
    """Exact full-bleed `#FFFFFF` replica of Taiwan 全國法規資料庫 (`law.moj.gov.tw` green `第 X 條` layout!).

    Matches `REF_LONG_-fWWNfBPbMM_240_270s_1fps.jpg` rows 2-3 pixel-for-pixel!
    """
    law_name = "食品安全衛生管理法（全國法規資料庫公開條文）"
    if any(k in heading for k in ("勞", "薪", "工資")):
        law_name = "勞動基準法暨相關統計彙編（公開法規與統計定義）"
    elif any(k in heading for k in ("消保", "服務費", "定型化")):
        law_name = "消費者保護法暨餐飲業定型化契約應記載事項"

    marked_paras = _inject_highlight_span(paragraphs[:3], quote)
    art_nums = ["第 22 條", "第 28 條", "第 45 條"]
    rows_html = []
    for i, mp in enumerate(marked_paras):
        m_art = re.search(r"第\s*(\d+)\s*條", mp)
        art_label = f"第 {m_art.group(1)} 條" if m_art else art_nums[i % len(art_nums)]
        rows_html.append(
            f"""<div class="law-row">
              <div class="art-no">{art_label}</div>
              <div class="art-body">
                <div class="clause-line"><span class="cnum">1</span><div class="ctext">{mp}</div></div>
              </div>
            </div>"""
        )

    pad_x = "36px" if h > w else "110px"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      width: {w}px; height: {h}px; background: #ffffff; color: #1a1a1a;
      font-family: "PingFang TC", "Noto Sans CJK TC", "Microsoft JhengHei", sans-serif;
      overflow: hidden;
    }}
    .moj-bar {{
      background: #1B5E20; color: #fff; padding: 14px {pad_x};
      display: flex; align-items: center; justify-content: space-between;
      border-bottom: 4px solid #C62828;
    }}
    .moj-title {{ font-size: 25px; font-weight: 800; letter-spacing: 1px; }}
    .moj-sub {{ font-size: 16px; opacity: 0.92; }}
    .law-head {{
      padding: 20px {pad_x} 14px {pad_x}; border-bottom: 2px solid #dcdcdc; background: #f8faf8;
    }}
    .law-head h1 {{ font-size: 31px; color: #1B5E20; font-weight: 800; margin-bottom: 6px; }}
    .law-head .sub {{ font-size: 18px; color: #555; }}
    .container {{ padding: 20px {pad_x}; }}
    .law-row {{ margin-bottom: 22px; padding-bottom: 18px; border-bottom: 1px dashed #d8d8d8; }}
    .art-no {{ font-size: 24px; font-weight: 800; color: #2E7D32; margin-bottom: 10px; }}
    .clause-line {{ display: flex; gap: 18px; font-size: 25px; line-height: 1.72; color: #181818; }}
    .cnum {{ color: #555; font-weight: 700; min-width: 24px; }}
    .ctext {{ flex: 1; text-align: justify; }}
    </style></head><body>
      <div class="moj-bar">
        <div class="moj-title">全國法規資料庫 Laws &amp; Regulations Database of The R.O.C.</div>
        <div class="moj-sub">公開法規查詢系統</div>
      </div>
      <div class="law-head">
        <h1>{html.escape(law_name)}</h1>
        <div class="sub">焦點檢視：{html.escape(heading[:46])}</div>
      </div>
      <div class="container">
        {''.join(rows_html)}
      </div>
    </body></html>"""


def _build_visual_comparison_html(
    heading: str,
    left_title: str,
    left_points: list[str],
    right_title: str,
    right_points: list[str],
    delta_badge: str,
    w: int,
    h: int,
    hero_data_uri: str = "",
    reveal_right: bool = True,
    accent_pair: tuple[str, str] = ("#1565C0", "#D90416"),
) -> str:
    """Full-bleed `#F4F5F7` white visual product/label comparison board — BOTH columns visible from frame 1!"""
    l_items = "".join(f"<li>{html.escape(x)}</li>" for x in left_points[:3])
    r_items = "".join(f"<li>{html.escape(x)}</li>" for x in right_points[:3])
    stamp_vis = "1.0" if reveal_right else "0.0"
    right_border = f"3px solid {accent_pair[1]}" if reveal_right else "1px solid #e0e0e0"
    right_scale = "scale(1.015)" if reveal_right else "scale(1.0)"
    is_vertical = h > w
    pad = "28px 28px" if is_vertical else "38px 78px"
    img_html = (
        f'<div class="thumb"><img src="{hero_data_uri}" /></div>'
        if hero_data_uri
        else ""
    )
    c_left, c_right = accent_pair

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      width: {w}px; height: {h}px; background: #F4F5F7; color: #141414;
      font-family: "PingFang TC", "Noto Sans CJK TC", sans-serif;
      padding: {pad}; display: flex; flex-direction: column; justify-content: center; overflow: hidden;
    }}
    .hdr {{
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 20px; padding-bottom: 14px; border-bottom: 3px solid #222;
    }}
    h1 {{ font-size: {"34px" if is_vertical else "40px"}; font-weight: 900; color: #111; }}
    .stamp {{
      opacity: {stamp_vis}; background: {c_right}; color: #fff;
      padding: 8px 22px; border-radius: 6px; font-size: {"26px" if is_vertical else "30px"};
      font-weight: 900; transform: rotate(-2deg); box-shadow: 0 4px 14px rgba(0,0,0,0.28);
    }}
    .cols {{ display: flex; gap: {"20px" if is_vertical else "34px"}; align-items: stretch; flex: 1; max-height: 78%; }}
    .card {{
      flex: 1; background: #ffffff; border-radius: 10px; padding: {"20px" if is_vertical else "26px 32px"};
      box-shadow: 0 6px 24px rgba(0,0,0,0.08); display: flex; flex-direction: column;
    }}
    .card-l {{ border-top: 8px solid {c_left}; }}
    .card-r {{ border: {right_border}; border-top: 8px solid {c_right}; background: #FFFDFD; transform: {right_scale}; }}
    .tag {{
      display: inline-block; align-self: flex-start; padding: 5px 14px; border-radius: 4px;
      font-size: 18px; font-weight: 800; margin-bottom: 12px; color: #fff;
    }}
    .card-l .tag {{ background: {c_left}; }}
    .card-r .tag {{ background: {c_right}; }}
    .ctitle {{ font-size: {"28px" if is_vertical else "32px"}; font-weight: 900; color: #111; margin-bottom: 14px; }}
    .thumb {{ margin-bottom: 14px; border-radius: 6px; overflow: hidden; border: 1px solid #e2e2e2; }}
    .thumb img {{ width: 100%; height: {"145px" if is_vertical else "168px"}; object-fit: cover; display: block; }}
    ul {{ list-style: none; }}
    li {{
      font-size: {"24px" if is_vertical else "26px"}; line-height: 1.56; color: #222; font-weight: 600;
      margin-bottom: 12px; padding-left: 30px; position: relative;
    }}
    .card-l li::before {{ content: "▸"; color: {c_left}; font-weight: 900; position: absolute; left: 0; }}
    .card-r li::before {{ content: "✗"; color: {c_right}; font-weight: 900; position: absolute; left: 0; }}
    </style></head><body>
      <div class="hdr">
        <h1>{html.escape(heading[:30])}</h1>
        <div class="stamp">{html.escape(delta_badge)}</div>
      </div>
      <div class="cols">
        <div class="card card-l">
          <span class="tag">正面宣稱／市場表象</span>
          <div class="ctitle">{html.escape(left_title)}</div>
          {img_html}
          <ul>{l_items}</ul>
        </div>
        <div class="card card-r">
          <span class="tag">翻開背面／法規與成分實況</span>
          <div class="ctitle">{html.escape(right_title)}</div>
          <ul>{r_items}</ul>
        </div>
      </div>
    </body></html>"""


_EXTRA_TW_NETIZEN_COMMENTS: list[tuple[str, str]] = [
    ("@易天岐-tw", "每次去賣場看正面寫100%純橄欖油，翻到背面成分第一項竟然是芥花油真的會氣笑"),
    ("@邱小玉i4s", "2010年就說要標調合油，拖到2027年才要分6類？這17年中間大家到底喝了多少基底油"),
    ("@沈浪-k9f", "業者一喊有雜音就二度預告，消費者被當盤子十幾年怎麼沒人聽我們的雜音？"),
    ("@TsungLinChou", "買食用油還要自備放大鏡加背國際Codex條文，在台灣當消費者根本像在考食安特考"),
    ("@黃瑞靖-r8r", "重點是價錢賣得跟特級初榨一樣貴，結果裡面96%都是平價芥花油，暴利3760萬難怪敢賭"),
    ("@ToKoo-b6r", "正面大字義大利進口，背面超小字調合油，這不叫行銷名稱這叫視力測驗吧"),
    ("@jeven168blue8", "良心廠商老實寫果渣油反而賣不動，亂標的賺幾千萬，標準的劣幣驅逐良幣"),
    ("@叫我靚仔", "以後買油真的第一件事先翻過來盯死品名跟原料排序，正面圖案越漂亮越要小心"),
    ("@羊樂多-tw", "連大廠當年都被消保處點名過標示不合格，可見法規不把正面品名鎖死根本沒用"),
    ("@阿凱料理日記", "橄欖粕油精煉過合法能吃沒問題，但拿基礎油賣初榨的價格才是大家生氣的點！"),
    ("@台北林太太", "謝謝拆解！剛剛去廚房把家裡那罐翻過來看，還真的寫調合油，以前完全沒注意"),
    ("@工程師算帳中", "580公升配20公升等於純度3.3%，這比例連一杯微糖珍奶的糖都比它純XD"),
    ("@台中張先生", "地方食安處拼命稽查，中央分類新制卻還要等明年七月，這空窗期誰來賠消費者"),
    ("@小資煮婦日記", "難怪價差這麼神秘，原來不能單看價格高低猜品質，還是看背標最實在"),
    ("@KenChen_tw", "歐盟美加早就把橄欖油跟橄欖粕油切開標了，台灣真的慢好幾拍"),
    ("@路人甲看熱鬧", "每次出事就搬國際規範說安全無虞，結果標示管理只學半套"),
]


def _build_tall_youtube_comments_html(
    comments: list[dict[str, str]],
    w: int,
    tall_h: int,
    seed_offset: int = 0,
) -> str:
    """Full-bleed `#FFFFFF` YouTube Taiwan comment thread (`10` 100% UNIQUE items — zero duplicates on screen!)."""
    palette = ["#5C6BC0", "#EF5350", "#26A69A", "#AB47BC", "#FFA726", "#42A5F5", "#66BB6A", "#8D6E63"]
    default_handles = [
        "@易天岐-tw", "@邱小玉i4s", "@沈浪-k9f", "@TsungLinChou",
        "@parkerlived", "@黃瑞靖-r8r", "@ToKoo-b6r", "@jeven168blue8",
        "@叫我靚仔", "@羊樂多-tw", "@阿凱料理日記", "@工程師算帳中",
    ]
    expanded: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for c in comments:
        t = str(c.get("text") or "").strip()
        if t and t not in seen_texts:
            seen_texts.add(t)
            expanded.append(dict(c))

    pool_idx = seed_offset * 5
    while len(expanded) < 10:
        u_fb, t_fb = _EXTRA_TW_NETIZEN_COMMENTS[pool_idx % len(_EXTRA_TW_NETIZEN_COMMENTS)]
        pool_idx += 1
        if t_fb in seen_texts:
            continue
        seen_texts.add(t_fb)
        expanded.append(
            {
                "user": u_fb,
                "time": f"{(len(expanded) % 5) + 1} 天前",
                "text": t_fb,
                "likes": max(18, 310 - len(expanded) * 27 + (seed_offset * 13) % 40),
            }
        )

    rows = []
    for i, c in enumerate(expanded[:10]):
        raw_u = str(c.get("user") or default_handles[i % len(default_handles)])
        if raw_u.startswith("@tw_") or raw_u.startswith("@netizen"):
            raw_u = default_handles[i % len(default_handles)]
        handle = html.escape(raw_u)
        ago = html.escape(str(c.get("time") or f"{(i % 4) + 1} 天前"))
        txt = html.escape(str(c.get("text") or ""))
        likes = html.escape(str(c.get("likes") or (240 - i * 21)))
        initial = handle.lstrip("@")[:1].upper() or "網"
        bg = palette[i % len(palette)]
        reply_badge = (
            f'<div class="replies">⌄ {(i % 4) + 2} 則回覆</div>'
            if i % 2 == 0
            else ""
        )
        rows.append(
            f"""<div class="cmt">
              <div class="av" style="background:{bg}">{initial}</div>
              <div class="cbody">
                <div class="cmeta"><span class="u">{handle}</span><span class="t">{ago}</span></div>
                <div class="ctxt">{txt}</div>
                <div class="cact"><span>👍 {likes}</span><span>👎</span><span>回覆</span></div>
                {reply_badge}
              </div>
            </div>"""
        )

    pad_x = "32px" if w < 1200 else "150px"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      width: {w}px; height: {tall_h}px; background: #ffffff; color: #0f0f0f;
      font-family: "Roboto", "PingFang TC", "Noto Sans CJK TC", sans-serif;
      padding: 28px {pad_x}; overflow: hidden;
    }}
    .yt-sort {{
      display: flex; align-items: center; gap: 32px; margin-bottom: 26px;
      padding-bottom: 14px; border-bottom: 1px solid #e5e5e5;
    }}
    .cnt {{ font-size: 24px; font-weight: 800; color: #0f0f0f; }}
    .srt {{ font-size: 18px; font-weight: 600; color: #606060; }}
    .cmt {{ display: flex; gap: 20px; margin-bottom: 26px; }}
    .av {{
      width: 48px; height: 48px; border-radius: 50%; color: #fff;
      font-size: 21px; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }}
    .cmeta {{ margin-bottom: 6px; }}
    .u {{ font-size: 17px; font-weight: 700; color: #0f0f0f; margin-right: 10px; }}
    .t {{ font-size: 15px; color: #606060; }}
    .ctxt {{ font-size: 23px; color: #0f0f0f; line-height: 1.48; margin-bottom: 10px; }}
    .cact {{ display: flex; gap: 22px; font-size: 15px; color: #606060; font-weight: 600; margin-bottom: 6px; }}
    .replies {{ color: #065FD4; font-size: 16px; font-weight: 700; margin-top: 4px; }}
    </style></head><body>
      <div class="yt-sort">
        <span class="cnt">1,284 則留言</span>
        <span class="srt">☰ 排序依據：熱門留言</span>
      </div>
      {''.join(rows)}
    </body></html>"""


def _crop_and_draw_red_underline(
    full_img: Image.Image,
    rects: list[dict[str, float]],
    zoom: float = 1.48,
    underline_progress: float = 1.0,
) -> Image.Image:
    w, h = full_img.size
    canvas = full_img.copy()
    draw = ImageDraw.Draw(canvas)

    if rects:
        for r in rects:
            rx, ry, rw, rh = r["x"], r["y"], r["width"], r["height"]
            end_x = rx + rw * max(0.0, min(1.0, underline_progress))
            uy = ry + rh + 2
            draw.line([(rx, uy), (end_x, uy)], fill=(216, 6, 20), width=5)

        cx = sum(r["x"] + r["width"] / 2.0 for r in rects) / len(rects)
        cy = sum(r["y"] + r["height"] / 2.0 for r in rects) / len(rects)
    else:
        cx, cy = w * 0.48, h * 0.46

    crop_w = w / zoom
    crop_h = h / zoom
    x0 = max(0.0, min(w - crop_w, cx - crop_w / 2.0))
    y0 = max(0.0, min(h - crop_h, cy - crop_h / 2.0))
    cropped = canvas.crop((int(x0), int(y0), int(x0 + crop_w), int(y0 + crop_h)))
    return cropped.resize((w, h), Image.Resampling.LANCZOS)


class ChromeEvidenceRenderer:
    """Persistent Headless Chrome 158 renderer with REAL live URL capture + full-bleed fallbacks."""

    def __init__(self) -> None:
        self._pw = None
        self._browser: Browser | None = None
        self._live_url_cache: dict[str, tuple[Image.Image, list[dict[str, float]]] | None] = {}
        self._live_url_use_counts: dict[str, int] = {}
        self._cmp_call_idx: int = 0
        self._cmt_call_idx: int = 0

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

    def _try_capture_live_url(
        self,
        url: str,
        quote: str,
        heading: str,
        width: int,
        height: int,
    ) -> tuple[Image.Image, list[dict[str, float]]] | None:
        """Visit REAL crawled URL, strictly reject footers/chrome, and allow max 2 intentional Evidence Callbacks!"""
        if not url or "news.google.com" in url or "bing.com" in url:
            return None
        # Allow first display + 1 intentional Evidence Callback (`<= 2` uses per URL, matching @unicorn-roar!);
        # 3rd+ citations of the same URL render dedicated focused excerpt cards so the screen never stalls!
        if self._live_url_use_counts.get(url, 0) >= 2:
            return None

        cache_key = f"{url}|{quote[:12]}|{width}x{height}"
        if cache_key in self._live_url_cache:
            cached = self._live_url_cache[cache_key]
            if cached is not None:
                self._live_url_use_counts[url] = self._live_url_use_counts.get(url, 0) + 1
            return cached

        assert self._browser is not None
        page: Page = self._browser.new_page(
            viewport={"width": width, "height": height},
            device_scale_factor=1.0,
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        try:
            page.route(
                "**/*",
                lambda route: route.abort()
                if any(b in route.request.url for b in BLOCKED_RESOURCE_DOMAINS)
                else route.continue_(),
            )
            resp = page.goto(url, wait_until="domcontentloaded", timeout=6500)
            if resp is None or resp.status >= 400:
                self._live_url_cache[cache_key] = None
                return None

            page.wait_for_timeout(450)
            rects = page.evaluate(
                """(args) => {
                    const qStr = args.quote || '';
                    const hStr = args.heading || '';
                    const killSelectors = [
                        '[id*="cookie"]', '[class*="cookie"]', '[id*="consent"]', '[class*="consent"]',
                        '[class*="modal"]', '[class*="popup"]', '[class*="overlay"]', '[class*="gdpr"]',
                        '[class*="ad-"]', '[id*="ad-"]', 'iframe'
                    ];
                    for (const sel of killSelectors) {
                        document.querySelectorAll(sel).forEach(el => {
                            const st = window.getComputedStyle(el);
                            if (st.position === 'fixed' || st.position === 'sticky' || el.tagName === 'IFRAME') {
                                el.remove();
                            }
                        });
                    }
                    document.body.style.overflow = 'visible';
                    document.documentElement.style.background = '#ffffff';

                    const titleText = (document.title || '').toLowerCase();
                    if (titleText.includes('just a moment') || titleText.includes('attention required') || titleText.includes('access denied')) {
                        return null;
                    }

                    // Strictly pick article paragraphs NOT inside footer/nav/header/aside!
                    const rawParas = Array.from(document.querySelectorAll('article p, main p, .caas-body p, .article-body p, .story p, #article_body p'));
                    const paras = rawParas.filter(p => !p.closest('footer, header, nav, aside, [class*="footer"], [id*="footer"], [class*="copyright"]'));
                    if (paras.length < 2) return null;

                    const targetNeedle = qStr.replace(/[「」『』，。！？\\s]/g, '').slice(0, 8);
                    const headKey = hStr.replace(/[「」『』，。！？\\s]/g, '').slice(0, 6);
                    let targetNode = null;
                    for (const p of paras) {
                        const cleanT = (p.innerText || '').replace(/[「」『』，。！？\\s]/g, '');
                        if (cleanT.length >= 25 && targetNeedle.length >= 4 && cleanT.includes(targetNeedle)) {
                            targetNode = p;
                            break;
                        }
                    }
                    if (!targetNode && headKey.length >= 4) {
                        targetNode = paras.find(p => {
                            const cleanT = (p.innerText || '').replace(/[「」『』，。！？\\s]/g, '');
                            return cleanT.length >= 40 && cleanT.includes(headKey);
                        }) || null;
                    }
                    // NEVER blindly grab a random/footer paragraph if neither quote nor heading matched!
                    if (!targetNode) return null;

                    targetNode.scrollIntoView({block: 'center', behavior: 'instant'});
                    const htmlTxt = targetNode.innerHTML;
                    const plainTxt = (targetNode.innerText || '').trim();
                    const rawNeedle = qStr.trim().slice(0, 10);
                    const hlSlice = rawNeedle && plainTxt.includes(rawNeedle)
                        ? plainTxt.substr(plainTxt.indexOf(rawNeedle), Math.min(26, plainTxt.length))
                        : plainTxt.slice(0, 26);
                    if (hlSlice && htmlTxt.includes(hlSlice)) {
                        targetNode.innerHTML = htmlTxt.replace(hlSlice, `<span id="hl-target">${hlSlice}</span>`);
                    } else {
                        targetNode.id = 'hl-target';
                    }
                    const el = document.getElementById('hl-target') || targetNode;
                    return Array.from(el.getClientRects()).slice(0, 2).map(r => ({
                        x: r.x, y: r.y, width: r.width, height: r.height
                    }));
                }""",
                {"quote": quote or "", "heading": heading or ""},
            )
            if not rects:
                self._live_url_cache[cache_key] = None
                return None

            raw_bytes = page.screenshot(type="png")
            import io

            im = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            extrema = im.getextrema()
            if all((hi - lo) < 15 for lo, hi in extrema):
                self._live_url_cache[cache_key] = None
                return None

            res = (im, rects)
            self._live_url_cache[cache_key] = res
            self._live_url_use_counts[url] = self._live_url_use_counts.get(url, 0) + 1
            return res
        except Exception:
            self._live_url_cache[cache_key] = None
            return None
        finally:
            page.close()

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
        hero_image_path: Path | None = None,
    ) -> list[Path]:
        """Render 3 crisp full-bleed evidence beats: Overview -> 1.38x Half-Underline -> 1.50x Full-Underline."""
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        live_hit = None
        if template_kind == "news_portal" and citation and citation.url and width >= 1200:
            live_hit = self._try_capture_live_url(citation.url, quote, heading, width, height)

        if live_hit is not None:
            base_im, rects = live_hit
        else:
            assert self._browser is not None
            page = self._browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1.0)
            try:
                hero_uri = _img_to_data_uri(hero_image_path)
                if template_kind in ("official_gov_doc", "gov_web_portal"):
                    html_doc = _build_moj_statute_html(heading, paragraphs, quote, citation, width, height)
                else:
                    html_doc = _build_fullbleed_news_html(
                        heading, paragraphs, quote, citation, width, height, hero_data_uri=hero_uri
                    )
                page.set_content(html_doc, wait_until="domcontentloaded")
                raw_png = out_prefix.parent / f"{out_prefix.name}_raw.png"
                page.screenshot(path=str(raw_png))
                rects = page.evaluate(
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

        wm_sz = 16 if width >= 1200 else 18
        beats: list[Path] = []

        b1 = base_im.copy()
        if include_watermark and citation:
            draw_citation_watermark(ImageDraw.Draw(b1), citation, x=8, y=5, font_size=wm_sz)
        p1 = out_prefix.parent / f"{out_prefix.name}_beat1.png"
        b1.save(p1)
        beats.append(p1)

        b2 = _crop_and_draw_red_underline(base_im, rects, zoom=1.38, underline_progress=0.55)
        if include_watermark and citation:
            draw_citation_watermark(ImageDraw.Draw(b2), citation, x=8, y=5, font_size=wm_sz)
        p2 = out_prefix.parent / f"{out_prefix.name}_beat2.png"
        b2.save(p2)
        beats.append(p2)

        b3 = _crop_and_draw_red_underline(base_im, rects, zoom=1.50, underline_progress=1.0)
        if include_watermark and citation:
            draw_citation_watermark(ImageDraw.Draw(b3), citation, x=8, y=5, font_size=wm_sz)
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
        hero_image_path: Path | None = None,
    ) -> list[Path]:
        assert self._browser is not None
        page = self._browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1.0)
        hero_uri = _img_to_data_uri(hero_image_path)
        wm_sz = 16 if width >= 1200 else 18
        beats: list[Path] = []
        accent_themes = [
            ("#1565C0", "#D90416"),
            ("#2E7D32", "#C62828"),
            ("#4527A0", "#D84315"),
            ("#00695C", "#AD1457"),
            ("#283593", "#BF360C"),
        ]
        accent_pair = accent_themes[self._cmp_call_idx % len(accent_themes)]
        self._cmp_call_idx += 1
        try:
            for b_idx, rev in enumerate([False, True], 1):
                html_doc = _build_visual_comparison_html(
                    heading,
                    left_title,
                    left_points,
                    right_title,
                    right_points,
                    delta_badge,
                    width,
                    height,
                    hero_data_uri=hero_uri,
                    reveal_right=rev,
                    accent_pair=accent_pair,
                )
                page.set_content(html_doc, wait_until="domcontentloaded")
                bp = out_prefix.parent / f"{out_prefix.name}_cmp{b_idx}.png"
                bp.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(bp))
                if citation:
                    im = Image.open(bp).convert("RGB")
                    draw_citation_watermark(ImageDraw.Draw(im), citation, x=8, y=5, font_size=wm_sz)
                    im.save(bp)
                beats.append(bp)
        finally:
            page.close()
        return beats

    def render_scrolling_comment_sheet(
        self,
        comments: list[dict[str, str]],
        width: int,
        stage_height: int,
        out_path: Path,
    ) -> Path:
        """Render a tall `width x (stage_height + 900)` `#FFFFFF` YouTube comment sheet for 30fps scrolling."""
        assert self._browser is not None
        tall_h = stage_height + 900
        seed_off = self._cmt_call_idx
        self._cmt_call_idx += 1
        page = self._browser.new_page(viewport={"width": width, "height": tall_h}, device_scale_factor=1.0)
        try:
            html_doc = _build_tall_youtube_comments_html(comments, width, tall_h, seed_offset=seed_off)
            page.set_content(html_doc, wait_until="domcontentloaded")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(out_path))
        finally:
            page.close()
        return out_path

    def render_comment_wall_beats(
        self,
        comments: list[dict[str, str]],
        width: int,
        height: int,
        out_prefix: Path,
    ) -> list[Path]:
        """Backwards-compatible 3-frame sliced wrapper over the tall YouTube comment sheet."""
        sheet_p = self.render_scrolling_comment_sheet(
            comments, width, height, out_prefix.parent / f"{out_prefix.name}_sheet.png"
        )
        sheet = Image.open(sheet_p).convert("RGB")
        beats: list[Path] = []
        for idx, y0 in enumerate([0, 220, 460], 1):
            bp = out_prefix.parent / f"{out_prefix.name}_cmt{idx}.png"
            sheet.crop((0, y0, width, y0 + height)).save(bp)
            beats.append(bp)
        return beats
