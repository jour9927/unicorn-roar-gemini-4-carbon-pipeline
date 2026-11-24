"""Automated unit & integration tests for V2 獨角獸那麼叫 (@unicorn-roar) pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from unicorn_roar.composer.video_composer import _render_short_header_banner
from unicorn_roar.discovery.topic_radar import discover_and_select_topic
from unicorn_roar.discovery.web_crawler import SourceCitation
from unicorn_roar.scriptwriter.distiller import (
    SIX_ACT_SCHEMA,
    load_distilled_playbook,
    retrieve_act_exemplars,
)
from unicorn_roar.scriptwriter.generator import KineticLine, SceneSpec
from unicorn_roar.visuals.chrome_cards import ChromeEvidenceRenderer
from unicorn_roar.visuals.image_gen import render_kinetic_black_beats


def test_source_citation_watermark_format() -> None:
    cite = SourceCitation(
        source_id="S1",
        publisher="衛生福利部食藥署",
        title="果蔬汁品名及標示規定問答集 - 食藥署",
        url="https://www.fda.gov.tw/example",
        published_date="2026-10-01",
        accessed_date="20261125",
        excerpt="原汁含有率達百分之十以上者始得標示為果蔬汁。",
        paragraphs=["依食品安全衛生管理法第22條規定，原汁含有率達百分之十以上者始得標示為果蔬汁。"],
    )
    wm = cite.onscreen_watermark
    assert "衛生福利部食藥署" in wm
    assert "accessed:20261125" in wm


def test_v2_chrome_renderer_and_kinetic_beats(tmp_path: Path) -> None:
    cite = SourceCitation(
        source_id="S1",
        publisher="農業部漁業署",
        title="臺灣鯛品名標示管理規範函釋",
        url="https://www.fa.gov.tw/example",
        published_date="2026-04-07",
        accessed_date="20261125",
        excerpt="倘產品實際投入原料確實為臺灣鯛（吳郭魚），品名標示僅見鯛魚字樣，恐有易生誤解之虞。",
        paragraphs=[
            "近期民眾反映市售便當及餐飲通路將吳郭魚標示為鯛魚販售，引發消費資訊混淆。",
            "倘產品實際投入原料確實為臺灣鯛（吳郭魚），品名標示僅見鯛魚字樣，恐有易生誤解之虞，應依食品安全衛生管理法第28條辦理。",
        ],
    )
    with ChromeEvidenceRenderer() as cr:
        doc_beats = cr.render_two_beat_evidence(
            template_kind="official_gov_doc",
            heading=cite.title,
            paragraphs=cite.paragraphs,
            quote="實際投入原料確實為臺灣鯛（吳郭魚）",
            citation=cite,
            width=1920,
            height=1080,
            out_prefix=tmp_path / "gov_doc",
        )
        assert len(doc_beats) == 3
        assert all(p.exists() and p.stat().st_size > 25_000 for p in doc_beats)

        cmt_beats = cr.render_comment_wall_beats(
            comments=[
                {"user": "@tw_foodie", "time": "2小時前", "text": "原來便當店的鯛魚一直是吳郭魚！", "likes": 512},
                {"user": "@keelung_fish", "time": "3小時前", "text": "真鯛跟吳郭魚價差好幾倍耶", "likes": 318},
            ],
            width=1920,
            height=1080,
            out_prefix=tmp_path / "cmts",
        )
        assert len(cmt_beats) == 3

    kin_scene = SceneSpec(
        scene_id="L03",
        scene_mode="kinetic_black",
        narration="中位數乘上三分之二直接貫破地板。",
        kinetic_lines=[
            KineticLine(text="你: 3萬 排名50/100", highlight="3萬", highlight_color="yellow"),
            KineticLine(text="他: 3百萬 排名1/100", highlight="3百萬", highlight_color="red"),
        ],
    )
    k_beats = render_kinetic_black_beats(kin_scene, 1920, 1080, tmp_path / "kin")
    assert len(k_beats) == 3

    hdr = _render_short_header_banner(1080, 350, "台灣沒有麵包", "只有狠活", "red")
    assert hdr.size == (1080, 350)


def test_distilled_playbook_and_breath_cutter() -> None:
    import importlib.util

    pb = load_distilled_playbook()
    assert pb["empirical_metrics"]["long_episode_count"] >= 20
    exs = retrieve_act_exemplars(pb, "food_labeling", SIX_ACT_SCHEMA[0]["act_id"])
    assert len(exs) == 2

    spec = importlib.util.spec_from_file_location(
        "rtx_eng", Path(__file__).resolve().parents[1] / "scripts/rtx6000_gpu_engine.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    sr = 24000
    tone = 0.25 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
    long_sil = np.zeros(int(0.6 * sr), dtype=np.float32)
    combined = np.concatenate([tone, long_sil, tone])
    trimmed = mod.tight_cut_breaths(combined, sr)
    assert len(trimmed) < len(combined) - int(0.45 * sr)
