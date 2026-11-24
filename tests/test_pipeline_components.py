"""Automated tests for 獨角獸那麼叫 (@unicorn-roar) pipeline components."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from unicorn_roar.composer.video_composer import _render_slanted_hook_banner
from unicorn_roar.discovery.topic_radar import discover_and_select_topic
from unicorn_roar.discovery.web_crawler import SourceCitation
from unicorn_roar.scriptwriter.generator import KineticLine, SceneSpec
from unicorn_roar.visuals.image_gen import (
    render_evidence_doc_card,
    render_kinetic_black_card,
)


def test_source_citation_watermark_format() -> None:
    cite = SourceCitation(
        source_id="S1",
        publisher="衛生福利部食藥署",
        title="果蔬汁品名及標示規定問答集 - 食藥署",
        url="https://www.fda.gov.tw/example",
        published_date="2026-10-01",
        accessed_date="20261125",
        excerpt="原汁含有率達百分之十以上者始得標示為果蔬汁。",
    )
    wm = cite.onscreen_watermark
    assert "衛生福利部食藥署" in wm
    assert "accessed:20261125" in wm


def test_render_evidence_and_kinetic_cards(tmp_path: Path) -> None:
    cite = SourceCitation(
        source_id="S1",
        publisher="農業部漁業署",
        title="115年4月7日漁四字第1151404303A號函",
        url="https://www.fa.gov.tw/example",
        published_date="2026-04-07",
        accessed_date="20261125",
        excerpt="倘產品實際投入原料確實為臺灣鯛（吳郭魚），品名標示僅見鯛魚字樣，恐有易生誤解之虞。",
    )
    ev_scene = SceneSpec(
        scene_id="L02",
        scene_mode="evidence_doc",
        narration="如果實際投入原料是臺灣鯛，也就是吳郭魚，卻只標示鯛魚，恐有誤解。",
        citation_id="S1",
        document_heading="農業部漁業署函釋：臺灣鯛（吳郭魚）品名標示規範",
        document_excerpt=cite.excerpt,
        red_underline_quote="實際投入原料確實為臺灣鯛（吳郭魚）",
    )
    ev_out = render_evidence_doc_card(ev_scene, cite, 1920, 1080, tmp_path / "ev.png")
    assert ev_out.exists() and ev_out.stat().st_size > 20_000

    kin_scene = SceneSpec(
        scene_id="L03",
        scene_mode="kinetic_black",
        narration="中位數乘上三分之二直接貫破地板。",
        kinetic_lines=[
            KineticLine(text="你: 3萬 排名50/100", highlight="3萬", highlight_color="yellow"),
            KineticLine(text="他: 3百萬 排名1/100", highlight="3百萬", highlight_color="red"),
        ],
    )
    kin_out = render_kinetic_black_card(kin_scene, cite, 1920, 1080, tmp_path / "kin.png")
    assert kin_out.exists() and kin_out.stat().st_size > 10_000

    hook = _render_slanted_hook_banner(1080, 360, "說吳郭魚是鯛魚", "的人 回答我!", ["吳郭魚", "鯛魚", "回答我!"])
    assert hook.size == (1080, 360)


def test_manual_topic_selection() -> None:
    winner, cands = discover_and_select_topic(manual_topic="全麥麵包只有10%全麥也能叫全麥嗎？")
    assert winner.rank == 1
    assert "全麥" in winner.proposed_title
    assert len(cands) == 1


def test_distilled_playbook_retrieval() -> None:
    from unicorn_roar.scriptwriter.distiller import (
        SIX_ACT_SCHEMA,
        load_distilled_playbook,
        retrieve_act_exemplars,
    )

    pb = load_distilled_playbook()
    assert pb["empirical_metrics"]["long_episode_count"] >= 20
    assert pb["empirical_metrics"]["long_avg_cjk_chars"] >= 3800
    exs = retrieve_act_exemplars(pb, "food_labeling", SIX_ACT_SCHEMA[0]["act_id"])
    assert len(exs) == 2
    assert len(exs[0]["excerpt"]) >= 120

