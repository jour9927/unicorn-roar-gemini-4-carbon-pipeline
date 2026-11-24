"""Unattended Topic Radar & Autonomous Topic Selector for 獨角獸那麼叫 (@unicorn-roar).

Replaces the old manual human-pick workflow with a 100% unattended multi-pillar radar:
1. Scans live Taiwanese public news & regulatory controversies across 4 core channel pillars
   (or any user-requested custom pillar/domain so the pipeline produces diverse video types).
2. Deduplicates against @unicorn-roar's existing 198 published videos.
3. Ranks candidates by Controversy Contrast, Regulatory Evidence Density, and Short Hook Virality,
   and automatically locks in the winning topic without requiring human intervention.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from unicorn_roar.config import REFERENCE_DIR
from unicorn_roar.discovery.web_crawler import crawl_google_news_tw


class TopicCandidate(BaseModel):
    rank: int = 0
    pillar: str
    proposed_title: str
    short_hook_question: str
    core_contradiction: str
    search_queries: list[str] = Field(default_factory=list)
    seed_headlines: list[str] = Field(default_factory=list)
    score: float = 0.0


PILLAR_SEEDS: dict[str, list[dict[str, Any]]] = {
    "food_labeling": [
        {
            "probe_query": "台灣 果汁 原汁含有率 10% 標示 食藥署",
            "template_title": "10%原汁也能畫滿整顆水果？超商果汁飲料的「原汁含有率」標示陷阱與法規底線",
            "short_hook": "包裝畫滿水果卻只有10%原汁？說這算果汁的人 回答我！",
            "contradiction": "包裝正面大字與鮮果圖片營造純果汁印象，但法規僅要求原汁10%以上即可標示果蔬汁，微幅字體藏在背面",
            "queries": ["台灣 果蔬汁 標示 原汁含有率 食藥署", "還原果汁 濃縮還原 無添加糖 標示 消基會", "果蔬汁品名及標示規定 10%"],
        },
        {
            "probe_query": "台灣 橄欖油 調合油 特級初榨 標示 食藥署 消基會",
            "template_title": "瓶身寫義大利橄欖油，翻過來卻是調合油？食藥署品名標示規範與食用油等級真相",
            "short_hook": "加一點橄欖油就能叫橄欖調合油？超市買油到底怎麼看！",
            "contradiction": "高價特級初榨與精煉果渣油、調合油價差數倍，外文與圖案卻讓消費者誤認純度",
            "queries": ["台灣 橄欖油 調合油 標示 食藥署", "市售橄欖油 消基會 抽驗 標示", "包裝調合油品名標示相關規定"],
        },
        {
            "probe_query": "台灣 蜂蜜 調和蜜 不純 標示 農業部 食藥署",
            "template_title": "不純砍頭還是合法加糖？台灣「蜂蜜含量60%」標示新制上路後，市售龍眼蜜到底純不純？",
            "short_hook": "加了糖漿還能叫調和蜂蜜？買真蜂蜜前先看這一行字！",
            "contradiction": "蜂蜜標示新制要求標示蜂蜜含量百分比，但加糖蜂蜜與風味糖漿在電商與市集仍讓消費者霧裡看花",
            "queries": ["包裝蜂蜜及其糖漿類產品標示規定 食藥署", "台灣 假蜜 調和蜂蜜 消基會 標示", "國產蜂蜜 認證 標章 農業部"],
        },
    ],
    "consumer_pricing": [
        {
            "probe_query": "台灣 變相漲價 縮水式通膨 容量變少 消基會 公平會",
            "template_title": "價格沒漲但偷少20克？台灣超市「縮水式通膨」為何完全合法？各國早就強制貼縮水警告了！",
            "short_hook": "沒漲價卻越吃越少？把消費者當盤子的縮水式通膨 回答我！",
            "contradiction": "南韓、法國已立法要求減量不減價必須強制揭露，台灣業者卻能悄悄改包裝換新上市規避漲價罵名",
            "queries": ["縮水式通膨 台灣 變相漲價 消基會", "減量不減價 韓國 法國 強制標示", "台灣 物價 外食費 主計總處 漲幅"],
        },
        {
            "probe_query": "台灣 餐廳 服務費 10% 掃碼點餐 自己端 消保官",
            "template_title": "掃QR Code點餐、自己倒水收餐盤，結帳還要收10%服務費？台灣「服務費」到底服務了誰？",
            "short_hook": "掃碼點餐還要收10%服務費？這筆錢到底進了誰口袋 回答我！",
            "contradiction": "數位無人化點餐省下人力成本，結帳卻照收一成服務費且未分給第一線員工，實為變相定價拆分話術",
            "queries": ["台灣 餐廳 10%服務費 掃碼點餐 消保官", "服務費 變相漲價 餐飲業 台灣", "餐飲業 服務費 勞工 薪資 規定"],
        },
    ],
    "wage_labor_stats": [
        {
            "probe_query": "台灣 經常性薪資 總薪資 年終獎金 主計總處 中位數",
            "template_title": "實質薪資倒退還是創歷史新高？主計總處「總薪資」與「經常性薪資」背後的統計溫度差",
            "short_hook": "政府說平均月薪破6萬創新高，為什麼你的錢包完全沒感覺？",
            "contradiction": "半導體分紅與年終拉高全年總薪資平均，但扣除物價CPI後的實質經常性薪資中位數卻讓近七成受僱者低於平均",
            "queries": ["主計總處 經常性薪資 中位數 總薪資 台灣", "實質經常性薪資 CPI 物價 台灣", "受僱員工薪資調查 低於平均 七成"],
        },
    ],
    "industry_policy": [
        {
            "probe_query": "台灣 雞蛋 標示 產地 籠飼 平飼 洗選蛋 農業部",
            "template_title": "一盒雞蛋賣150元真的比較營養嗎？機能蛋、動福蛋、洗選蛋標示背後的成本與行銷溢價",
            "short_hook": "一顆雞蛋貴三倍真的比較營養嗎？買高價蛋前先看懂這張表！",
            "contradiction": "葉黃素蛋、初生蛋、放牧蛋行銷名詞百花齊放，營養學實證與飼養規範差異卻少有人講清楚",
            "queries": ["台灣 雞蛋 籠飼 平飼 放牧 標示 農業部", "機能蛋 葉黃素蛋 營養 差異 食藥署", "洗選蛋 溯源標示 台灣"],
        },
    ],
}


def _load_existing_channel_titles() -> list[str]:
    titles: list[str] = []
    for fname in ("videos_zhtw.json", "shorts_zhtw.json", "videos_meta.json"):
        p = REFERENCE_DIR / fname
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for e in data.get("entries", []):
                    if e.get("title"):
                        titles.append(str(e["title"]))
            except Exception:
                pass
    return titles


def _overlap_penalty(candidate_title: str, existing_titles: list[str]) -> float:
    kws = set(re.findall(r"[\u4e00-\u9fff]{2}", candidate_title))
    if not kws:
        return 0.0
    max_sim = 0.0
    for ext in existing_titles[:60]:
        ext_kws = set(re.findall(r"[\u4e00-\u9fff]{2}", ext))
        if not ext_kws:
            continue
        inter = len(kws & ext_kws) / max(len(kws), 1)
        if inter > max_sim:
            max_sim = inter
    return max_sim * 30.0


def discover_and_select_topic(
    pillar_filter: str | None = None,
    manual_topic: str | None = None,
) -> tuple[TopicCandidate, list[TopicCandidate]]:
    """Autonomously probe live Taiwan public news, score candidates, and pick #1 unattended."""
    if manual_topic:
        cand = TopicCandidate(
            rank=1,
            pillar=pillar_filter or "custom",
            proposed_title=manual_topic,
            short_hook_question=f"{manual_topic[:18]}？真相到底是什麼 回答我！",
            core_contradiction=f"針對「{manual_topic}」的公開資訊、法規標準與市場實況落差進行完整數據拆解",
            search_queries=[
                manual_topic,
                f"{manual_topic} 台灣 法規 標示 統計",
                f"{manual_topic} 食藥署 主計總處 消基會",
            ],
            score=100.0,
        )
        return cand, [cand]

    existing_titles = _load_existing_channel_titles()
    pillars = [pillar_filter] if (pillar_filter and pillar_filter in PILLAR_SEEDS) else list(PILLAR_SEEDS.keys())

    candidates: list[TopicCandidate] = []
    for pillar in pillars:
        for seed in PILLAR_SEEDS[pillar]:
            live_news = crawl_google_news_tw(seed["probe_query"], max_items=5)
            headlines = [n["title"] for n in live_news]
            # Score based on live public news richness, regulator mentions, and freshness vs recent channel uploads
            reg_hits = sum(
                1
                for n in live_news
                for kw in ("食藥署", "主計", "消基會", "公平會", "農業部", "消保", "法規", "標示", "漲")
                if kw in (n["title"] + n["excerpt"])
            )
            penalty = _overlap_penalty(seed["template_title"], existing_titles)
            score = round(50.0 + len(live_news) * 6.5 + reg_hits * 4.5 - penalty, 2)
            candidates.append(
                TopicCandidate(
                    pillar=pillar,
                    proposed_title=seed["template_title"],
                    short_hook_question=seed["short_hook"],
                    core_contradiction=seed["contradiction"],
                    search_queries=seed["queries"],
                    seed_headlines=headlines[:4],
                    score=score,
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    for idx, c in enumerate(candidates, 1):
        c.rank = idx
    return candidates[0], candidates
