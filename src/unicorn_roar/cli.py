"""Command-Line Interface for 獨角獸那麼叫 (@unicorn-roar) Unattended Video Pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from unicorn_roar.pipeline import run_unattended_episode


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="unicorn-roar",
        description="獨角獸那麼叫 (@unicorn-roar) Gemini 4 Carbon 無人值守長短片雙軌產線",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    run_p = sub.add_parser("run", help="無人值守一鍵生成 16:9 長片 + 9:16 推流 Short")
    run_p.add_argument(
        "--pillar",
        choices=["food_labeling", "consumer_pricing", "wage_labor_stats", "industry_policy"],
        default=None,
        help="指定主題類別（預設自動跨類別評分挑選最高分議題）",
    )
    run_p.add_argument(
        "--topic",
        type=str,
        default=None,
        help="手動指定題目（預設留空即為 100%% 無人值守自動選題）",
    )
    run_p.add_argument("--long-scenes", type=int, default=10, help="長片場景數（預設 10）")
    run_p.add_argument("--short-scenes", type=int, default=5, help="Short 場景數（預設 5）")
    run_p.add_argument("--run-id", type=str, default=None, help="指定輸出批次 ID")

    args = parser.parse_args()
    summary = run_unattended_episode(
        pillar=getattr(args, "pillar", None),
        manual_topic=getattr(args, "topic", None),
        long_scenes=getattr(args, "long_scenes", 10),
        short_scenes=getattr(args, "short_scenes", 5),
        run_id=getattr(args, "run_id", None),
    )
    print(
        json.dumps(
            {
                "ok": summary["ok"],
                "run_id": summary["run_id"],
                "long_title": summary["long_title"],
                "short_title": summary["short_title"],
                "outputs": summary["outputs"],
                "verification": summary["verification"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
