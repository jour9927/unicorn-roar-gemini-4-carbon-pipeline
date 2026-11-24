"""Command-Line Interface for 獨角獸那麼叫 (@unicorn-roar) Unattended Video Pipeline."""
from __future__ import annotations

import argparse
import json

from unicorn_roar.pipeline import run_unattended_episode


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="unicorn-roar",
        description="獨角獸那麼叫 (@unicorn-roar) Gemini 4 Carbon 無人值守六幕長篇雙軌產線（12-15分鐘正片 + 2分鐘推流Short）",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    run_p = sub.add_parser("run", help="無人值守一鍵生成 12-15 分鐘 16:9 正片 + 2 分鐘 9:16 推流 Short")
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
    run_p.add_argument(
        "--scale",
        type=float,
        default=0.85,
        help="篇幅倍率（預設 0.85 ≈ 3,900-4,200 字 / 13-15 分鐘正片；1.0 ≈ 4,700 字 / 18-19 分鐘深度長片）",
    )
    run_p.add_argument("--run-id", type=str, default=None, help="指定輸出批次 ID")

    args = parser.parse_args()
    summary = run_unattended_episode(
        pillar=getattr(args, "pillar", None),
        manual_topic=getattr(args, "topic", None),
        scale_factor=float(getattr(args, "scale", 1.0)),
        run_id=getattr(args, "run_id", None),
    )
    print(
        json.dumps(
            {
                "ok": summary["ok"],
                "run_id": summary["run_id"],
                "long_title": summary["long_title"],
                "long_cjk_chars": summary["long_cjk_chars"],
                "short_title": summary["short_title"],
                "short_cjk_chars": summary["short_cjk_chars"],
                "outputs": summary["outputs"],
                "verification": summary["verification"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
