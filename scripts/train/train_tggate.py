# -*- coding: utf-8 -*-
"""
# TG-GATE SSL 学習エントリポイント（シム）

実体は `lib/trainer/entry.py` に移した（REFACTOR_PLAN.md §5-0「scripts/ はインフラ専用、
研究コードは lib/」）。既存27実験の `run_slurm.sh` がこのパスを直接叩くため、パスと
CLI引数の互換を保つ薄い呼び出し口としてのみここに残す。

@author: Katsuhisa MORITA
"""
from lib.trainer.entry import build_parser, run_from_args

if __name__ == '__main__':
    args = build_parser().parse_args()
    run_from_args(args, filename='train_tggate')
