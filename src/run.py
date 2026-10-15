"""Command-line entry point for WiTra training, evaluation and aggregation.

Defaults reproduce the paper's compact configuration (depth 2, width 160,
FFN ratio 1, eight heads, BF16/SDPA, floor-head correction for MAIN).
"""
from __future__ import annotations

import trainer


def main():
    args = trainer.parser().parse_args()
    return {"train": trainer.train, "evaluate": trainer.evaluate,
            "aggregate": trainer.aggregate}[args.command](args)


if __name__ == "__main__":
    main()
