#!/usr/bin/env python3
"""Lightweight residual diagnostics for forecast_backtest_results CSV output."""
from __future__ import annotations

import argparse
import csv
from statistics import median


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    m = median(values)
    return median([abs(v - m) for v in values])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    maes = [float(r["mae"]) for r in rows if r.get("mae")]
    if not maes:
        print("no_rows")
        return
    med = median(maes)
    mad = _mad(maes)
    print(f"rows={len(rows)} median_mae={med:.6f} mad_mae={mad:.6f}")
    for r in rows:
        mae = float(r["mae"])
        if mad > 0 and abs(mae - med) > 3 * mad:
            print(f"outlier method={r['method']} metric={r['metric']} horizon={r['horizon']} mae={mae:.6f}")


if __name__ == "__main__":
    main()
