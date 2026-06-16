#!/usr/bin/env python3
"""Rolling-origin baseline forecasts for spread_snapshots.

This is an offline research tool inspired by Forecasting: Principles and Practice:
compare simple baselines before relying on more complex trading signals.
"""
from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from collections import defaultdict
from statistics import mean


def _rows(db_path: str, metric: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        q = f"""
            SELECT timestamp, expiry, strike, {metric} AS y
            FROM spread_snapshots
            WHERE {metric} IS NOT NULL AND executable = 1
            ORDER BY expiry, strike, timestamp
        """
        return conn.execute(q).fetchall()
    finally:
        conn.close()


def _forecast(method: str, hist: list[float], season: int) -> float | None:
    if not hist:
        return None
    if method == "naive":
        return hist[-1]
    if method == "mean":
        return mean(hist)
    if method == "drift":
        if len(hist) < 2:
            return hist[-1]
        return hist[-1] + (hist[-1] - hist[0]) / (len(hist) - 1)
    if method == "seasonal_naive":
        if len(hist) > season:
            return hist[-season]
        return hist[-1]
    raise ValueError(method)


def run(db_path: str, metric: str, horizon: int, min_train: int, season: int, out_csv: str) -> int:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in _rows(db_path, metric):
        grouped[(row["expiry"], int(row["strike"]))].append(float(row["y"]))

    methods = ["naive", "mean", "drift", "seasonal_naive"]
    stats = {m: {"abs": [], "sq": [], "scale": [], "dir_ok": []} for m in methods}

    for series in grouped.values():
        if len(series) <= min_train + horizon:
            continue
        scale_errors = [abs(series[i] - series[i - 1]) for i in range(1, len(series))]
        mase_scale = mean(scale_errors) if scale_errors else 0.0
        for origin in range(min_train, len(series) - horizon):
            hist = series[:origin]
            actual = series[origin + horizon - 1]
            last = hist[-1]
            for method in methods:
                pred = _forecast(method, hist, season)
                if pred is None:
                    continue
                err = actual - pred
                stats[method]["abs"].append(abs(err))
                stats[method]["sq"].append(err * err)
                if mase_scale > 0:
                    stats[method]["scale"].append(abs(err) / mase_scale)
                stats[method]["dir_ok"].append(1 if (actual - last) * (pred - last) >= 0 else 0)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "horizon", "method", "n", "mae", "rmse", "mase", "direction_hit_rate"])
        for method in methods:
            n = len(stats[method]["abs"])
            if not n:
                continue
            writer.writerow([
                metric,
                horizon,
                method,
                n,
                mean(stats[method]["abs"]),
                math.sqrt(mean(stats[method]["sq"])),
                mean(stats[method]["scale"]) if stats[method]["scale"] else "",
                mean(stats[method]["dir_ok"]),
            ])
    return sum(len(v) for v in grouped.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path")
    parser.add_argument("--metric", choices=["maker_net_sell", "maker_net_buy"], default="maker_net_sell")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min-train", type=int, default=30)
    parser.add_argument("--season", type=int, default=12)
    parser.add_argument("--out", default="forecast_backtest_results.csv")
    args = parser.parse_args()
    n = run(args.db_path, args.metric, args.horizon, args.min_train, args.season, args.out)
    print(f"processed_points={n} output={args.out}")


if __name__ == "__main__":
    main()
