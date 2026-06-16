#!/usr/bin/env python3
"""Residual diagnostics for spread forecast CSV output.

Accepts either the summary CSV or the per-origin residual CSV produced by
spread_forecast_backtest.py. The residual path is preferred because FPP-style
model checking focuses on residual bias, autocorrelation and interval coverage,
not just aggregate error tables.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from statistics import mean, median


def _mad(values: list[float]) -> float:
    if not values:
        return 0.0
    m = median(values)
    return median([abs(v - m) for v in values])


def _acf(values: list[float], lag: int) -> float:
    if len(values) <= lag or lag < 1:
        return 0.0
    mu = mean(values)
    denom = sum((v - mu) ** 2 for v in values)
    if denom == 0:
        return 0.0
    return sum((values[i] - mu) * (values[i - lag] - mu) for i in range(lag, len(values))) / denom


def _ljung_box_q(values: list[float], max_lag: int) -> float:
    n = len(values)
    if n <= max_lag + 1:
        return 0.0
    return n * (n + 2) * sum((_acf(values, k) ** 2) / (n - k) for k in range(1, max_lag + 1))


def _float(row: dict[str, str], key: str) -> float | None:
    val = row.get(key, "")
    if val in (None, ""):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _diagnose_residual_rows(rows: list[dict[str, str]], max_lag: int) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        grouped[(r.get("metric", ""), r.get("horizon", ""), r.get("method", ""))].append(r)

    print("metric,horizon,method,n,bias,mae,rmse,acf1,ljung_box_q,mad_abs_error,pi80_coverage,pi95_coverage,flags")
    for (metric, horizon, method), group in sorted(grouped.items()):
        residuals = [_float(r, "residual") for r in group]
        residuals = [v for v in residuals if v is not None]
        if not residuals:
            continue
        abs_errors = [abs(v) for v in residuals]
        inside80 = [_float(r, "inside80") for r in group]
        inside95 = [_float(r, "inside95") for r in group]
        inside80 = [v for v in inside80 if v is not None]
        inside95 = [v for v in inside95 if v is not None]
        acf1 = _acf(residuals, 1)
        q = _ljung_box_q(residuals, min(max_lag, max(1, len(residuals) // 5)))
        se_bias = math.sqrt(mean([(v - mean(residuals)) ** 2 for v in residuals]) / len(residuals)) if len(residuals) > 1 else 0.0
        flags = []
        if se_bias > 0 and abs(mean(residuals)) > 2 * se_bias:
            flags.append("biased")
        if abs(acf1) > 2 / math.sqrt(len(residuals)):
            flags.append("autocorrelated")
        if inside80 and mean(inside80) < 0.70:
            flags.append("pi80_undercoverage")
        if inside95 and mean(inside95) < 0.90:
            flags.append("pi95_undercoverage")
        bias = mean(residuals)
        mae = mean(abs_errors)
        rmse = math.sqrt(mean([v * v for v in residuals]))
        pi80 = f"{mean(inside80):.6f}" if inside80 else ""
        pi95 = f"{mean(inside95):.6f}" if inside95 else ""
        print(
            f"{metric},{horizon},{method},{len(residuals)},"
            f"{bias:.6f},{mae:.6f},{rmse:.6f},"
            f"{acf1:.6f},{q:.6f},{_mad(abs_errors):.6f},"
            f"{pi80},{pi95},{'|'.join(flags) if flags else 'ok'}"
        )


def _diagnose_summary_rows(rows: list[dict[str, str]]) -> None:
    maes = [float(r["mae"]) for r in rows if r.get("mae")]
    if not maes:
        print("no_rows")
        return
    med = median(maes)
    mad = _mad(maes)
    print(f"rows={len(rows)} median_mae={med:.6f} mad_mae={mad:.6f}")
    for r in rows:
        mae = float(r["mae"])
        flags = []
        if mad > 0 and abs(mae - med) > 3 * mad:
            flags.append("mae_outlier")
        if r.get("bias"):
            try:
                if abs(float(r["bias"])) > mae * 0.25:
                    flags.append("large_bias_vs_mae")
            except ValueError:
                pass
        if flags:
            print(f"flag method={r.get('method')} metric={r.get('metric')} horizon={r.get('horizon')} mae={mae:.6f} flags={'|'.join(flags)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--max-lag", type=int, default=10)
    args = parser.parse_args()
    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("no_rows")
        return
    if "residual" in rows[0]:
        _diagnose_residual_rows(rows, args.max_lag)
    else:
        _diagnose_summary_rows(rows)


if __name__ == "__main__":
    main()
