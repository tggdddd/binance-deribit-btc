#!/usr/bin/env python3
"""Rolling-origin baseline forecasts for spread_snapshots.

Offline research tool inspired by *Forecasting: Principles and Practice* (FPP):
start with transparent benchmark methods, use time-series cross-validation, report
scale-free errors, and inspect residuals before promoting any signal into live
trading thresholds.
"""
from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, median

METRICS = {"maker_net_sell", "maker_net_buy", "net_sell", "net_buy", "spread_sell", "spread_buy"}
BASE_METHODS = ("naive", "mean", "drift", "seasonal_naive")
SIMPLE_UNION_METHOD = "simple_union"
METHODS = (*BASE_METHODS, SIMPLE_UNION_METHOD)


@dataclass(frozen=True)
class Observation:
    timestamp: str
    expiry: str
    strike: int
    y: float


def _rows(db_path: str, metric: str) -> list[Observation]:
    if metric not in METRICS:
        raise ValueError(f"unsupported metric={metric!r}; choose one of {sorted(METRICS)}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        q = f"""
            SELECT timestamp, expiry, strike, {metric} AS y
            FROM spread_snapshots
            WHERE {metric} IS NOT NULL AND executable = 1
            ORDER BY expiry, strike, timestamp
        """
        return [
            Observation(str(r["timestamp"]), str(r["expiry"]), int(r["strike"]), float(r["y"]))
            for r in conn.execute(q).fetchall()
        ]
    finally:
        conn.close()


def _forecast(method: str, hist: list[float], season: int, horizon: int) -> float | None:
    if not hist:
        return None
    if method == "naive":
        return hist[-1]
    if method == "mean":
        return mean(hist)
    if method == "drift":
        if len(hist) < 2:
            return hist[-1]
        return hist[-1] + horizon * (hist[-1] - hist[0]) / (len(hist) - 1)
    if method == "seasonal_naive":
        if len(hist) > season:
            idx = len(hist) - season + ((horizon - 1) % season)
            return hist[idx] if 0 <= idx < len(hist) else hist[-season]
        return hist[-1]
    if method == SIMPLE_UNION_METHOD:
        preds = [_forecast(m, hist, season, horizon) for m in BASE_METHODS]
        preds = [p for p in preds if p is not None]
        return median(preds) if preds else None
    raise ValueError(method)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def _summarize(errors: list[float], abs_scaled: list[float], dir_ok: list[int], inside80: list[int], inside95: list[int]) -> dict[str, float | int | str]:
    n = len(errors)
    if not n:
        return {}
    abs_errors = [abs(e) for e in errors]
    return {
        "n": n,
        "bias": mean(errors),
        "mae": mean(abs_errors),
        "rmse": math.sqrt(mean([e * e for e in errors])),
        "mase": mean(abs_scaled) if abs_scaled else "",
        "median_ae": median(abs_errors),
        "p90_ae": _quantile(abs_errors, 0.90),
        "direction_hit_rate": mean(dir_ok),
        "pi80_coverage": mean(inside80) if inside80 else "",
        "pi95_coverage": mean(inside95) if inside95 else "",
    }


def run(db_path: str, metric: str, horizon: int, min_train: int, season: int,
        out_csv: str, residuals_out: str | None) -> int:
    grouped: dict[tuple[str, int], list[Observation]] = defaultdict(list)
    for row in _rows(db_path, metric):
        grouped[(row.expiry, row.strike)].append(row)

    stats = {m: {"errors": [], "scale": [], "dir_ok": [], "inside80": [], "inside95": []} for m in METHODS}
    residual_rows: list[list[object]] = []

    for (expiry, strike), observations in grouped.items():
        series = [o.y for o in observations]
        if len(series) <= min_train + horizon:
            continue
        scale_errors = [abs(series[i] - series[i - 1]) for i in range(1, len(series))]
        mase_scale = mean(scale_errors) if scale_errors else 0.0
        for origin in range(min_train, len(series) - horizon + 1):
            hist = series[:origin]
            actual_idx = origin + horizon - 1
            actual = series[actual_idx]
            actual_ts = observations[actual_idx].timestamp
            last = hist[-1]
            for method in METHODS:
                pred = _forecast(method, hist, season, horizon)
                if pred is None:
                    continue
                historical_residuals: list[float] = []
                interval_candidates: list[tuple[float, float, float, float]] = []
                interval_methods = BASE_METHODS if method == SIMPLE_UNION_METHOD else (method,)
                for interval_method in interval_methods:
                    method_residuals: list[float] = []
                    for back_origin in range(min_train, origin):
                        back_hist = series[:back_origin]
                        back_idx = back_origin + horizon - 1
                        if back_idx >= origin:
                            break
                        back_pred = _forecast(interval_method, back_hist, season, horizon)
                        if back_pred is not None:
                            method_residuals.append(series[back_idx] - back_pred)
                    if method == SIMPLE_UNION_METHOD:
                        if len(method_residuals) >= 20:
                            interval_candidates.append((
                                pred + _quantile(method_residuals, 0.10),
                                pred + _quantile(method_residuals, 0.90),
                                pred + _quantile(method_residuals, 0.025),
                                pred + _quantile(method_residuals, 0.975),
                            ))
                    else:
                        historical_residuals = method_residuals

                lo80 = hi80 = lo95 = hi95 = ""
                in80 = in95 = ""
                if method == SIMPLE_UNION_METHOD and interval_candidates:
                    lo80 = min(c[0] for c in interval_candidates)
                    hi80 = max(c[1] for c in interval_candidates)
                    lo95 = min(c[2] for c in interval_candidates)
                    hi95 = max(c[3] for c in interval_candidates)
                elif len(historical_residuals) >= 20:
                    lo80 = pred + _quantile(historical_residuals, 0.10)
                    hi80 = pred + _quantile(historical_residuals, 0.90)
                    lo95 = pred + _quantile(historical_residuals, 0.025)
                    hi95 = pred + _quantile(historical_residuals, 0.975)
                if lo80 != "":
                    in80 = int(float(lo80) <= actual <= float(hi80))
                    in95 = int(float(lo95) <= actual <= float(hi95))
                    stats[method]["inside80"].append(in80)
                    stats[method]["inside95"].append(in95)

                err = actual - pred
                stats[method]["errors"].append(err)
                if mase_scale > 0:
                    stats[method]["scale"].append(abs(err) / mase_scale)
                stats[method]["dir_ok"].append(1 if (actual - last) * (pred - last) >= 0 else 0)
                residual_rows.append([
                    metric, horizon, method, expiry, strike, origin, actual_ts,
                    actual, pred, err, abs(err), lo80, hi80, lo95, hi95, in80, in95,
                ])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["metric", "horizon", "method", "n", "bias", "mae", "rmse", "mase",
                      "median_ae", "p90_ae", "direction_hit_rate", "pi80_coverage", "pi95_coverage"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHODS:
            row = _summarize(stats[method]["errors"], stats[method]["scale"], stats[method]["dir_ok"],
                             stats[method]["inside80"], stats[method]["inside95"])
            if row:
                writer.writerow({"metric": metric, "horizon": horizon, "method": method, **row})

    if residuals_out:
        with open(residuals_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "horizon", "method", "expiry", "strike", "origin", "timestamp",
                             "actual", "forecast", "residual", "abs_error", "lo80", "hi80", "lo95", "hi95",
                             "inside80", "inside95"])
            writer.writerows(residual_rows)
    return sum(len(v) for v in grouped.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling-origin spread forecast benchmark")
    parser.add_argument("db_path")
    parser.add_argument("--metric", choices=sorted(METRICS), default="maker_net_sell")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min-train", type=int, default=30)
    parser.add_argument("--season", type=int, default=12, help="seasonal period in snapshot rows")
    parser.add_argument("--out", default="forecast_backtest_results.csv")
    parser.add_argument("--residuals-out", default="forecast_residuals.csv",
                        help="write per-origin residuals for diagnostic checks; empty disables")
    args = parser.parse_args()
    if args.horizon < 1 or args.min_train < 2 or args.season < 1:
        raise SystemExit("horizon>=1, min-train>=2 and season>=1 are required")
    residuals_out = args.residuals_out or None
    n = run(args.db_path, args.metric, args.horizon, args.min_train, args.season, args.out, residuals_out)
    print(f"processed_points={n} summary={args.out} residuals={residuals_out or 'disabled'}")


if __name__ == "__main__":
    main()
