#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Project-relative raw inputs and processed outputs
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "external"

RAW_UCI = OUT_DIR / "raw" / "household_power_consumption.txt"
RAW_NSRDB = OUT_DIR / "raw" / "4466719_25.92_117.73_2020.csv"

OUT_UCI = OUT_DIR / "uci_household_hourly.csv"
OUT_NSRDB = OUT_DIR / "nsrdb_fujian_2020_hourly.csv"

UCI_CHUNKSIZE = 250_000

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(path: Path) -> str:
    return sha256_file(path)


def write_csv(df: pd.DataFrame, path: Path, float_fmt: dict[str, str]) -> None:
    """Deterministic UTF-8 / LF CSV writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8",
              float_format=None, na_rep="")


def fmt_float(x: float, digits: int) -> str:
    return f"{x:.{digits}f}"


# --------------------------------------------------------------------------
# 1. UCI household power -> hourly mean load in kW
# --------------------------------------------------------------------------


def build_uci_hourly() -> dict:
    if not RAW_UCI.exists():
        raise FileNotFoundError(f"raw UCI file not found: {RAW_UCI}")

    provider = pd.read_csv(RAW_UCI, sep=";", nrows=0)
    raw_cols = list(provider.columns)

    # Accumulators keyed by (naive, wall-clock) hour.
    keys = pd.DatetimeIndex([])
    sums = np.array([], dtype="float64")
    counts = np.array([], dtype="int64")
    all_minutes = np.array([], dtype="int64")  # every row, valid or not

    n_rows = 0
    n_missing_gap = 0
    n_all_missing = 0
    missing_per_col: dict[str, int] = {c: 0 for c in raw_cols}
    first_ts = None
    last_ts = None
    n_dup_ts = 0
    n_gaps_over_1min = 0
    max_gap_minutes = 0
    max_gap_start = None
    max_gap_end = None
    prev_last_ts = None

    reader = pd.read_csv(
        RAW_UCI,
        sep=";",
        chunksize=UCI_CHUNKSIZE,
        dtype={c: "string" for c in raw_cols},
        na_values=["?"],
        keep_default_na=True,
        engine="c",
    )

    for chunk in reader:
        n_rows += len(chunk)
        ts = pd.to_datetime(
            chunk["Date"].str.cat(chunk["Time"], sep=" "),
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )
        if ts.isna().any():
            raise ValueError("unparseable timestamp rows in UCI file")
        if first_ts is None:
            first_ts = ts.iloc[0]
        last_ts = ts.iloc[-1]

        # time-grid diagnostics (rows with '?' still carry a timestamp)
        if prev_last_ts is not None:
            n_gaps_over_1min += 0  # cross-chunk continuity handled below
        d = ts.diff().dt.total_seconds().div(60)
        d.iloc[0] = (ts.iloc[0] - prev_last_ts).total_seconds() / 60 if prev_last_ts is not None else 1
        gapmask = d > 1
        if gapmask.any():
            n_gaps_over_1min += int(gapmask.sum())
            gi = int(np.flatnonzero(gapmask.values)[np.argmax(d.values[gapmask.values])])
            g = int(d.iloc[gi])
            if g > max_gap_minutes:
                max_gap_minutes = g
                max_gap_start = ts.iloc[gi - 1]
                max_gap_end = ts.iloc[gi]
        n_dup_ts += int((d == 0).sum())
        prev_last_ts = ts.iloc[-1]

        gap = pd.to_numeric(chunk["Global_active_power"], errors="coerce")
        n_missing_gap += int(gap.isna().sum())
        for c in raw_cols:
            if c in ("Date", "Time"):
                continue
            missing_per_col[c] += int(chunk[c].isna().sum())
        n_all_missing += int(
            chunk[raw_cols[2:]].isna().all(axis=1).sum()
        )

        hour = ts.dt.floor("h")
        valid = gap.notna()

        g_valid = pd.DataFrame({"h": hour[valid], "v": gap[valid]}).groupby("h")["v"].agg(["sum", "count"])
        g_all = hour.value_counts().sort_index()

        # merge into accumulators (union with the keys already accumulated)
        new_keys = keys.union(g_valid.index).union(g_all.index).sort_values()
        sums = pd.Series(sums, index=keys).reindex(new_keys, fill_value=0.0)
        counts = pd.Series(counts, index=keys).reindex(new_keys, fill_value=0)
        allm = pd.Series(all_minutes, index=keys).reindex(new_keys, fill_value=0)
        sums = sums.add(g_valid["sum"], fill_value=0.0)
        counts = counts.add(g_valid["count"], fill_value=0)
        allm = allm.add(g_all, fill_value=0)
        keys = new_keys
        sums = sums.values
        counts = counts.values.astype("int64")
        all_minutes = allm.values.astype("int64")

    hours = pd.DataFrame(
        {
            "sum_kW": sums,
            "n_minutes": counts,
            "n_rows": all_minutes,
        },
        index=keys,
    ).sort_index()

    # Fully missing hours are dropped from the CSV (documented), partial hours kept.
    kept = hours[hours["n_minutes"] > 0].copy()
    kept["load_kW"] = kept["sum_kW"] / kept["n_minutes"]

    out = pd.DataFrame(
        {
            "timestamp": kept.index.strftime("%Y-%m-%d %H:%M:%S"),
            "load_kW": [fmt_float(v, 6) for v in kept["load_kW"]],
            "n_minutes": kept["n_minutes"].astype("int64").astype(str),
        }
    )
    write_csv(out, OUT_UCI, {})

    # ---- statistics for the report -------------------------------------
    span_hours = int((last_ts.floor("h") - first_ts.floor("h")).total_seconds() // 3600) + 1
    full_gap_hours = span_hours - len(kept)
    partial_hours = int((kept["n_minutes"] < 60).sum())
    partial_minutes_missing = int((60 - kept["n_minutes"]).where(kept["n_minutes"] < 60, 0).sum())
    span_minutes = int(round((last_ts - first_ts).total_seconds() / 60)) + 1

    # contiguity of fully-missing hours (the documented outage block(s))
    miss_hours = hours[hours["n_minutes"] == 0].index
    miss_blocks = []
    if len(miss_hours):
        run_id = (pd.Series(miss_hours).diff() != pd.Timedelta(hours=1)).cumsum().values
        s = pd.Series(miss_hours)
        for _, g in s.groupby(run_id):
            miss_blocks.append(
                {"start": str(g.iloc[0]), "end": str(g.iloc[-1]), "hours": int(len(g))}
            )
        miss_blocks.sort(key=lambda b: -b["hours"])

    hours_by_year = hours.copy()
    hours_by_year["year"] = hours_by_year.index.year
    years = []
    for y, g in hours_by_year.groupby("year"):
        years.append(
            {
                "year": int(y),
                "hours_in_span": int(len(g)),
                "hours_with_data": int((g["n_minutes"] > 0).sum()),
                "minutes_missing": int((60 - g["n_minutes"]).sum()),
            }
        )

    # DST probe: France switches to/from summer time on the last Sunday of
    # March/October.  The recorded grid has no gaps and no duplicates (checked
    # above), so the question is whether hour 02 exists on the spring-forward
    # date (it should NOT in true local civil time).  This decides whether the
    # recorded wall clock is fixed UTC+1 or true local time.
    dst_dates = [
        "2007-03-25", "2008-03-30", "2009-03-29", "2010-03-28",
        "2007-10-28", "2008-10-26", "2009-10-25", "2010-10-31",
    ]
    dst_probe = {}
    for d in dst_dates:
        day = hours[hours.index.normalize() == pd.Timestamp(d)]
        dst_probe[d] = {
            "hours_0200_minutes": int(day.loc[day.index.hour == 2, "n_rows"].sum()),
            "hours_total": int(len(day)),
        }

    stats = {
        "raw_path": str(RAW_UCI),
        "raw_bytes": RAW_UCI.stat().st_size,
        "raw_sha256": sha256_file(RAW_UCI),
        "raw_columns": raw_cols,
        "n_rows": n_rows,
        "n_duplicate_timestamps": n_dup_ts,
        "n_time_gaps_over_1min": n_gaps_over_1min,
        "max_time_gap_minutes": max_gap_minutes,
        "max_time_gap_start": str(max_gap_start),
        "max_time_gap_end": str(max_gap_end),
        "n_missing_global_active_power": n_missing_gap,
        "missing_fraction_global_active_power": n_missing_gap / n_rows,
        "n_rows_all_measures_missing": n_all_missing,
        "missing_per_column": missing_per_col,
        "first_timestamp": str(first_ts),
        "last_timestamp": str(last_ts),
        "span_minutes_naive": span_minutes,
        "span_hours": span_hours,
        "n_hours_written": int(len(kept)),
        "n_hours_fully_missing": int(full_gap_hours),
        "n_hours_partial": partial_hours,
        "n_minutes_missing_in_partial_hours": partial_minutes_missing,
        "n_minutes_total_valid": int(kept["n_minutes"].sum()),
        "first_missing_hour": str(miss_hours.min()) if len(miss_hours) else None,
        "last_missing_hour": str(miss_hours.max()) if len(miss_hours) else None,
        "missing_hour_blocks": miss_blocks,
        "per_year": years,
        "dst_probe": dst_probe,
        "output_sha256": sha256_text(OUT_UCI),
        "output_bytes": OUT_UCI.stat().st_size,
        "years": sorted({t.year for t in [first_ts, last_ts]} | {y for y in range(first_ts.year, last_ts.year + 1)}),
    }
    return stats


# --------------------------------------------------------------------------
# 2. NSRDB hourly 2020 -> local standard time (UTC+8)
# --------------------------------------------------------------------------


def build_nsrdb_hourly() -> dict:
    if not RAW_NSRDB.exists():
        raise FileNotFoundError(f"raw NSRDB file not found: {RAW_NSRDB}")

    # Lines 0-1 are the NSRDB metadata block (units row + values row);
    # line 2 is the column-name row; data start on line 3.
    df = pd.read_csv(RAW_NSRDB, skiprows=[0, 1])

    meta = pd.read_csv(RAW_NSRDB, nrows=2, header=None)
    meta_hdr = meta.iloc[0].tolist()
    meta_val = meta.iloc[1].tolist()
    meta_map = dict(zip(meta_hdr, meta_val))

    required = [
        "Year", "Month", "Day", "Hour", "Minute",
        "DNI", "DHI", "GHI", "Temperature", "Wind Speed",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"NSRDB missing expected columns: {missing_cols}")

    if int(df["Minute"].abs().sum()) != 0:
        raise ValueError("NSRDB Minute column is not identically zero")

    # NSRDB reports data in GMT (metadata 'Time Zone' = 0) for a site whose
    # local standard time is UTC+8 (metadata 'Local Time Zone' = 8).  China has
    # no daylight saving time, so local standard time = UTC + 8 h.
    utc = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=df["Day"], hour=df["Hour"]),
        utc=True,
    )
    local = utc.dt.tz_convert("Etc/GMT-8").dt.tz_localize(None)

    out = pd.DataFrame(
        {
            "timestamp": local.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "dni_wm2": [fmt_float(v, 1) for v in df["DNI"].astype(float)],
            "dhi_wm2": [fmt_float(v, 1) for v in df["DHI"].astype(float)],
            "ghi_wm2": [fmt_float(v, 1) for v in df["GHI"].astype(float)],
            "temp_c": [fmt_float(v, 6) for v in df["Temperature"].astype(float)],
            "wind_ms": [fmt_float(v, 6) for v in df["Wind Speed"].astype(float)],
        }
    )
    if not out["timestamp"].is_monotonic_increasing or out["timestamp"].duplicated().any():
        raise ValueError("NSRDB local timestamps are not strictly increasing")
    write_csv(out, OUT_NSRDB, {})

    # gaps in the local timestamp grid (e.g. Feb 29 absent from source)
    tv = local.values
    diffs = pd.Series(tv).diff().dropna()
    grid_gaps = pd.Series(diffs[diffs > np.timedelta64(1, "h")])
    gap_desc = [
        {
            "after": str(pd.Timestamp(tv[i - 1])),
            "next": str(pd.Timestamp(tv[i])),
            "missing_hours": int((tv[i] - tv[i - 1]) / np.timedelta64(1, "h")) - 1,
        }
        for i in np.flatnonzero(diffs.values > np.timedelta64(1, "h")) + 1
    ]

    def _num(x):
        try:
            return float(x)
        except Exception:
            return str(x)

    stats = {
        "raw_path": str(RAW_NSRDB),
        "raw_bytes": RAW_NSRDB.stat().st_size,
        "raw_sha256": sha256_file(RAW_NSRDB),
        "meta": {k: _num(v) for k, v in meta_map.items()},
        "n_rows": int(len(df)),
        "n_unique_dates": int(df[["Year", "Month", "Day"]].drop_duplicates().shape[0]),
        "years": sorted(df["Year"].unique().tolist()),
        "months_present": sorted(df["Month"].unique().tolist()),
        "feb29_present": bool(((df["Month"] == 2) & (df["Day"] == 29)).any()),
        "first_utc": str(utc.iloc[0]),
        "last_utc": str(utc.iloc[-1]),
        "first_local": str(local.iloc[0]),
        "last_local": str(local.iloc[-1]),
        "local_grid_gaps": gap_desc,
        "n_negative_temperature": int((df["Temperature"] < 0).sum()),
        "n_missing_any_required": int(df[required].isna().any(axis=1).sum()),
        "n_sentinel_minus9999": int((df[["DNI", "DHI", "GHI", "Temperature", "Wind Speed"]] == -9999).any(axis=1).sum()),
        "value_ranges": {
            c: {
                "min": float(df[c].min()),
                "max": float(df[c].max()),
                "mean": float(df[c].mean()),
            }
            for c in ["DNI", "DHI", "GHI", "Temperature", "Wind Speed"]
        },
        "wind_speed_note": "reported m/s in source metadata; values are implausibly low (max 1.1 m/s)",
        "output_sha256": sha256_text(OUT_NSRDB),
        "output_bytes": OUT_NSRDB.stat().st_size,
    }
    return stats


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("building external validation datasets")
    print("=" * 72)

    uci = build_uci_hourly()
    nsrdb = build_nsrdb_hourly()

    summary = {"generated_by": "code/build_external_data.py", "uci": uci, "nsrdb": nsrdb}

    # print a compact machine-readable summary (also useful for the BUILD_REPORT)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    # persist the summary next to the data so the report numbers are traceable
    summary_path = OUT_DIR / "build_summary.json"
    with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")

    print("\nwrote:")
    for p in (OUT_UCI, OUT_NSRDB, summary_path):
        print(f"  {p}  ({p.stat().st_size} bytes, sha256={sha256_file(p)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
