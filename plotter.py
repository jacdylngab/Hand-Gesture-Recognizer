#!/usr/bin/env python3
"""
Plot selected IMU time series from a CSV.
- Assumes first column is time (ms since start OR ISO datetime OR seconds)
- Lets you interactively pick any subset of columns to plot (e.g., ax, ay, az, gx, gy, gz)
- Shows the plot and saves a PNG next to the CSV
"""

import sys
import os
import math
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise SystemExit(f"Failed to read CSV: {e}")
    if df.shape[1] < 2:
        raise SystemExit("CSV must have at least 2 columns (time + 1 signal).")
    return df

def parse_time_col(s: pd.Series):
    """
    Convert the first column into a pandas datetime-like index and a display label.
    Handles:
      - integer/float seconds
      - integer/float milliseconds
      - ISO timestamp strings
    Returns time_seconds (float series), xlabel (str)
    """
    x = s.copy()
    name = s.name if s.name is not None else "time"
    # Try numeric first
    if pd.api.types.is_numeric_dtype(x):
        # Heuristic: if values look like ms (median > 1e6), convert to seconds
        med = float(x.median())
        if med > 1e6:
            return (x.astype(float) / 1000.0, f"{name} (s, from ms)")
        # Already seconds
        return (x.astype(float), f"{name} (s)")
    # Try parse as datetime
    try:
        dt = pd.to_datetime(x, utc=False, errors="raise")
        # use seconds relative to start
        t0 = dt.iloc[0]
        ts = (dt - t0).dt.total_seconds()
        return (ts, f"{name} (s since {t0})")
    except Exception:
        # Last resort: try to coerce to float
        try:
            xf = x.astype(float)
            med = float(xf.median())
            if med > 1e6:
                return (xf / 1000.0, f"{name} (s, from ms)")
            return (xf, f"{name} (s)")
        except Exception:
            raise SystemExit("Could not interpret the first column as time (seconds/ms or timestamp).")

def pick_columns_interactively(df: pd.DataFrame):
    """
    Show available columns (excluding time col) and ask user to pick a subset.
    Accepts comma-separated names OR indices, or 'all' for everything.
    """
    time_col = df.columns[0]
    candidates = list(df.columns[1:])
    if not candidates:
        raise SystemExit("No data columns found beyond the time column.")
    print("\nAvailable columns to plot (choose any combination):")
    for i, c in enumerate(candidates):
        print(f"  [{i}] {c}")
    print("\nType: a comma-separated list of indices or names")
    print("  examples: 0,1,2   or   ax,ay,az   or   all")
    raw = input("Your selection: ").strip()
    if raw.lower() in ("all", "a", "*"):
        return candidates

    # Parse indices and/or names
    chosen = []
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    for t in tokens:
        if t.isdigit():
            idx = int(t)
            if 0 <= idx < len(candidates):
                chosen.append(candidates[idx])
        else:
            # match by exact name first, else case-insensitive
            if t in candidates:
                chosen.append(t)
            else:
                matches = [c for c in candidates if c.lower() == t.lower()]
                if matches:
                    chosen.append(matches[0])
                else:
                    print(f"  [warn] Skipping unknown column: {t}")
    # remove duplicates, preserve order
    seen = set()
    dedup = []
    for c in chosen:
        if c not in seen:
            dedup.append(c); seen.add(c)
    if not dedup:
        raise SystemExit("No valid columns selected.")
    return dedup

def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("csv", nargs="?", help="Path to CSV (first column = time)")
    args = ap.parse_args()

    if not args.csv:
        try:
            args.csv = input("Path to CSV: ").strip().strip('"')
        except EOFError:
            pass
    if not args.csv:
        raise SystemExit("No file provided.")
    if not os.path.isfile(args.csv):
        raise SystemExit(f"File not found: {args.csv}")

    df = load_csv(args.csv)

    # Parse time
    time_s, xlabel = parse_time_col(df.iloc[:, 0])
    df_time = df.copy()
    df_time.insert(0, "_tsecs_", time_s)
    df_time.drop(columns=[df.columns[0]], inplace=True)

    # Let user pick columns
    cols = pick_columns_interactively(df)
    y = df[cols].apply(pd.to_numeric, errors="coerce")
    if y.isnull().all().all():
        raise SystemExit("Selected columns contain no numeric data.")
    # Drop rows with NaNs in selected columns
    mask = ~y.isnull().any(axis=1)
    t = time_s[mask]
    y = y[mask]

    # Plot
    plt.figure(figsize=(10, 5))
    for c in cols:
        plt.plot(t, y[c], label=c)  # no explicit colors

    plt.title(os.path.basename(args.csv))
    plt.xlabel(xlabel)
    plt.ylabel("value")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(loc="best")

    # Save PNG alongside CSV
    png_path = os.path.splitext(args.csv)[0] + "_plot.png"
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(png_path, dpi=150)
    print(f"\nSaved figure: {png_path}")
    plt.show()

if __name__ == "__main__":
    main()
