# -*- coding: utf-8 -*-
"""
autosuggest_scales.py

Heuristically proposes sku_scale.json and month_scale.json updates
based on a current submission CSV (and optional external/history data).

USAGE:
  python autosuggest_scales.py --submission C:\Won\submission_persona_agent.csv --root C:\Won

Outputs:
  C:\Won\sku_scale_suggested.json
  C:\Won\month_scale_suggested.json
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

MONTHS = [str(p) for p in pd.period_range("2024-07","2025-06",freq="M")]

def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # detect month columns
    mcols = [c for c in df.columns if c!="product_name"]
    if len(mcols) != 12 or not any([str(c).startswith("20") for c in mcols]):
        # rename to YYYY-MM if needed
        mcols = MONTHS
        # assume current file is already YYYY-MM or reorder externally; minimal check here
    return df

def suggest_sku_scale(sub_df: pd.DataFrame, lo=0.75, hi=1.25):
    # Total per product
    mcols = [c for c in sub_df.columns if c!="product_name"]
    totals = sub_df[mcols].sum(axis=1)
    mean, std = totals.mean(), totals.std(ddof=1) if len(totals)>1 else 0.0
    scales = {}
    for pn, tot in zip(sub_df["product_name"], totals):
        if std == 0:
            s = 1.0
        else:
            z = (tot - mean)/std
            if z > 1.0:
                s = 1.0 - 0.1*min(2.0, z)   # downscale if too high
            elif z < -1.0:
                s = 1.0 + 0.1*min(2.0, -z)  # upscale if too low
            else:
                s = 1.0
        scales[pn] = float(np.clip(s, lo, hi))
    return scales

def suggest_month_scale(sub_df: pd.DataFrame, lo=0.8, hi=1.2):
    # Aggregate per-month across products
    mcols = [c for c in sub_df.columns if c!="product_name"]
    msum = sub_df[mcols].sum(axis=0)
    # Normalize to mean 1.0
    meanv = msum.mean() if msum.mean()!=0 else 1.0
    raw = (msum / meanv).values.astype(float)
    # smooth
    smoothed = []
    for i, v in enumerate(raw):
        prevv = raw[i-1] if i>0 else raw[i]
        nextv = raw[i+1] if i<len(raw)-1 else raw[i]
        sv = 0.25*prevv + 0.5*v + 0.25*nextv
        smoothed.append(float(np.clip(sv, lo, hi)))
    # map to MONTHS
    return {MONTHS[i]: smoothed[i] for i in range(min(12,len(MONTHS)))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=str, required=True)
    ap.add_argument("--root", type=str, default=r"C:\Won")
    args = ap.parse_args()
    root = Path(args.root)
    sub = load_submission(Path(args.submission))

    sku_scale = suggest_sku_scale(sub)
    month_scale = suggest_month_scale(sub)

    (root/"sku_scale_suggested.json").write_text(json.dumps(sku_scale, ensure_ascii=False, indent=2), encoding="utf-8")
    (root/"month_scale_suggested.json").write_text(json.dumps({
        "meta":{"months": MONTHS},
        "global": [month_scale[m] for m in MONTHS]
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Wrote:", root/"sku_scale_suggested.json")
    print("[OK] Wrote:", root/"month_scale_suggested.json")

if __name__ == "__main__":
    main()
