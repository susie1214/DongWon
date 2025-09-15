#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, pandas as pd, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="submission csv list")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    dfs = [pd.read_csv(p) for p in args.inputs]
    base = dfs[0][["product_name"]].copy()
    month_cols = [c for c in dfs[0].columns if c.startswith("months_since_launch_")]

    arrs = [df[month_cols].astype(float).values for df in dfs]
    mean_vals = np.mean(arrs, axis=0)
    out = base.copy()
    for i,c in enumerate(month_cols):
        out[c] = mean_vals[:, i]
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] ensemble saved: {args.out_csv}")

if __name__ == "__main__":
    main()
