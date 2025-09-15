# -*- coding: utf-8 -*-
"""
generate_templates_from_submission.py

Reads user's sample_submission.csv + product_info1.csv
and produces data templates required by the persona-forecast pipeline:

- promo_calendar.csv              (product_name, month, discount_rate, display, coupon, campaign_name)
- external_monthly.csv            (month, tavg, feels_like, precip_mm, search_index, holiday_dummy, online_share, overseas_market_index, pet_household_share)
- macro_caps.csv                  (month, cap_total[, cap_{category} ...])
- month_scale.json                (category default curves + per-product overrides; months 2024-07..2025-06)
- sku_scale.json                  (per-product scalar = 1.0 by default)
- category_stats.csv              (category-level biases; optional; can be computed from history if available)

USAGE (Windows):
  python generate_templates_from_submission.py --root C:\Won
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd
import numpy as np

MONTHS = [str(p) for p in pd.period_range("2024-07","2025-06",freq="M")]

def load_inputs(root: Path):
    sub = pd.read_csv(root/"sample_submission.csv")
    prod = pd.read_csv(root/"product_info1.csv")
    return sub, prod

def get_product_list_and_months(sub: pd.DataFrame):
    # Try to infer month columns, else fallback to 12 months window
    month_cols = [c for c in sub.columns if c not in ["product_name","product_id","id"]]
    if not month_cols or len(month_cols) != 12:
        months = MONTHS
    else:
        # Normalize to YYYY-MM if the template used month1..month12-like headers
        if not any([str(c).startswith("20") for c in month_cols]):
            months = MONTHS
        else:
            months = [str(c) for c in month_cols]
    products = sub["product_name"].dropna().unique().tolist()
    return products, months

def write_promo_calendar(root: Path, products, months):
    rows = []
    for pn in products:
        for m in months:
            rows.append({
                "product_name": pn,
                "month": m,
                "discount_rate": 0.0,  # 0~1
                "display": 0,          # 0/1
                "coupon": 0,           # 0/1
                "campaign_name": ""
            })
    df = pd.DataFrame(rows)
    df.to_csv(root/"promo_calendar.csv", index=False, encoding="utf-8-sig")

def write_external_monthly(root: Path, months):
    # Extended with online_share, overseas_market_index, pet_household_share
    rows = []
    for m in months:
        rows.append({
            "month": m,
            "tavg": "", "feels_like": "", "precip_mm": "",
            "search_index": "", "holiday_dummy": "",
            "online_share": "",              # (0~1) share of online channel sales
            "overseas_market_index": "",     # (0~100) custom index you define via RAG
            "pet_household_share": ""        # (0~1) estimated share of pet-owning households
        })
    pd.DataFrame(rows).to_csv(root/"external_monthly.csv", index=False, encoding="utf-8-sig")

def seasonal_curve(name: str, months: list):
    # Simple canned seasonalities for demonstration
    import math
    out = []
    for m in months:
        mm = int(str(m)[5:7])
        if "음료" in name or "beverage" in name.lower():
            val = 1.0 + 0.20*math.sin((mm-6)/12*2*math.pi)   # summer peak
        elif "요거트" in name or "yogurt" in name.lower():
            val = 1.0 + 0.15*math.sin((mm-2)/12*2*math.pi)   # spring/early-summer
        else:
            val = 1.0
        out.append(round(max(0.75, val), 3))
    # Normalize to mean ~1.0
    meanv = sum(out)/len(out)
    out = [round(v/meanv,3) for v in out]
    return out

def write_month_scale(root: Path, prod: pd.DataFrame, months):
    cats = prod["category"].fillna("UNKNOWN").unique().tolist()
    category_curves = {}
    for c in cats:
        category_curves[c] = seasonal_curve(str(c), months)

    data = {
        "meta": {"months": months, "note": "category default + product_overrides"},
        "category": category_curves,
        "product_overrides": {}   # you can fill later per product if needed
    }
    with open(root/"month_scale.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_sku_scale(root: Path, products):
    data = {pn: 1.0 for pn in products}
    with open(root/"sku_scale.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_macro_caps(root: Path, months, prod: pd.DataFrame):
    rows = [{"month": m, "cap_total": ""} for m in months]
    # Optional: add per-category columns
    cats = prod["category"].fillna("UNKNOWN").unique().tolist()
    df = pd.DataFrame(rows)
    for c in cats:
        df[f"cap_{c}"] = ""   # fill if you have category-level caps
    df.to_csv(root/"macro_caps.csv", index=False, encoding="utf-8-sig")

def write_category_stats(root: Path, prod: pd.DataFrame):
    # Placeholder columns; You should compute from history (price elasticity, promo lift, etc.)
    # If you have history_monthly.csv, you can compute these programmatically (separate script).
    cats = prod["category"].fillna("UNKNOWN").unique().tolist()
    rows = []
    for c in cats:
        rows.append({
            "category": c,
            "bias_health": 0.5,
            "bias_price": 0.5,
            "bias_convenience": 0.5,
            "bias_young": 0.5
        })
    pd.DataFrame(rows).to_csv(root/"category_stats.csv", index=False, encoding="utf-8-sig")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=r"C:\Won", help="Folder containing sample_submission.csv & product_info1.csv")
    args = ap.parse_args()
    root = Path(args.root)

    sub, prod = load_inputs(root)
    products, months = get_product_list_and_months(sub)

    write_promo_calendar(root, products, months)
    write_external_monthly(root, months)
    write_macro_caps(root, months, prod)
    write_month_scale(root, prod, months)
    write_sku_scale(root, products)
    write_category_stats(root, prod)

    print("[OK] Generated: promo_calendar.csv, external_monthly.csv, macro_caps.csv, month_scale.json, sku_scale.json, category_stats.csv")

if __name__ == "__main__":
    main()
