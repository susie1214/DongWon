# -*- coding: utf-8 -*-
"""
dw_persona_forecast_agent.py
- 입력:
  C:\Won\product_info1.csv
  C:\Won\sample_submission.csv
  (옵션) C:\Won\promo_calendar.csv, C:\Won\external_monthly.csv, C:\Won\macro_caps.csv, C:\Won\history_monthly.csv
- 출력:
  C:\Won\submission_persona_agent.csv

기능 요약:
1) 제품별 LLM 싱글턴 프롬프트 생성(옵션) 혹은 규칙기반 페르소나 생성(최소 10개 속성 + 가중치)
2) 제품군별 페르소나 가중치 분해, 월별 시즈널리티(P_k,month) 반영
3) 프로모션 비선형 반응(할인율/진열/쿠폰) 곡선 반영
4) 외부변수(날씨/검색/휴일) 훅: 있으면 자동 merge
5) 앙상블:
   - (기본) 시뮬 3시나리오(Base/Promo/Conservative) 블렌딩
   - (옵션) history_monthly.csv 존재 시 LGBM/XGB/CB 스태킹
6) 미시-거시 리컨실리에이션: macro_caps.csv or 카테고리 비중 목표가 있으면 스케일 조정
7) sample_submission 포맷 그대로 예측 채워 저장
"""

import os, json, math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# ===== 사용자 설정 =====
ROOT = Path(r"C:\Won")
PRODUCT_CSV = ROOT / "product_info1.csv"
SUBMISSION_CSV = ROOT / "sample_submission.csv"

PROMO_CSV = ROOT / "promo_calendar.csv"         # [옵션] product_name, month(YYYY-MM), discount_rate, display, coupon
EXTERNAL_CSV = ROOT / "external_monthly.csv"    # [옵션] month(YYYY-MM), tavg, feels_like, precip_mm, search_index, holiday_dummy
MACRO_CAPS_CSV = ROOT / "macro_caps.csv"        # [옵션] month(YYYY-MM), cap_total or per-category caps
HISTORY_CSV = ROOT / "history_monthly.csv"      # [옵션] month(YYYY-MM), product_name, qty (실측)

# LLM 실제 호출 사용 여부 (기본 False: 규칙기반/stub)
USE_LLM = True

# 예측 기간
START_YM = "2024-07"
END_YM   = "2025-06"

# 랜덤 시드 (재현성)
SEED = 42
np.random.seed(SEED)

# ===== 유틸 =====
def month_range(start_ym: str, end_ym: str) -> List[str]:
    s = pd.Period(start_ym, freq="M")
    e = pd.Period(end_ym,   freq="M")
    return [str(p) for p in pd.period_range(s, e, freq="M")]

MONTHS = month_range(START_YM, END_YM)

def safe_read_csv(path: Path, **kw) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, **kw)
    return pd.DataFrame()

def ensure_columns(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def clip_nonneg(x):
    return max(0.0, float(x))

# ===== 1) 데이터 로드 =====
prod = safe_read_csv(PRODUCT_CSV)
sub  = safe_read_csv(SUBMISSION_CSV)

if prod.empty or sub.empty:
    raise FileNotFoundError("필수 입력 파일이 없습니다. product_info1.csv, sample_submission.csv를 확인하세요.")

# product_info1.csv 예상 컬럼: product_name, category, base_price, pack_size, ... (없는 컬럼은 자동 보정)
prod = ensure_columns(prod, ["product_name","category","base_price","pack_size"])
prod["category"]   = prod["category"].fillna("UNKNOWN")
prod["base_price"] = prod["base_price"].fillna(prod["base_price"].median() if prod["base_price"].notna().any() else 5000)

# sample_submission.csv: product_name + 12개 월 컬럼(혹은 별도 포맷)
# 내부 처리를 위해 long 포맷으로 변환
def melt_submission_template(sub_df: pd.DataFrame) -> pd.DataFrame:
    # month 컬럼 추정
    month_cols = [c for c in sub_df.columns if c not in ["product_name","product_id","id"]]
    # 월 명칭이 없으면 MONTHS로 강제 부여
    if not month_cols:
        month_cols = MONTHS
        # 빈 템플릿 생성
        out = []
        for pn in sub_df["product_name"].unique():
            for m in MONTHS:
                out.append({"product_name": pn, "month": m, "pred": 0.0})
        return pd.DataFrame(out)
    long = sub_df.melt(id_vars=[c for c in sub_df.columns if c not in month_cols],
                       value_vars=month_cols, var_name="month", value_name="pred")
    # month 값이 YYYY-MM 형태가 아니면 재매핑 시도
    if not long["month"].astype(str).str.contains(r"\d{4}-\d{2}", regex=True).any():
        # month1..12 같은 경우 START_YM~END_YM 매핑
        month_map = {mc: MONTHS[i] for i,mc in enumerate(month_cols) if i < len(MONTHS)}
        long["month"] = long["month"].map(month_map).fillna(MONTHS[0])
    long["pred"] = 0.0
    return long

sub_long = melt_submission_template(sub)

# ===== 2) LLM 싱글턴 프롬프트 (문자열만 생성) & 규칙기반 Stub 페르소나 =====
def build_single_turn_prompt_for_products(df_products: pd.DataFrame) -> str:
    """
    제품 단위 최소로, 한 번에 여러 제품을 넣어도 '싱글 턴' 제약을 지킵니다.
    - 응답 포맷: JSON (제품별 persona 리스트, 각 10+ 속성, 가중치, 구매확률%, 월별 패턴[12])
    """
    items = []
    for _, r in df_products.iterrows():
        items.append({
            "product_name": r["product_name"],
            "category": r["category"],
            "base_price": float(r["base_price"])
        })
    payload = {
        "objective": "신제품 출시 후 12개월(2024-07~2025-06) 월별 수요 예측을 위한 페르소나 생성",
        "requirements": [
            "각 제품별로 6~10개 페르소나",
            "각 페르소나는 최소 10개 이상의 속성(연령, 성별, 소득, 지역, 채널 등)과 속성별 가중치 포함",
            "각 페르소나는 구매확률(%)과 월별 구매빈도 패턴(12개 길이, 1.0=중립 스케일) 포함",
            "JSON 포맷으로만 응답(주석/설명 없음)"
        ],
        "products": items,
        "months": MONTHS
    }
    # 실제로는 이 문자열을 LLM에 단 한 번 전달합니다.
    return json.dumps(payload, ensure_ascii=False, indent=2)

def stub_generate_personas_for_product(row: pd.Series) -> List[Dict]:
    """
    규칙기반 stub: 카테고리/가격을 기반으로 6~8개의 페르소나를 생성
    - 각 페르소나: 10개 이상의 속성 + 가중치, 구매확률(%), 월별 패턴(길이 12)
    """
    category = str(row["category"]).lower()
    base_price = float(row["base_price"])
    # 카테고리별 선호/속성 비중 가이드
    cat_bias = {
        "요거트": dict(health=0.8, convenience=0.6, family=0.4, young=0.5),
        "음료": dict(refresh=0.7, price=0.6, convenience=0.5, young=0.6),
        "UNKNOWN": dict(health=0.5, price=0.5, convenience=0.5, family=0.5, young=0.5)
    }
    bias = cat_bias.get(category, cat_bias["UNKNOWN"])

    # 6~8명
    k = np.random.choice([6,7,8], 1)[0]
    personas = []
    for i in range(k):
        # 속성 10개 이상 (예시)
        attrs = {
            "age_band": np.random.choice(["15-24","25-34","35-44","45-54","55+"]),
            "gender": np.random.choice(["F","M"]),
            "income_band": np.random.choice(["low","mid","high"], p=[0.3,0.5,0.2]),
            "region_kr": np.random.choice(["Seoul","Metro","Provincial"]),
            "channel": np.random.choice(["ONLINE","OFFLINE","MIX"]),
            "price_sensitivity": round(np.clip(np.random.normal(bias.get("price",0.5), 0.15), 0, 1), 3),
            "promo_sensitivity": round(np.clip(np.random.normal(0.6, 0.2), 0, 1), 3),
            "brand_loyalty":      round(np.clip(np.random.normal(0.5, 0.2), 0, 1), 3),
            "innovation_seek":    round(np.clip(np.random.normal(0.5 + 0.2*(category=="요거트"), 0.2), 0, 1), 3),
            "family_size": np.random.choice(["single","couple","kids"]),
            "health_pref":        round(np.clip(np.random.normal(bias.get("health",0.5), 0.2), 0, 1), 3),
            "convenience_pref":   round(np.clip(np.random.normal(bias.get("convenience",0.5), 0.2), 0, 1), 3),
            "youthful_pref":      round(np.clip(np.random.normal(bias.get("young",0.5), 0.2), 0, 1), 3),
            "eco_pref":           round(np.clip(np.random.normal(0.4, 0.2), 0, 1), 3),
        }
        # 속성 가중치(합=1) 샘플
        attr_keys = list(attrs.keys())
        raw_w = np.abs(np.random.normal(1.0, 0.5, size=len(attr_keys)))
        weights = (raw_w / raw_w.sum()).round(4)
        attr_weights = dict(zip(attr_keys, weights))

        # 구매확률(%) – 가격/건강/편의 영향
        base_prob = 15 + 10*(1-attrs["price_sensitivity"]) + 10*attrs["convenience_pref"] + 10*attrs["health_pref"]
        price_adj = -0.0008 * max(0, base_price - 3000)  # 가격 높을수록 감소
        purchase_prob = float(np.clip(base_prob + price_adj + np.random.normal(0, 3), 1, 90))

        # 월별 패턴 (12) – 여름↑(음료), 새해다짐/봄↑(요거트) 등
        pattern = []
        for m in MONTHS:
            mm = int(m.split("-")[1])
            if "음료" in category:
                s = 1.0 + 0.25*np.sin((mm-6)/12*2*np.pi)  # 여름 피크
            elif "요거트" in category:
                s = 1.0 + 0.18*np.sin((mm-2)/12*2*np.pi)  # 봄/초여름
            else:
                s = 1.0
            pattern.append(round(float(max(0.7, s + np.random.normal(0,0.03))), 3))

        personas.append({
            "id": f"P{i+1:02d}",
            "share_weight": round(float(np.random.dirichlet(np.ones(1))[0]), 4),  # 임시, 나중에 정규화
            "attributes": attrs,
            "attr_weights": attr_weights,
            "purchase_prob_pct": purchase_prob,
            "monthly_pattern": pattern
        })

    # share_weight 정규화
    sw = np.array([p["share_weight"] for p in personas])
    sw = sw / sw.sum()
    for i, p in enumerate(personas):
        p["share_weight"] = round(float(sw[i]), 4)
    return personas

# ===== 3) 프로모션 비선형 반응 함수 =====
def promo_uplift(discount_rate: float, display: int, coupon: int) -> float:
    """
    할인: 포화(tanh) 형태 탄력, 진열/쿠폰: 가산 후 포화
    """
    d = max(0.0, float(discount_rate))
    uplift_disc = 1.0 + 0.8 * math.tanh(3.0 * d)        # 0~80% 범위 내 포화
    uplift_disp = 1.0 + (0.12 if int(display)==1 else 0.0)
    uplift_coup = 1.0 + (0.10 if int(coupon)==1  else 0.0)
    return uplift_disc * uplift_disp * uplift_coup

# ===== 4) 외부변수 머지 =====
promo_df    = safe_read_csv(PROMO_CSV)
external_df = safe_read_csv(EXTERNAL_CSV)
macro_caps  = safe_read_csv(MACRO_CAPS_CSV)
history_df  = safe_read_csv(HISTORY_CSV)

if not promo_df.empty:
    promo_df = ensure_columns(promo_df, ["product_name","month","discount_rate","display","coupon"])
    promo_df["month"] = promo_df["month"].astype(str)

if not external_df.empty:
    external_df = ensure_columns(external_df, ["month","tavg","feels_like","precip_mm","search_index","holiday_dummy"])
    external_df["month"] = external_df["month"].astype(str)

if not macro_caps.empty:
    macro_caps = ensure_columns(macro_caps, ["month","cap_total"])
    macro_caps["month"] = macro_caps["month"].astype(str)

# ===== 5) 제품별 페르소나 생성 (LLM 싱글턴 프롬프트 or stub) =====
if USE_LLM:
    # 실제 LLM 호출은 환경에 맞게 구현 (여기선 프롬프트 생성까지만)
    single_turn_prompt = build_single_turn_prompt_for_products(prod)
    print("\n[LLM SINGLE-TURN PROMPT]\n", single_turn_prompt)
    # TODO: 여기에 OpenAI 등 API 호출 → JSON 파싱 → product_personas dict 구성
    # 여기서는 규칙기반 stub로 대체
    pass

# 규칙기반 stub로 제품별 페르소나 생성
product_personas: Dict[str, List[Dict]] = {}
for _, r in prod.iterrows():
    product_personas[r["product_name"]] = stub_generate_personas_for_product(r)

# ===== 6) 시뮬레이션 기반 월별 수요 산출 =====
def base_monthly_units(product_row: pd.Series) -> float:
    """
    제품 기본 월간 규모(베이스라인). 가격/팩사이즈 간단 반영.
    - 실제론 TAM/가격탄력/카테고리별 평균 등을 넣으면 좋음.
    """
    price = float(product_row["base_price"])
    # 가격이 낮을수록 기본 수요 높게 (임의식)
    return max(50.0, 3000.0 / max(1000.0, price))

def apply_external_multiplier(row: pd.Series, month: str) -> float:
    """
    외부변수 멀티플라이어. external_monthly.csv가 있으면 반영, 없으면 1.0
    """
    if external_df.empty:
        return 1.0
    ex = external_df[external_df["month"]==month]
    if ex.empty:
        return 1.0
    ex = ex.iloc[0]
    # 날씨/검색/휴일의 간단한 영향 (스케일은 안정적으로 작게)
    mul = 1.0
    if not pd.isna(ex.get("tavg")):
        mul *= 1.0 + 0.01 * ((float(ex["tavg"]) - 15.0)/10.0)  # 따뜻할수록 소폭+
    if not pd.isna(ex.get("search_index")):
        mul *= 1.0 + 0.02 * (float(ex["search_index"])/100.0)
    if not pd.isna(ex.get("holiday_dummy")):
        mul *= 1.0 + 0.03 * (1.0 if int(ex["holiday_dummy"])==1 else 0.0)
    return max(0.7, float(mul))

def scenario_multiplier(scn: str) -> float:
    return {"BASE":1.0, "PROMO_HEAVY":1.12, "CONSERV":0.92}.get(scn, 1.0)

def simulate_monthly(product_row: pd.Series, month: str, scenario: str) -> float:
    """
    페르소나 혼합 + 월별 패턴 + 프로모션 + 외부변수 → 월 예상 수량
    """
    pn = product_row["product_name"]
    personas = product_personas.get(pn, [])
    if not personas:
        return 0.0

    # 프로모션
    if not promo_df.empty:
        pr = promo_df[(promo_df["product_name"]==pn) & (promo_df["month"]==month)]
        if not pr.empty:
            pr = pr.iloc[0]
            p_mul = promo_uplift(pr.get("discount_rate",0.0), pr.get("display",0), pr.get("coupon",0))
        else:
            p_mul = 1.0
    else:
        p_mul = 1.0

    ext_mul = apply_external_multiplier(product_row, month)
    scn_mul = scenario_multiplier(scenario)

    base_units = base_monthly_units(product_row)  # 제품 베이스라인
    # 페르소나 혼합
    total = 0.0
    for p in personas:
        sw = float(p["share_weight"])
        prob = float(p["purchase_prob_pct"])/100.0
        # 월 인덱스
        try:
            idx = MONTHS.index(month)
            patt = float(p["monthly_pattern"][idx])
        except:
            patt = 1.0
        total += base_units * sw * (0.5 + 1.5*prob) * patt

    total *= p_mul * ext_mul * scn_mul
    return max(0.0, float(total))

def predict_all_scenarios(sub_long_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    out = sub_long_df.copy()
    out["pred"] = 0.0
    # 조인을 위해 메타 붙이기
    merged = out.merge(prod[["product_name","category","base_price","pack_size"]], on="product_name", how="left")
    preds = []
    for _, row in merged.iterrows():
        y = simulate_monthly(row, str(row["month"]), scenario)
        preds.append(y)
    out["pred"] = preds
    out["scenario"] = scenario
    return out

base_pred   = predict_all_scenarios(sub_long, "BASE")
promo_pred  = predict_all_scenarios(sub_long, "PROMO_HEAVY")
cons_pred   = predict_all_scenarios(sub_long, "CONSERV")

# 간단 블렌딩(스태킹 대용): 가중 평균
blend = base_pred.copy()
blend["pred"] = 0.2*cons_pred["pred"].values + 0.6*promo_pred["pred"].values + 0.2*base_pred["pred"].values
blend["scenario"] = "BLEND_SIM"

# ===== 7) (옵션) 감독학습 앙상블: history_monthly.csv가 있으면 스태킹 =====
def try_supervised_stack(current_pred: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return current_pred
    try:
        from lightgbm import LGBMRegressor
        import xgboost as xgb
        from catboost import CatBoostRegressor
    except Exception:
        # 라이브러리 없으면 시뮬 블렌드 유지
        return current_pred

    # 특징 구성 (단순 예시): price, scenario별 예측치, month 더미
    feat = current_pred.merge(prod[["product_name","base_price","category"]], on="product_name", how="left")
    # month to numeric features
    feat["month_num"] = feat["month"].str.slice(5,7).astype(int)
    # scenario별 wide pivot
    wide = pd.pivot_table(feat, index=["product_name","month"], columns="scenario", values="pred").reset_index()
    wide = wide.merge(prod[["product_name","base_price","category"]], on="product_name", how="left")
    wide["month_num"] = wide["month"].str.slice(5,7).astype(int)
    # 라벨 merge
    hist = history_df.copy()
    hist["month"] = hist["month"].astype(str)
    wide = wide.merge(hist.rename(columns={"qty":"y"}), on=["product_name","month"], how="left")

    # 훈련/예측 분리: 과거 존재 구간만 학습
    train = wide.dropna(subset=["y"]).copy()
    if train.empty or train["y"].sum() <= 0:
        return current_pred

    X_cols = [c for c in ["BASE","PROMO_HEAVY","CONSERV","BLEND_SIM","base_price","month_num"] if c in wide.columns]
    # 범주 인코딩(간단)
    for col in ["category"]:
        if col in wide.columns:
            cats = {k:i for i,k in enumerate(sorted(wide[col].dropna().unique()))}
            wide[col+"_id"] = wide[col].map(cats)
            if col+"_id" in train.columns:
                X_cols.append(col+"_id")

    X_tr = train[X_cols].values
    y_tr = train["y"].values

    # 세 모델 학습 후 평균
    models = []
    try:
        models.append(LGBMRegressor(n_estimators=600, learning_rate=0.05, subsample=0.9))
    except: pass
    try:
        models.append(xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, subsample=0.9, max_depth=6, tree_method="hist"))
    except: pass
    try:
        models.append(CatBoostRegressor(iterations=800, depth=6, learning_rate=0.05, verbose=False))
    except: pass
    if not models:
        return current_pred

    for m in models:
        m.fit(X_tr, y_tr)

    # 전체에 대해 예측치 재계산 → supervised_blend
    X_all = wide[X_cols].values
    preds = np.mean([m.predict(X_all) for m in models], axis=0)
    wide["supervised_blend"] = preds

    # 최신 예측으로 갱신
    df_new = current_pred.merge(wide[["product_name","month","supervised_blend"]], on=["product_name","month"], how="left")
    df_new["pred"] = df_new["supervised_blend"].fillna(df_new["pred"])
    df_new.drop(columns=["supervised_blend"], inplace=True)
    df_new["scenario"] = "SUPERVISED_BLEND"
    return df_new

final_pred = try_supervised_stack(blend)

# ===== 8) 미시-거시 리컨실리에이션 (총량/카테고리 제약) =====
def reconcile_caps(df_pred: pd.DataFrame) -> pd.DataFrame:
    out = df_pred.copy()
    if macro_caps.empty:
        return out
    for m in MONTHS:
        cap = macro_caps[macro_caps["month"]==m]
        if cap.empty or pd.isna(cap.iloc[0].get("cap_total")):
            continue
        tgt = float(cap.iloc[0]["cap_total"])
        mask = out["month"]==m
        cur_sum = out.loc[mask, "pred"].sum()
        if cur_sum > 0 and tgt > 0:
            scale = tgt / cur_sum
            out.loc[mask, "pred"] = out.loc[mask, "pred"] * scale
    return out

final_pred = reconcile_caps(final_pred)

# ===== 9) 제출 포맷으로 피벗 & 저장 =====
def pivot_to_submission_format(pred_long: pd.DataFrame, tmpl: pd.DataFrame) -> pd.DataFrame:
    # tmpl의 월 순서대로 wide 생성
    # tmpl의 month 컬럼 이름이 없는 경우가 있어, melt 시 처리된 month가 존재한다고 가정
    month_cols = [c for c in tmpl.columns if c not in ["product_name","product_id","id"]]
    if not month_cols:
        month_cols = MONTHS
    wide = pred_long.pivot_table(index="product_name", columns="month", values="pred", aggfunc="mean").reset_index()
    # 월 컬럼을 템플릿 순서로 재배치
    month_map = {}
    if not any([m for m in month_cols if str(m).startswith("20")]):
        # 템플릿에 YYYY-MM이 없으면 12개 컬럼을 월1..12로 간주 → 매핑
        mc_map = {MONTHS[i]: month_cols[i] for i in range(min(len(MONTHS), len(month_cols)))}
        wide = wide.rename(columns=mc_map)
    else:
        # 템플릿이 YYYY-MM이면 그대로
        pass

    # 템플릿과 merge하여 제품 순서 유지
    out = tmpl.copy()
    use_cols = ["product_name"] + [c for c in out.columns if c!="product_name"]
    out = out[["product_name"]].merge(wide, on="product_name", how="left")
    # 남은 월 없는 경우 0 채움
    for c in out.columns:
        if c!="product_name":
            out[c] = out[c].fillna(0.0)
    # 소수점 처리(필요 시 반올림)
    for c in out.columns:
        if c!="product_name":
            out[c] = out[c].astype(float)
    return out

submission_out = pivot_to_submission_format(final_pred[["product_name","month","pred"]], sub)
save_path = ROOT / "submission_persona_agent.csv"
submission_out.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"[OK] saved: {save_path}")
