#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DW Persona Forecast — v8.1 (OpenAI + Simulation, robust AW parsing)
- OpenAI function-calling 강제(JSON 깨짐 방지)
- product_info 확장(price, base_units) 반영
- 프롬프트: '동원 마케팅→본사 재고관리 30년차 부장' 관점 + B2B/기관/해외바이어 포함
- 키워드 감지: 광고모델/SNS/저당·락토프리 → 초기 3개월/타깃군 보정
- SKU별 프로모션 덮어쓰기(공통 승수 × SKU 전용 승수)
- 초기 3개월 가중(SNS/광고 노출 제품)
- 로그 강화(콘솔 + .log 파일)
- attribute_weights 파서 견고화 (dict/list/tuple/array 모두 허용)
- 옵션 SMAPE
"""

import os, sys, json, time, argparse, random, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# -----------------------------
# 환경변수(.env) 로드
# -----------------------------
load_dotenv()

# -----------------------------
# Metric
# -----------------------------
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom > 0
    value = np.zeros_like(denom, dtype=float)
    value[mask] = 2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
    return float(np.mean(value) * 100.0)

# -----------------------------
# Data classes
# -----------------------------
@dataclass
class Persona:
    name: str
    weight: float
    attributes: Dict[str, Any]
    attribute_weights: Dict[str, float]
    monthly_purchase_frequency: List[float]
    purchase_probability_pct: float

@dataclass
class ProductConfig:
    product_name: str
    category: Optional[str] = None
    price: Optional[float] = None
    base_units: float = 5000.0
    features: Optional[str] = None

# -----------------------------
# Helpers
# -----------------------------
def norm(s: Optional[str]) -> str:
    return (s or "").lower()

def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def append_log(log_path: str, msg: str):
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{now_ts()}] {msg}\n")
    except Exception:
        pass

# -----------------------------
# Category context + tuning
# -----------------------------
def get_category_context(category: Optional[str]) -> str:
    c = norm(category)
    ctx = []
    ctx.append("- 공통: 설(1-2월), 추석(9-10월), 연말(12월) 수요 상승 가능성.")
    ctx.append("- 11월: 코리아세일페스타·블랙프라이데이 등 판촉 강화.")
    ctx.append("- 학사: 3·9월 개학/개강, 1-2·7-8월 방학으로 가정 내 소비 패턴 변화.")
    if any(k in c for k in ["yogurt","요거트","그릭","건강","헬스","발효유"]):
        ctx += ["- 건강/다이어트 시즌: 1월, 5-6월 수요↑.", "- 고단백/저지방 트렌드.", "- 건강 이미지 광고효과↑."]
    if any(k in c for k in ["beverage","음료","커피","주스","탄산","water","워터"]):
        ctx += ["- 기온 민감: 7-8월 수요 급증.", "- 겨울 따뜻한 음료 전환.", "- 편의점 1+1/멀티팩 민감."]
    if any(k in c for k in ["rte","간편식","즉석","밀키트","라면","스낵","축산캔"]):
        ctx += ["- 방학/시험/직장인 점심 수요 변동.", "- 온라인 장보기/구독 반복 구매."]
    if any(k in c for k in ["유가","참치","캔","통조림","오일","tuna","조미료","액상조미료","참기름"]):
        ctx += ["- 원자재/운임 변동 → 가격/프로모션 민감.", "- 비축/대량구매 스파이크."]
    if any(k in c for k in ["키즈","아동","영양","베이비"]):
        ctx += ["- 학부모: 안전성/영양/리뷰 의존↑.", "- 학기/어린이날 수요."]
    return "\n".join(ctx)

def tune_persona_ranges_by_category(category: Optional[str]) -> Dict[str, Any]:
    c = norm(category)
    cfg = {"prob_range": (18, 55), "bias": {}}
    if any(k in c for k in ["yogurt","요거트","그릭","건강","헬스","발효유"]):
        cfg["prob_range"] = (22, 60)
        cfg["bias"] = {"price_sensitivity": ("down", 0.1), "brand_loyalty": ("up", 0.1),
                       "sustainability_preference": ("up", 0.08), "innovation_seeking": ("up", 0.05)}
    elif any(k in c for k in ["beverage","음료","water","워터","주스","탄산","커피"]):
        cfg["prob_range"] = (25, 65)
        cfg["bias"] = {"price_sensitivity": ("up", 0.08), "promotion_sensitivity": ("up", 0.1),
                       "channel_preference": ("up_ON", 0.05)}
    elif any(k in c for k in ["rte","간편식","즉석","밀키트","라면","스낵","축산캔"]):
        cfg["prob_range"] = (20, 58)
        cfg["bias"] = {"innovation_seeking": ("up", 0.05), "review_dependence": ("up", 0.08)}
    elif any(k in c for k in ["유가","참치","캔","통조림","오일","tuna","조미료","액상조미료","참기름"]):
        cfg["prob_range"] = (18, 52)
        cfg["bias"] = {"price_sensitivity": ("up", 0.08), "brand_loyalty": ("up", 0.06)}
    elif any(k in c for k in ["키즈","아동","영양","베이비"]):
        cfg["prob_range"] = (20, 56)
        cfg["bias"] = {"review_dependence": ("up", 0.1), "risk_aversion": ("up", 0.08)}
    return cfg

# -----------------------------
# JSON repair (rare fallback)
# -----------------------------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I); s = re.sub(r"\s*```$", "", s)
    return s.strip()

def repair_json_string(s: str) -> str:
    s = _strip_code_fences(s)
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start: s = s[start:end+1]
    s = re.sub(r'\}\s*\{', '},{', s)  # }{
    s = re.sub(r'([0-9\]\}])\s*("(?=[^"]+"\s*:))', r'\1,\2', s)  # missing commas
    s = re.sub(r",\s*([\}\]])", r"\1", s)  # trailing commas
    try:
        json.loads(s); return s
    except Exception:
        s = re.sub(r"(\s*)'([^']+)'\s*:", r'\1"\2":', s)
        s = re.sub(r":\s*'([^'\\]*(?:\\.[^'\\]*)*)'", r': "\1"', s)
        s = re.sub(r",\s*([\}\]])", r"\1", s)
        return s

def parse_personas_json(raw_text: str) -> dict:
    raw = _strip_code_fences(raw_text)
    try:
        return json.loads(raw)
    except Exception:
        fixed = repair_json_string(raw_text)
        return json.loads(fixed)

# -----------------------------
# NEW: attribute_weights 안전 변환기
# -----------------------------
def coerce_attribute_weights(raw_aw, attributes) -> Dict[str, float]:
    """
    attribute_weights가 dict/list/기타 어떤 형태로 와도
    {속성명: 가중치(float)} 딕셔너리로 안전 변환 + 0~1 정규화.
    """
    mapping: Dict[str, float] = {}

    # 1) dict
    if isinstance(raw_aw, dict):
        for k, v in raw_aw.items():
            try:
                mapping[str(k)] = float(v)
            except Exception:
                continue

    # 2) list 류
    elif isinstance(raw_aw, list):
        ok = False
        # 2-1) [{"name": "...", "weight": ...}, ...]
        for item in raw_aw:
            if isinstance(item, dict) and ("name" in item) and ("weight" in item):
                try:
                    mapping[str(item["name"])] = float(item["weight"])
                    ok = True
                except Exception:
                    pass
        # 2-2) [["price_sensitivity", 0.12], ...]
        if not ok:
            for item in raw_aw:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    try:
                        mapping[str(item[0])] = float(item[1])
                        ok = True
                    except Exception:
                        pass
        # 2-3) [0.12, 0.07, ...] → attributes의 앞에서부터 할당
        if not ok:
            keys = list(attributes.keys()) if isinstance(attributes, dict) else []
            for i, v in enumerate(raw_aw[:len(keys)]):
                try:
                    mapping[str(keys[i])] = float(v)
                except Exception:
                    pass

    # 3) 실패 시 균등
    if not mapping:
        keys = list(attributes.keys()) if isinstance(attributes, dict) else []
        if keys:
            w = 1.0 / len(keys)
            mapping = {k: w for k in keys}
        else:
            mapping = {}

    # 4) 0 미만 제거 + 정규화
    mapping = {k: max(0.0, float(v)) for k, v in mapping.items()}
    s = sum(mapping.values())
    if s > 0:
        mapping = {k: v / s for k, v in mapping.items()}

    return mapping

# -----------------------------
# Prompt + Tools
# -----------------------------
def build_persona_schema(n_personas: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "personas": {
                "type": "array",
                "minItems": n_personas, "maxItems": n_personas,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "weight": {"type": "number"},
                        "attributes": {"type": "object"},
                        "attribute_weights": {"type": "object"},
                        "purchase_probability_pct": {"type": "number"},
                        "monthly_purchase_frequency": {"type": "array", "minItems":12,"maxItems":12,"items":{"type":"number"}}
                    },
                    "required": ["name","weight","attributes","attribute_weights","purchase_probability_pct","monthly_purchase_frequency"]
                }
            }
        },
        "required": ["personas"]
    }

def normalize_category_guess(product: ProductConfig) -> str:
    name = (product.product_name or "").lower()
    feat  = (product.features or "").lower()
    txt = name + " " + feat
    if any(k in txt for k in ["참기름","sesame","goma","참유"]): return "오일/참기름"
    return product.category or ""

def build_single_turn_prompt(product: ProductConfig, n_personas: int) -> str:
    cat_hint = normalize_category_guess(product) or product.category
    cat_ctx = get_category_context(cat_hint)
    features_txt = f"- 제품 특성: {product.features}" if product.features else "- 제품 특성: (제공 없음)"

    # 핵심: '동원 30년차 부장' + B2B/기관 포함 + 경영학/경제학 의사결정 시점
    guidance = f"""
너는 동원참치 마케팅 부서에서 오래 근무했고 현재는 본사 재고관리 담당 30년차 부장이다.
경영학·경제학 관점에서 수익성(마진), 성장성(판매량), 재고회전, 납품 안정성, 소비자/바이어 심리를 모두 고려해 판단한다.

아래 제품에 대해, '싱글 턴(single turn)'으로 소비자/바이어 페르소나를 정확히 {n_personas}명 생성하라.
각 페르소나는 최소 10개 이상의 속성과 각 속성별 영향 가중치(0~1)를 포함하고,
월별 구매 빈도 패턴(12개월: 7월~다음해 6월, 합=1.0)과 구매 확률(%)을 제공한다.

페르소나 타깃은 개인소비자뿐 아니라 B2B(식자재/급식/편의점/대형마트 바이어), 공공기관 구매담당, 해외 바이어도 포함한다.
- B2B/기관은 대량구매 특성상 weight는 상대적으로 높이고, 구매확률은 낮게 설정하라.
- B2B/기관의 월별 빈도는 명절/학기/계약 갱신 시점에 뾰족하게(frequency↑) 배분하라.
- 개인소비자는 리뷰·가격·프로모션·광고모델의 영향을 더 크게 받는다.

제품 정보:
- 제품명: {product.product_name}
- 카테고리: {product.category or "미상"}
- 가격(원): {product.price if product.price is not None else "미상"}
{features_txt}

카테고리 맥락:
{cat_ctx}

출력 형식:
- 도구 호출(function call)로만 답변한다(추가 텍스트 금지). 인자는 'personas' 배열 하나뿐이다.
- personas[i] 필수 키: name, weight, attributes{{최소10개}}, attribute_weights{{0~1}}, purchase_probability_pct, monthly_purchase_frequency(길이12, 합=1)

구체적 가이드:
- 속성 예: age, gender, income_band, region_kr, education, marital_status, household_size,
           channel_preference(ON/OFF/MIXED), price_sensitivity, promotion_sensitivity, brand_loyalty,
           innovation_seeking, review_dependence, sustainability_preference, risk_aversion,
           buyer_type(개인/기업/공공/해외), org_size, contract_cycle, channel(B2C/B2B) 등
- 광고모델/저당/락토프리/SNS 바이럴 등 제품 특성에 맞춰 타깃군/시즌성 보정하라.
""".strip()

    ftxt = (product.features or "")
    if "광고모델" in ftxt:
        guidance += "\n- 광고모델 효과: 20~30대 여성·수도권에서 구매확률↑(약 1.2배)로 반영."
    if ("SNS" in ftxt) or ("바이럴" in ftxt):
        guidance += "\n- SNS 바이럴: 출시 초 1~3개월 빈도↑(약 1.15배)로 반영."
    if ("저당" in ftxt) or ("락토프리" in ftxt):
        guidance += "\n- 건강 트렌드: 20~40대 건강 관심층 재구매율↑(약 1.1배)로 반영."

    guidance += "\n반드시 함수 호출로만(personas 인자) 답하라."
    return guidance

# -----------------------------
# OpenAI call (function-calling)
# -----------------------------
def build_persona_schema_for_tools(n_personas: int) -> dict:
    return build_persona_schema(n_personas)

def call_openai_with_tools(prompt: str, api_key: str, n_personas: int,
                           model: str="gpt-4o-mini", temperature: float=0.1,
                           timeout: int=120, max_retries: int=2) -> dict:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type":"application/json","Authorization":f"Bearer {api_key}"}
    tools = [{
        "type":"function",
        "function":{"name":"submit_personas","description":"Return personas as strict JSON only.",
                    "parameters": build_persona_schema_for_tools(n_personas)}
    }]
    payload = {
        "model": model, "temperature": float(temperature),
        "messages": [
            {"role":"system","content":"Return output ONLY via function call `submit_personas`."},
            {"role":"user","content": prompt}
        ],
        "tools": tools,
        "tool_choice": {"type":"function","function":{"name":"submit_personas"}},
        "max_tokens": 7000
    }

    last_err = None
    for _ in range(max_retries+1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout); r.raise_for_status()
            data = r.json()
            tcalls = data["choices"][0]["message"].get("tool_calls") or []
            if not tcalls:
                content = data["choices"][0]["message"].get("content") or "{}"
                return parse_personas_json(content)
            args_str = tcalls[0]["function"].get("arguments","{}")
            return json.loads(args_str)
        except Exception as e:
            last_err = e; time.sleep(1.2)
    raise RuntimeError(f"OpenAI tool-call 실패: {last_err}")

# -----------------------------
# Offline dummy personas (category-aware + buyer_type 포함)
# -----------------------------
def offline_dummy_personas(n: int, category: Optional[str], seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed); np_rng = np.random.default_rng(seed)
    personas = []
    w = np.array([rng.random() + 0.1 for _ in range(n)], dtype=float); w = w / w.sum()
    cfg = tune_persona_ranges_by_category(category); p_lo, p_hi = cfg["prob_range"]; bias = cfg["bias"]

    for i in range(n):
        buyer_type = rng.choices(["개인","기업","공공","해외"], weights=[0.72,0.18,0.06,0.04])[0]
        age = rng.randint(22, 65) if buyer_type=="개인" else rng.randint(28, 60)
        ch_pref = rng.choice(["ON","OFF","MIXED"])
        attrs = {
            "buyer_type": buyer_type,
            "age": age,
            "gender": rng.choice(["male","female"]) if buyer_type=="개인" else "n/a",
            "income_band": rng.choice(["<2M","2-4M","4-6M","6-8M","8M+"]) if buyer_type=="개인" else "org",
            "org_size": rng.choice(["소","중","대"]) if buyer_type!="개인" else "n/a",
            "contract_cycle": rng.choice(["분기","반기","연간"]) if buyer_type!="개인" else "n/a",
            "region_kr": rng.choice(["서울","경기","인천","부산","대구","광주","대전","울산","세종","강원","충북","충남","전북","전남","경북","경남","제주"]),
            "channel": rng.choice(["B2C","B2B"]) if buyer_type!="개인" else "B2C",
            "channel_preference": ch_pref,
            "price_sensitivity": round(rng.random(),3),
            "promotion_sensitivity": round(rng.random(),3),
            "brand_loyalty": round(rng.random(),3),
            "innovation_seeking": round(rng.random(),3),
            "review_dependence": round(rng.random(),3),
            "sustainability_preference": round(rng.random(),3),
            "risk_aversion": round(rng.random(),3),
        }

        # buyer_type 경향
        if buyer_type!="개인": w[i] *= 1.3
        prob_base = rng.uniform(p_lo, p_hi)
        if buyer_type!="개인": prob_base *= 0.8
        prob_base = max(5.0, min(90.0, prob_base))

        keys = list(attrs.keys())
        aw = np_rng.random(len(keys)); aw = aw / aw.sum()

        def clamp01(x): return float(max(0.0, min(1.0, x)))
        for k,v in bias.items():
            if k in attrs and isinstance(attrs[k], (int,float)):
                direction, delta = v
                attrs[k] = clamp01(float(attrs[k]) + (delta if direction=="up" else -delta))
        if bias.get("channel_preference") == ("up_ON", 0.05) and attrs["channel_preference"]!="ON":
            if rng.random() < 0.25: attrs["channel_preference"]="ON"

        # 기본 빈도
        m = np.abs(np_rng.normal(loc=1.0, scale=0.2, size=12))
        if buyer_type!="개인":
            for idx in [0,1,2,8,9]:   # 1,2,3,9,10월
                m[idx] *= 1.25
        m = m / m.sum()

        personas.append({
            "name": f"Persona_{i+1}",
            "weight": float(round(w[i],6)),
            "attributes": attrs,
            "attribute_weights": {k: float(round(v,4)) for k,v in zip(keys, aw)},
            "purchase_probability_pct": float(round(prob_base,2)),
            "monthly_purchase_frequency": [float(round(x,6)) for x in m.tolist()]
        })
    tot = sum(p["weight"] for p in personas)
    for p in personas: p["weight"] = float(p["weight"]/tot)
    return {"personas": personas}

# -----------------------------
# Simulation
# -----------------------------
def simulate_monthly_units(personas: List[Persona],
                           base_units: float = 5000.0,
                           category_seasonality: Optional[Dict[int, float]] = None,
                           promo_lift_by_month: Optional[Dict[int, float]] = None,
                           mc_runs: int = 800,
                           seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    months = 12
    default_seasonality = {1:1.05, 2:1.08, 3:0.98, 4:0.97, 5:0.99, 6:0.98,
                           7:1.00, 8:1.02, 9:1.07, 10:1.03, 11:1.02, 12:1.10}
    if category_seasonality is None: category_seasonality = default_seasonality
    if promo_lift_by_month is None: promo_lift_by_month = {m:1.0 for m in range(1,13)}

    def k_to_calendar_month(k: int) -> int:
        return (6 + k) if k <= 6 else (k - 6)

    totals = np.zeros((mc_runs, months), dtype=float)
    for run in range(mc_runs):
        month_units = np.zeros(months, dtype=float)
        for p in personas:
            prob = p.purchase_probability_pct / 100.0
            freq = np.array(p.monthly_purchase_frequency, dtype=float)
            expected = p.weight * base_units * prob * freq
            noise = np.clip(rng.lognormal(mean=0.0, sigma=0.15, size=months), 0.5, 2.0)
            expected *= noise
            month_units += expected
        for k in range(1, months+1):
            cal_m = k_to_calendar_month(k)
            month_units[k-1] *= category_seasonality.get(cal_m,1.0) * promo_lift_by_month.get(cal_m,1.0)
        totals[run,:] = month_units
    return np.mean(totals, axis=0)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", type=str, required=True)
    ap.add_argument("--sample_submission", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--llm_backend", type=str, default="openai", choices=["none","openai"])
    ap.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY",""))
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--personas_per_product", type=int, default=60)  # 권장 60~80
    ap.add_argument("--mc_runs", type=int, default=800)              # 권장 800~1000
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--category_seasonality_json", type=str, default="")
    ap.add_argument("--promo_lift_json", type=str, default="")
    ap.add_argument("--promo_lift_per_sku_json", type=str, default="")
    ap.add_argument("--actuals_csv", type=str, default="")
    args = ap.parse_args()

    log_path = os.path.splitext(args.out_csv)[0] + ".log"
    append_log(log_path, f"START model={args.openai_model}, personas={args.personas_per_product}, mc={args.mc_runs}, seed={args.seed}")

    random.seed(args.seed); np.random.seed(args.seed)

    prod_df = pd.read_csv(args.products)
    sub_df  = pd.read_csv(args.sample_submission)
    month_cols = [c for c in sub_df.columns if c.startswith("months_since_launch_")]
    for c in month_cols:
        sub_df[c] = pd.to_numeric(sub_df[c], errors="coerce").astype(float)

    category_seasonality = None
    if args.category_seasonality_json and os.path.exists(args.category_seasonality_json):
        with open(args.category_seasonality_json, "r", encoding="utf-8") as f:
            category_seasonality = {int(k): float(v) for k,v in json.load(f).items()}

    promo_lift = None
    if args.promo_lift_json and os.path.exists(args.promo_lift_json):
        with open(args.promo_lift_json, "r", encoding="utf-8") as f:
            promo_lift = {int(k): float(v) for k,v in json.load(f).items()}

    promo_lift_per_sku = {}
    if args.promo_lift_per_sku_json and os.path.exists(args.promo_lift_per_sku_json):
        with open(args.promo_lift_per_sku_json, "r", encoding="utf-8") as f:
            promo_lift_per_sku = json.load(f)

    out = sub_df.copy()
    product_names = list(out["product_name"].astype(str).values)
    all_persona_json, prompt_log = {}, []

    backend = args.llm_backend
    append_log(log_path, f"Backend={backend}")

    for p_name in product_names:
        t0 = time.time()
        print(f"[INFO] Processing {p_name} ...", flush=True)
        append_log(log_path, f"PROCESS {p_name} start")

        meta = prod_df.loc[prod_df["product_name"] == p_name].head(1)
        if len(meta)==0:
            product = ProductConfig(product_name=p_name)
        else:
            row = meta.iloc[0].to_dict()
            if all(k in row for k in ["category_level_1","category_level_2","category_level_3"]):
                cats = [str(row.get("category_level_1","")).strip(), str(row.get("category_level_2","")).strip(), str(row.get("category_level_3","")).strip()]
                cats = [c for c in cats if c]; category = " > ".join(cats) if cats else None
            else:
                category = row.get("category", None)
            features  = str(row.get("product_feature","")).strip() if "product_feature" in row else None
            price     = float(row["price"]) if "price" in row and pd.notna(row["price"]) else None
            base_units= float(row["base_units"]) if "base_units" in row and pd.notna(row["base_units"]) else 5000.0
            product = ProductConfig(p_name, category, price, base_units, features)

        # 프롬프트
        prompt = build_single_turn_prompt(product, args.personas_per_product)
        prompt_log.append({"product": p_name, "prompt": prompt})

        # LLM or dummy
        if backend == "openai":
            if not args.openai_api_key:
                print("[WARN] OPENAI_API_KEY 비어있음 → offline dummy", file=sys.stderr, flush=True)
                append_log(log_path, f"{p_name} API_KEY_MISSING → dummy")
                persona_json = offline_dummy_personas(args.personas_per_product, product.category, seed=args.seed)
            else:
                try:
                    persona_json = call_openai_with_tools(prompt, api_key=args.openai_api_key,
                                                          n_personas=args.personas_per_product,
                                                          model=args.openai_model, temperature=args.temperature)
                except Exception as e:
                    # fallback: json_object
                    try:
                        url = "https://api.openai.com/v1/chat/completions"
                        headers = {"Content-Type":"application/json","Authorization":f"Bearer {args.openai_api_key}"}
                        payload = {
                            "model": args.openai_model,
                            "messages": [
                                {"role":"system","content":"Return ONLY strict JSON conforming to the schema."},
                                {"role":"user","content": prompt}
                            ],
                            "response_format": {"type":"json_object"},
                            "temperature": float(args.temperature), "max_tokens": 7000
                        }
                        r = requests.post(url, headers=headers, json=payload, timeout=120); r.raise_for_status()
                        data = r.json(); content = data["choices"][0]["message"].get("content") or "{}"
                        persona_json = parse_personas_json(content)
                        append_log(log_path, f"{p_name} fallback_json_object OK")
                    except Exception as pe:
                        append_log(log_path, f"{p_name} OPENAI_FAIL → dummy : {pe}")
                        print(f"[WARN] OPENAI_FAIL → dummy : {p_name}", file=sys.stderr, flush=True)
                        persona_json = offline_dummy_personas(args.personas_per_product, product.category, seed=args.seed)
        else:
            persona_json = offline_dummy_personas(args.personas_per_product, product.category, seed=args.seed)

        plist = persona_json.get("personas", [])
        append_log(log_path, f"{p_name} personas_returned={len(plist)}")
        if not isinstance(plist, list) or len(plist)==0:
            append_log(log_path, f"{p_name} empty → dummy regen")
            plist = offline_dummy_personas(args.personas_per_product, product.category, seed=args.seed)["personas"]

        # 정규화 + 구조화 (★ 여기서 attribute_weights 안전 보정 사용)
        weights = np.array([max(0.0, float(p.get("weight", 0.0))) for p in plist], dtype=float)
        if weights.sum() <= 0: weights = np.ones(len(plist), dtype=float)
        weights = weights / weights.sum()

        personas: List[Persona] = []
        for i, pj in enumerate(plist):
            mf = np.array(pj.get("monthly_purchase_frequency", [1]*12), dtype=float)
            mf = np.maximum(mf, 0.0)
            if mf.sum() <= 0: mf[:] = 1.0
            mf = mf / mf.sum()

            attrs = dict(pj.get("attributes", {}))
            raw_aw = pj.get("attribute_weights", {})
            aw_dict = coerce_attribute_weights(raw_aw, attrs)

            personas.append(Persona(
                name = str(pj.get("name", f"Persona_{i+1}")),
                weight = float(weights[i]) if i < len(weights) else float(1.0/len(plist)),
                attributes = attrs,
                attribute_weights = aw_dict,
                monthly_purchase_frequency = mf.tolist(),
                purchase_probability_pct = float(pj.get("purchase_probability_pct", 30.0))
            ))

        # 초기 3개월 가중: SNS/광고 제품만
        if product.features and (("SNS" in product.features) or ("바이럴" in product.features) or ("광고모델" in product.features)):
            for px in personas:
                mf = np.array(px.monthly_purchase_frequency, dtype=float)
                mf[:3] = mf[:3] * 1.08  # 8% boost
                mf = mf / mf.sum()
                px.monthly_purchase_frequency = mf.tolist()
            append_log(log_path, f"{p_name} early3_boost_applied")

        # SKU별 승수 병합
        local_promo = dict(promo_lift or {})
        if p_name in promo_lift_per_sku:
            try:
                sku_map = {int(k):float(v) for k,v in promo_lift_per_sku[p_name].items()}
                for m, v in sku_map.items():
                    local_promo[m] = local_promo.get(m, 1.0) * v
                append_log(log_path, f"{p_name} promo_merged {sku_map}")
            except Exception as e:
                append_log(log_path, f"{p_name} promo_merge_fail {e}")

        # 시뮬레이션
        monthly_units = simulate_monthly_units(
            personas=personas,
            base_units=product.base_units,
            category_seasonality=category_seasonality,
            promo_lift_by_month=local_promo,
            mc_runs=args.mc_runs,
            seed=args.seed
        )

        # 제출 채움
        for k in range(12):
            col = f"months_since_launch_{k+1}"
            if col in out.columns:
                out.loc[out["product_name"] == p_name, col] = float(round(max(0.0, monthly_units[k]), 2))

        # 로그 JSON
        all_persona_json[p_name] = {
            "personas": [{
                "name": x.name, "weight": x.weight, "attributes": x.attributes,
                "attribute_weights": x.attribute_weights,
                "monthly_purchase_frequency": x.monthly_purchase_frequency,
                "purchase_probability_pct": x.purchase_probability_pct
            } for x in personas],
            "base_units": product.base_units
        }

        dt = time.time() - t0
        print(f"[INFO] Done {p_name} ({dt:.1f}s)", flush=True)
        append_log(log_path, f"PROCESS {p_name} done {dt:.1f}s base_units={product.base_units} promo_keys={[k for k,v in (local_promo or {}).items() if v!=1.0]}")

    # 저장
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jpath = os.path.splitext(args.out_csv)[0] + f".personas_{stamp}.json"
    ppath = os.path.splitext(args.out_csv)[0] + f".prompts_{stamp}.txt"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(all_persona_json, f, ensure_ascii=False, indent=2)
    with open(ppath, "w", encoding="utf-8") as f:
        for item in prompt_log:
            f.write(f"### PRODUCT: {item['product']}\n{item['prompt']}\n\n")

    print(f"[OK] submission saved: {args.out_csv}")
    print(f"[OK] personas saved: {jpath}")
    print(f"[OK] prompts saved: {ppath}")
    append_log(log_path, f"SAVED {args.out_csv} / {jpath} / {ppath}")

    # (옵션) SMAPE
    if args.actuals_csv and os.path.exists(args.actuals_csv):
        act = pd.read_csv(args.actuals_csv)
        merged = pd.merge(out, act, on="product_name", suffixes=("_pred","_true"))
        smapes = []
        for _, r in merged.iterrows():
            yp, yt = [], []
            for k in range(12):
                yp.append(float(r[f"months_since_launch_{k+1}_pred"]))
                yt.append(float(r[f"months_since_launch_{k+1}_true"]))
            smapes.append(smape(np.array(yt), np.array(yp)))
        overall = float(np.mean(smapes)) if smapes else float("nan")
        print(f"[INFO] SMAPE over 12 months: {overall:.3f}")
        append_log(log_path, f"SMAPE {overall:.3f}")

if __name__ == "__main__":
    main()
