#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DW Persona Forecast (OpenAI LLM + 시뮬레이션) — v6 (Function Calling + product_info 확장)
- .env 로드(OPENAI_API_KEY)
- ✅ OpenAI tools(function calling) + tool_choice 강제 → JSON을 함수 인자로만 반환(근본 해결)
- 텍스트 JSON 경로는 폴백 + 리페어 유지
- product_info.csv 의 (product_feature, category_level_1/2/3) 반영
- month 컬럼 float 캐스팅(FutureWarning 제거)
- 기본 MC 반복수 400 (안정화)
- 오프라인 더미 페르소나: 카테고리별 구매확률/성향 튜닝
- LLM 프롬프트에 카테고리 맥락(계절/프로모션/행사/학사/날씨/건강) + 제품 특성 자동 주입
"""

import os, sys, json, time, argparse, random, textwrap, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# -----------------------------
# 환경변수(.env) 로드
# -----------------------------
load_dotenv()  # .env에서 OPENAI_API_KEY 읽기

# -----------------------------
# Metric
# -----------------------------
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom > 0
    value = np.zeros_like(denom, dtype=float)
    value[mask] = 2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
    return np.mean(value) * 100.0

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
    base_units: float = 5000.0  # 제품 메타에 base_units 없을 때 기본 스케일
    features: Optional[str] = None

# -----------------------------
# Category helpers
# -----------------------------
def norm(s: Optional[str]) -> str:
    return (s or "").lower()

def get_category_context(category: Optional[str]) -> str:
    """카테고리별 프롬프트 보강 텍스트."""
    c = norm(category)
    ctx = []
    # 공통: 명절/연말·연초/대형행사/학사
    ctx.append("- 공통: 설(1-2월), 추석(9-10월), 연말(12월) 수요 상승 가능성.")
    ctx.append("- 11월 대형 유통 이벤트: 코리아세일페스타, 블랙프라이데이 등 판촉 강화.")
    ctx.append("- 학사 일정: 3·9월 개학·개강, 1-2·7-8월 방학으로 가정 내 소비 패턴 변화.")
    # 카테고리 세부
    if any(k in c for k in ["yogurt","요거트","그릭","건강","헬스","발효유"]):
        ctx += [
            "- 건강/다이어트 시즌: 1월(신년 결심), 5-6월(여름 체형관리) 수요↑.",
            "- 고단백/저지방 트렌드, 냉장 유통 신선도 고려.",
            "- 광고모델의 '건강·셀럽' 이미지 적합 시 리프트↑."
        ]
    if any(k in c for k in ["beverage","drink","음료","water","워터","주스","탄산","커피"]):
        ctx += [
            "- 기온 민감: 7-8월 무더위 시즌 수요 급증.",
            "- 1-2·11-12월 실내 소비/따뜻한 음료 전환 고려.",
            "- 편의점/자판기 채널, 1+1/멀티팩 프로모션 영향 큼."
        ]
    if any(k in c for k in ["rte","간편식","즉석","ready-to-eat","밀키트","라면","스낵","축산캔"]):
        ctx += [
            "- 방학/시험기간/직장인 점심 수요 변화.",
            "- 온라인 장보기/구독 배송 반복 구매 패턴."
        ]
    if any(k in c for k in ["유가","참치","캔","통조림","오일","seafood","fish","tuna","조미료","액상조미료","참기름"]):
        ctx += [
            "- 원자재/유가·해상운임 변동 → 가격/프로모션 민감도 상승.",
            "- 비축/대량구매와 프로모션 결합 시 스파이크."
        ]
    if any(k in c for k in ["키즈","아동","영양","베이비"]):
        ctx += [
            "- 학부모 타깃: 안전성/영양/리뷰 의존↑.",
            "- 학기 일정(입학·개학)과 어린이날(5월) 선물 수요."
        ]
    return "\n".join(ctx)

def tune_persona_ranges_by_category(category: Optional[str]) -> Dict[str, Any]:
    """
    카테고리별 기본 분포(구매확률 범위, 성향 편향) 설정.
    반환: {"prob_range":(lo,hi), "bias":{...}}
    """
    c = norm(category)
    cfg = {"prob_range": (18, 55), "bias": {}}
    if any(k in c for k in ["yogurt","요거트","그릭","건강","헬스","발효유"]):
        cfg["prob_range"] = (22, 60)
        cfg["bias"] = {
            "price_sensitivity": ("down", 0.1),
            "brand_loyalty": ("up", 0.1),
            "sustainability_preference": ("up", 0.08),
            "innovation_seeking": ("up", 0.05)
        }
    elif any(k in c for k in ["beverage","drink","음료","water","워터","주스","탄산","커피"]):
        cfg["prob_range"] = (25, 65)
        cfg["bias"] = {
            "price_sensitivity": ("up", 0.08),
            "promotion_sensitivity": ("up", 0.1),
            "channel_preference": ("up_ON", 0.05)
        }
    elif any(k in c for k in ["rte","간편식","즉석","ready-to-eat","밀키트","라면","스낵","축산캔"]):
        cfg["prob_range"] = (20, 58)
        cfg["bias"] = {
            "innovation_seeking": ("up", 0.05),
            "review_dependence": ("up", 0.08),
        }
    elif any(k in c for k in ["유가","참치","캔","통조림","오일","seafood","fish","tuna","조미료","액상조미료","참기름"]):
        cfg["prob_range"] = (18, 52)
        cfg["bias"] = {
            "price_sensitivity": ("up", 0.08),
            "brand_loyalty": ("up", 0.06),
        }
    elif any(k in c for k in ["키즈","아동","영양","베이비"]):
        cfg["prob_range"] = (20, 56)
        cfg["bias"] = {
            "review_dependence": ("up", 0.1),
            "risk_aversion": ("up", 0.08),
        }
    return cfg

# -----------------------------
# JSON repair helpers (fallback 경로용)
# -----------------------------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def repair_json_string(s: str) -> str:
    """
    JSON 문자열 복구:
      - 코드펜스 제거
      - JSON 객체만 추출
      - personas 배열에서 }{ → },{ 삽입
      - 값과 다음 키 사이 콤마 누락 보정
      - 트레일링 콤마 제거
      - (최후수단) 단일따옴표 → 이중따옴표
    """
    s = _strip_code_fences(s)
    start = s.find("{"); end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]

    def fix_personas_segment(txt: str) -> str:
        m = re.search(r'"personas"\s*:\s*\[', txt)
        if not m: return txt
        i = m.end(); depth = 1; j = i
        while j < len(txt) and depth > 0:
            if txt[j] == '[': depth += 1
            elif txt[j] == ']': depth -= 1
            j += 1
        seg = txt[i:j-1] if j-1 > i else ""
        seg = re.sub(r'\}\s*\{', '},{', seg)
        seg = re.sub(r'([0-9\]\}])\s*("(?=[^"]+"\s*:))', r'\1,\2', seg)
        return txt[:i] + seg + txt[j-1:]

    s = fix_personas_segment(s)
    s = re.sub(r'([0-9\]\}])\s*("(?=[^"]+"\s*:))', r'\1,\2', s)
    s = re.sub(r",\s*([\}\]])", r"\1", s)
    try:
        json.loads(s); return s
    except Exception:
        pass
    s = re.sub(r"(\s*)'([^']+)'\s*:", r'\1"\2":', s)
    s = re.sub(r':\s*\'([^\'\\]*(?:\\.[^\'\\]*)*)\'', r': "\1"', s)
    s = re.sub(r",\s*([\}\]])", r"\1", s)
    return s

def parse_personas_json(raw_text: str) -> dict:
    raw = _strip_code_fences(raw_text)
    try:
        return json.loads(raw)
    except Exception:
        pass
    fixed = repair_json_string(raw_text)
    return json.loads(fixed)

# -----------------------------
# Prompt (내용) + Tools(스키마) 생성
# -----------------------------
def build_persona_schema(n_personas: int) -> dict:
    """OpenAI function calling parameters(JSON Schema)"""
    return {
        "type": "object",
        "properties": {
            "personas": {
                "type": "array",
                "minItems": n_personas,
                "maxItems": n_personas,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "weight": {"type": "number"},
                        "attributes": {"type": "object"},
                        "attribute_weights": {"type": "object"},
                        "purchase_probability_pct": {"type": "number"},
                        "monthly_purchase_frequency": {
                            "type": "array",
                            "minItems": 12,
                            "maxItems": 12,
                            "items": {"type": "number"}
                        }
                    },
                    "required": [
                        "name","weight","attributes","attribute_weights",
                        "purchase_probability_pct","monthly_purchase_frequency"
                    ],
                    "additionalProperties": True
                }
            }
        },
        "required": ["personas"],
        "additionalProperties": False
    }

def normalize_category_guess(product: ProductConfig) -> str:
    """간단 카테고리 보정: 제품명/특성에 '참기름' 등 명시 키워드가 강하면 보정."""
    c = (product.category or "").lower()
    name = (product.product_name or "").lower()
    feat = (product.features or "").lower()
    text = name + " " + feat
    if any(k in text for k in ["참기름","sesame","goma","참유"]):
        return "오일/참기름"
    return product.category or ""

def build_single_turn_prompt(product: ProductConfig, n_personas: int) -> str:
    cat_hint = normalize_category_guess(product) or product.category
    cat_ctx = get_category_context(cat_hint)
    features_txt = f"- 제품 특성: {product.features}" if product.features else "- 제품 특성: (제공 없음)"
    guidance = f"""
당신은 한국 소비재(식품/음료) 시장 조사 전문가입니다.
다음 제품에 대해 '싱글 턴(single turn)'으로 소비자 페르소나를 정확히 {n_personas}명 생성하세요.
각 페르소나는 최소 10개 이상의 속성과 각 속성별 영향 가중치(0~1)를 포함하고,
월별 구매 빈도 패턴(12개월: 7월~다음해 6월, 합=1.0)과 구매 확률(%)을 제공합니다.
한국 시장의 계절성/행사/프로모션/광고모델 영향(아래 맥락 및 제품 특성 참고)을 반영하세요.

제품 정보:
- 제품명: {product.product_name}
- 카테고리: {product.category or "미상"}
- 가격(원): {product.price if product.price is not None else "미상"}
{features_txt}

카테고리 맥락:
{cat_ctx}

반드시 도구 호출로만(personas 인자) 답변하세요. 추가 설명 텍스트를 출력하지 마세요.
""".strip()
    return guidance

# -----------------------------
# OpenAI 호출 (Function Calling 강제)
# -----------------------------
def call_openai_with_tools(prompt: str, api_key: str,
                           n_personas: int,
                           model: str = "gpt-4o-mini",
                           temperature: float = 0.2,
                           timeout: int = 90,
                           max_retries: int = 2) -> dict:
    """
    tools + tool_choice 강제. 함수 인자(JSON)를 dict로 반환.
    실패 시 예외.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    parameters_schema = build_persona_schema(n_personas)
    tools = [{
        "type": "function",
        "function": {
            "name": "submit_personas",
            "description": "Return personas as strict JSON only.",
            "parameters": parameters_schema
        }
    }]

    payload = {
        "model": model,
        "temperature": float(temperature),
        "messages": [
            {"role": "system",
             "content": "Return output ONLY via the function call `submit_personas`. Do not write prose."},
            {"role": "user", "content": prompt}
        ],
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": "submit_personas"}},
        "max_tokens": 7000
    }

    last_err = None
    for _ in range(max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            tcalls = data["choices"][0]["message"].get("tool_calls") or []
            if not tcalls:
                # 드물게 tool_calls가 비어있으면 텍스트 경로로 폴백
                content = data["choices"][0]["message"].get("content") or ""
                if not isinstance(content, str):
                    content = str(content)
                return parse_personas_json(content)
            # 함수 인자(JSON 문자열)
            args_str = tcalls[0]["function"].get("arguments", "{}")
            return json.loads(args_str)
        except Exception as e:
            last_err = e
            time.sleep(1.2)
    raise RuntimeError(f"OpenAI tool-call 실패: {last_err}")

# -----------------------------
# Offline dummy personas (category-aware)
# -----------------------------
def offline_dummy_personas(n: int, category: Optional[str], seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    personas = []
    w = np.array([rng.random() + 0.1 for _ in range(n)], dtype=float); w = w / w.sum()

    cfg = tune_persona_ranges_by_category(category)
    p_lo, p_hi = cfg["prob_range"]
    bias = cfg["bias"]

    for i in range(n):
        age = rng.randint(18, 65)
        ch_pref = rng.choice(["ON","OFF","MIXED"])
        attrs = {
            "age": age,
            "gender": rng.choice(["male","female"]),
            "income_band": rng.choice(["<2M KRW","2-4M","4-6M","6-8M","8M+"]),
            "region_kr": rng.choice(["서울","경기","인천","부산","대구","광주","대전","울산","세종","강원","충북","충남","전북","전남","경북","경남","제주"]),
            "education": rng.choice(["고졸","대졸","대학원"]),
            "marital_status": rng.choice(["single","married","with_kids","other"]),
            "household_size": rng.randint(1,5),
            "channel_preference": ch_pref,
            "price_sensitivity": round(rng.random(),3),
            "promotion_sensitivity": round(rng.random(),3),
            "brand_loyalty": round(rng.random(),3),
            "innovation_seeking": round(rng.random(),3),
            "review_dependence": round(rng.random(),3),
            "sustainability_preference": round(rng.random(),3),
            "risk_aversion": round(rng.random(),3),
        }

        keys = list(attrs.keys())
        aw = np_rng.random(len(keys)); aw = aw / aw.sum()

        def clamp01(x): return float(max(0.0, min(1.0, x)))
        if "price_sensitivity" in bias:
            direction, delta = bias["price_sensitivity"]
            attrs["price_sensitivity"] = clamp01(attrs["price_sensitivity"] + (delta if direction=="up" else -delta))
        if "brand_loyalty" in bias:
            direction, delta = bias["brand_loyalty"]
            attrs["brand_loyalty"] = clamp01(attrs["brand_loyalty"] + (delta if direction=="up" else -delta))
        if "sustainability_preference" in bias:
            direction, delta = bias["sustainability_preference"]
            attrs["sustainability_preference"] = clamp01(attrs["sustainability_preference"] + (delta if direction=="up" else -delta))
        if "innovation_seeking" in bias:
            direction, delta = bias["innovation_seeking"]
            attrs["innovation_seeking"] = clamp01(attrs["innovation_seeking"] + (delta if direction=="up" else -delta))
        if "promotion_sensitivity" in bias:
            direction, delta = bias["promotion_sensitivity"]
            attrs["promotion_sensitivity"] = clamp01(attrs["promotion_sensitivity"] + (delta if direction=="up" else -delta))
        if "review_dependence" in bias:
            direction, delta = bias["review_dependence"]
            attrs["review_dependence"] = clamp01(attrs["review_dependence"] + (delta if direction=="up" else -delta))
        if "risk_aversion" in bias:
            direction, delta = bias["risk_aversion"]
            attrs["risk_aversion"] = clamp01(attrs["risk_aversion"] + (delta if direction=="up" else -delta))
        if bias.get("channel_preference") == ("up_ON", 0.05) and attrs["channel_preference"] != "ON":
            if rng.random() < 0.25:
                attrs["channel_preference"] = "ON"

        m = np.abs(np_rng.normal(loc=1.0, scale=0.2, size=12)); m = m / m.sum()

        personas.append({
            "name": f"Persona_{i+1}",
            "weight": float(round(w[i],6)),
            "attributes": attrs,
            "attribute_weights": {k: float(round(v,4)) for k,v in zip(keys, aw)},
            "purchase_probability_pct": float(round(rng.uniform(p_lo, p_hi),2)),
            "monthly_purchase_frequency": [float(round(x,6)) for x in m.tolist()]
        })
    return {"personas": personas}

# -----------------------------
# Simulation
# -----------------------------
def simulate_monthly_units(personas: List[Persona],
                           base_units: float = 5000.0,
                           category_seasonality: Optional[Dict[int, float]] = None,
                           promo_lift_by_month: Optional[Dict[int, float]] = None,
                           mc_runs: int = 400,  # 안정화 기본
                           seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    months = 12
    default_seasonality = {1:1.05, 2:1.08, 3:0.98, 4:0.97, 5:0.99, 6:0.98,
                           7:1.00, 8:1.02, 9:1.07, 10:1.03, 11:1.02, 12:1.10}
    if category_seasonality is None:
        category_seasonality = default_seasonality
    if promo_lift_by_month is None:
        promo_lift_by_month = {m:1.0 for m in range(1,13)}

    def k_to_calendar_month(k: int) -> int:
        return (6 + k) if k <= 6 else (k - 6)  # 1→7 … 12→6

    monthly_totals = np.zeros((mc_runs, months), dtype=float)
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
            season = category_seasonality.get(cal_m, 1.0)
            promo  = promo_lift_by_month.get(cal_m, 1.0)
            month_units[k-1] *= season * promo
        monthly_totals[run,:] = month_units
    return np.mean(monthly_totals, axis=0)

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
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--personas_per_product", type=int, default=40)
    ap.add_argument("--mc_runs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--category_seasonality_json", type=str, default="")
    ap.add_argument("--promo_lift_json", type=str, default="")
    ap.add_argument("--actuals_csv", type=str, default="")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    prod_df = pd.read_csv(args.products)
    sub_df  = pd.read_csv(args.sample_submission)

    # month 컬럼 float 캐스팅(FutureWarning 제거)
    month_cols = [c for c in sub_df.columns if c.startswith("months_since_launch_")]
    for c in month_cols:
        if c in sub_df.columns:
            sub_df[c] = pd.to_numeric(sub_df[c], errors="coerce").astype(float)

    category_seasonality = None
    promo_lift = None
    if args.category_seasonality_json and os.path.exists(args.category_seasonality_json):
        with open(args.category_seasonality_json, "r", encoding="utf-8") as f:
            category_seasonality = {int(k):float(v) for k,v in json.load(f).items()}
    if args.promo_lift_json and os.path.exists(args.promo_lift_json):
        with open(args.promo_lift_json, "r", encoding="utf-8") as f:
            promo_lift = {int(k):float(v) for k,v in json.load(f).items()}

    all_persona_json = {}
    prompt_log = []
    out = sub_df.copy()
    product_names = list(out["product_name"].astype(str).values)

    for p_name in product_names:
        meta = prod_df.loc[prod_df["product_name"] == p_name].head(1)
        if len(meta) == 0:
            product = ProductConfig(product_name=p_name)
        else:
            row = meta.iloc[0].to_dict()

            # 카테고리: category_level_1/2/3 → "A > B > C"
            if all(k in row for k in ["category_level_1","category_level_2","category_level_3"]):
                cats = [str(row.get("category_level_1","")).strip(),
                        str(row.get("category_level_2","")).strip(),
                        str(row.get("category_level_3","")).strip()]
                cats = [c for c in cats if c]
                category = " > ".join(cats) if cats else None
            else:
                category = row.get("category", None)  # 구버전 호환

            features = str(row.get("product_feature","")).strip() if "product_feature" in row else None
            price = float(row["price"]) if "price" in row and pd.notna(row["price"]) else None
            base_units = float(row["base_units"]) if "base_units" in row and pd.notna(row["base_units"]) else 5000.0

            product = ProductConfig(
                product_name = p_name,
                category     = category,
                price        = price,
                base_units   = base_units,
                features     = features,
            )
            print(f"[INFO] Processing {p_name} ...", flush=True)

        # 프롬프트(카테고리 맥락 + 제품특성 포함)
        prompt = build_single_turn_prompt(product, args.personas_per_product)
        prompt_log.append({"product": p_name, "prompt": prompt})

        # 페르소나 생성
        if args.llm_backend == "openai":
            if not args.openai_api_key:
                print("[WARN] OPENAI_API_KEY 비어있음 → offline dummy 사용", file=sys.stderr)
                persona_json = offline_dummy_personas(args.personas_per_product, product.category, seed=args.seed)
            else:
                try:
                    persona_json = call_openai_with_tools(
                        prompt, api_key=args.openai_api_key,
                        n_personas=args.personas_per_product,
                        model=args.openai_model, temperature=args.temperature
                    )
                except Exception as e:
                    # tool-call 경로 실패 → 텍스트 JSON 폴백 시도
                    try:
                        url = "https://api.openai.com/v1/chat/completions"
                        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.openai_api_key}"}
                        payload = {
                            "model": args.openai_model,
                            "messages": [
                                {"role":"system","content":"Return ONLY strict JSON conforming to the schema."},
                                {"role":"user","content": prompt}
                            ],
                            "response_format": {"type": "json_object"},
                            "temperature": float(args.temperature),
                            "max_tokens": 7000
                        }
                        r = requests.post(url, headers=headers, json=payload, timeout=90)
                        r.raise_for_status()
                        data = r.json()
                        content = data["choices"][0]["message"].get("content") or ""
                        persona_json = parse_personas_json(content)
                    except Exception as pe:
                        # raw 저장 후 더미
                        stamp = time.strftime("%Y%m%d_%H%M%S")
                        safe_name = re.sub(r'[^0-9A-Za-z가-힣._-]+', '_', p_name)
                        raw_path = os.path.splitext(args.out_csv)[0] + f".raw_{stamp}_{safe_name}.txt"
                        try:
                            with open(raw_path, "w", encoding="utf-8") as f:
                                f.write(str(pe))
                        except Exception:
                            pass
                        print(f"[WARN] JSON 파싱/호출 실패(제품={p_name}) → offline dummy 대체: {e}", file=sys.stderr)
                        persona_json = offline_dummy_personas(args.personas_per_product, product.category, seed=args.seed)
        else:
            persona_json = offline_dummy_personas(args.personas_per_product, product.category, seed=args.seed)

        plist = persona_json.get("personas", [])
        if not isinstance(plist, list) or len(plist) == 0:
            print(f"[WARN] 비어있는 페르소나 응답 → offline dummy로 대체 (제품={p_name})", file=sys.stderr)
            plist = offline_dummy_personas(args.personas_per_product, product.category, seed=args.seed)["personas"]

        # 정규화 및 구조화
        weights = np.array([max(0.0, float(p.get("weight", 0.0))) for p in plist], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(plist), dtype=float)
        weights = weights / weights.sum()

        personas: List[Persona] = []
        for i, pj in enumerate(plist):
            mf = np.array(pj.get("monthly_purchase_frequency", [1]*12), dtype=float)
            mf = np.maximum(mf, 0.0)
            if mf.sum() <= 0: mf[:] = 1.0
            mf = mf / mf.sum()
            personas.append(Persona(
                name = str(pj.get("name", f"Persona_{i+1}")),
                weight = float(weights[i]) if i < len(weights) else float(1.0/len(plist)),
                attributes = dict(pj.get("attributes", {})),
                attribute_weights = {k: float(v) for k,v in pj.get("attribute_weights", {}).items()},
                monthly_purchase_frequency = mf.tolist(),
                purchase_probability_pct = float(pj.get("purchase_probability_pct", 30.0))
            ))

        # 시뮬레이션
        monthly_units = simulate_monthly_units(
            personas=personas,
            base_units=product.base_units,
            category_seasonality=category_seasonality,
            promo_lift_by_month=promo_lift,
            mc_runs=args.mc_runs,
            seed=args.seed
        )

        # 제출 포맷 채우기
        for k in range(12):
            col = f"months_since_launch_{k+1}"
            if col in out.columns:
                out.loc[out["product_name"] == p_name, col] = float(round(max(0.0, monthly_units[k]),2))

        # 로그(JSON)
        all_persona_json[p_name] = {
            "personas": [{
                "name": x.name,
                "weight": x.weight,
                "attributes": x.attributes,
                "attribute_weights": x.attribute_weights,
                "monthly_purchase_frequency": x.monthly_purchase_frequency,
                "purchase_probability_pct": x.purchase_probability_pct,
            } for x in personas]
        }

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

if __name__ == "__main__":
    main()
