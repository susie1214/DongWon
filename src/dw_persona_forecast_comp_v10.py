#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DW Persona Forecast — v10 (Private Score 최적화 + 앙상블/스케일 보정)

핵심 포인트
- 평가방식 반영: 최종 순위는 Private Score(12개월 SMAPE)로 결정.
  · 리더보드 환산식: 20 * ((best_score) / (team_score)) ^ N  (N=1~5)
  · 과적합 방지, 장기 안정성(12개월 패턴)을 중시 → smoothing/보수적 scale 제공
- 두 파일만으로 기본 동작:
  · --products product_info1.csv
  · --sample_submission sample_submission.csv
- LLM 페르소나(단일턴 function-calling) + 오프라인 견고 더미
- 멀티시드 앙상블(--seeds "41,42,43") → 평균으로 안정화
- 스케일 보정(전역/월별/SKU별) + 초기 3개월 가중(광고/SNS SKU)
- 공통/개별 프로모션 승수 + A/B, VR, Export lift 옵션
- 견고 파서(coerce_attribute_weights)로 LLM JSON 변형 대응
- (옵션) SMAPE/환산점수 계산

실행 예시(LLM 없이 오프라인 시뮬레이터만):
  python dw_persona_forecast_comp_v10.py ^
    --products "C:/Won/product_info1.csv" ^
    --sample_submission "C:/Won/sample_submission.csv" ^
    --out_csv "C:/Won/submission_v10.csv" ^
    --llm_backend none ^
    --personas_per_product 60 --mc_runs 1000 --seed 42

실행 예시(LLM + 앙상블 + 보정 모두):
  python dw_persona_forecast_comp_v10.py ^
    --products "C:/Won/product_info1.csv" ^
    --sample_submission "C:/Won/sample_submission.csv" ^
    --out_csv "C:/Won/submission_v10.csv" ^
    --llm_backend openai --openai_model "gpt-4o-mini" --openai_api_key "sk-..." ^
    --personas_per_product 60 --mc_runs 1000 --seeds "41,42,43" ^
    --promo_lift_json "C:/Won/promo_lift.json" ^
    --promo_lift_per_sku_json "C:/Won/promo_lift_per_sku.json" ^
    --ab_json "C:/Won/ab_plan.json" ^
    --vr_json "C:/Won/vr_store.json" ^
    --export_json "C:/Won/export_markets.json" ^
    --early3_boost_factor 1.15 ^
    --global_scale 1.05 ^
    --sku_scale_json "C:/Won/sku_scale.json" ^
    --month_scale_json "C:/Won/month_scale.json" ^
    --smooth_ma 3 --service_level 0.85
"""

import os, sys, json, time, argparse, random, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import requests

# -----------------------------
# 0) Metric / Utilities
# -----------------------------
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """SMAPE: 0~200, 낮을수록 좋음."""
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom > 0
    value = np.zeros_like(denom, dtype=float)
    value[mask] = 2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
    return float(np.mean(value) * 100.0)

def private_leaderboard_score(best_score: float, team_score: float, N: float) -> float:
    """
    Private 리더보드 환산식:
      20 * ((best_score) / (team_score)) ^ N
    점수(team_score)는 SMAPE (낮을수록 좋음).
    """
    if team_score <= 0 or best_score <= 0: return 20.0
    return 20.0 * ((best_score / team_score) ** N)

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log_append(path: str, msg: str):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{now_ts()}] {msg}\n")
    except Exception:
        pass

# -----------------------------
# 1) Data classes
# -----------------------------
@dataclass
class Persona:
    name: str
    weight: float
    attributes: Dict[str, Any]
    attribute_weights: Dict[str, float]
    monthly_purchase_frequency: List[float]  # length 12, sum=1.0
    purchase_probability_pct: float          # 0..100

@dataclass
class ProductConfig:
    product_name: str
    category: Optional[str] = None
    price: Optional[float] = None
    base_units: float = 5000.0  # 제품별 스케일(중요!)
    features: Optional[str] = None

# -----------------------------
# 2) Category context & bias
# -----------------------------
def _norm(s: Optional[str]) -> str:
    return (s or "").lower()

def category_context(category: Optional[str]) -> str:
    """LLM 힌트용 카테고리 맥락."""
    c = _norm(category)
    ctx = []
    ctx.append("- 공통: 설(1-2월), 추석(9-10월), 연말(12월) 수요 상승.")
    ctx.append("- 11월: 블랙프라이데이·코세페 판촉 강화.")
    ctx.append("- 학사: 3·9월 개학/개강, 1-2·7-8월 방학.")
    if any(k in c for k in ["요거트","yogurt","그릭","발효유","헬스"]):
        ctx += ["- 건강/다이어트 시즌: 1월, 5-6월 수요↑.", "- 고단백/저지방 트렌드.", "- 건강 이미지 광고효과↑."]
    if any(k in c for k in ["음료","beverage","커피","주스","탄산","water","워터"]):
        ctx += ["- 기온 민감: 7-8월 수요↑.", "- 겨울 따뜻한 음료 전환.", "- 편의점 프로모 민감."]
    if any(k in c for k in ["간편식","즉석","밀키트","라면","스낵","축산캔"]):
        ctx += ["- 방학/시험/직장인 점심 수요 변동.", "- 온라인 장보기/구독 반복 구매."]
    if any(k in c for k in ["참치","캔","통조림","오일","조미료","참기름","tuna"]):
        ctx += ["- 원자재/운임 변동 → 가격/프로모 민감.", "- 비축/대량구매 스파이크."]
    return "\n".join(ctx)

def category_bias(category: Optional[str]) -> Dict[str, Any]:
    """
    오프라인 더미 생성 시 카테고리별 확률/성향 편향.
    """
    c = _norm(category)
    cfg = {"prob_range": (18, 55), "bias": {}}
    if any(k in c for k in ["요거트","yogurt","그릭","발효유","헬스"]):
        cfg["prob_range"] = (22, 60)
        cfg["bias"] = {"price_sensitivity":("down",0.1), "brand_loyalty":("up",0.1),
                       "sustainability_preference":("up",0.08), "innovation_seeking":("up",0.05)}
    elif any(k in c for k in ["음료","beverage","커피","주스","탄산","water","워터"]):
        cfg["prob_range"] = (25, 65)
        cfg["bias"] = {"price_sensitivity":("up",0.08), "promotion_sensitivity":("up",0.1),
                       "channel_preference":("up_ON",0.05)}
    elif any(k in c for k in ["간편식","즉석","밀키트","라면","스낵","축산캔"]):
        cfg["prob_range"] = (20, 58)
        cfg["bias"] = {"innovation_seeking":("up",0.05), "review_dependence":("up",0.08)}
    elif any(k in c for k in ["참치","캔","통조림","오일","조미료","참기름","tuna"]):
        cfg["prob_range"] = (18, 52)
        cfg["bias"] = {"price_sensitivity":("up",0.08), "brand_loyalty":("up",0.06)}
    return cfg

# -----------------------------
# 3) JSON hardening
# -----------------------------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def repair_json_string(s: str) -> str:
    s = _strip_code_fences(s)
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]
    s = re.sub(r'\}\s*\{', '},{', s)
    s = re.sub(r'([0-9\]\}])\s*("(?=[^"]+"\s*:))', r'\1,\2', s)
    s = re.sub(r",\s*([\}\]])", r"\1", s)
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

def coerce_attribute_weights(raw_aw, attributes) -> Dict[str, float]:
    """
    dict/list/tuple/숫자리스트 모두 {key: weight}로 변환 + 정규화.
    """
    mapping: Dict[str, float] = {}
    if isinstance(raw_aw, dict):
        for k,v in raw_aw.items():
            try: mapping[str(k)] = float(v)
            except Exception: pass
    elif isinstance(raw_aw, list):
        ok = False
        for item in raw_aw:
            if isinstance(item, dict) and ("name" in item) and ("weight" in item):
                try: mapping[str(item["name"])] = float(item["weight"]); ok=True
                except Exception: pass
        if not ok:
            for item in raw_aw:
                if isinstance(item, (list,tuple)) and len(item)==2:
                    try: mapping[str(item[0])] = float(item[1]); ok=True
                    except Exception: pass
        if not ok:
            keys = list(attributes.keys()) if isinstance(attributes, dict) else []
            for i, v in enumerate(raw_aw[:len(keys)]):
                try: mapping[str(keys[i])] = float(v)
                except Exception: pass
    if not mapping:
        keys = list(attributes.keys()) if isinstance(attributes, dict) else []
        if keys:
            w = 1.0/len(keys); mapping = {k:w for k in keys}
        else:
            mapping = {}
    mapping = {k:max(0.0,float(v)) for k,v in mapping.items()}
    s = sum(mapping.values())
    if s>0: mapping = {k:v/s for k,v in mapping.items()}
    return mapping

# -----------------------------
# 4) Prompt builder (대회 관점)
# -----------------------------
def normalize_category_guess(product: ProductConfig) -> str:
    txt = (product.product_name or "").lower() + " " + (product.features or "").lower()
    if any(k in txt for k in ["참기름","sesame","goma","참유"]): return "오일/참기름"
    return product.category or ""

def enterprise_flow_prompt() -> str:
    return (
        "대기업 소비자 예측 프로세스(마케팅·경영·경제):\n"
        "1) 트렌드/경쟁/내부R&D 조사\n"
        "2) 정성(FGI)·정량(설문)·소셜/검색/POS 데이터\n"
        "3) 디지털 A/B(광고·카피·가격·패키징)\n"
        "4) LLM 디지털 트윈(개인+B2B/공공/해외)\n"
        "   · B2B/기관: weight↑, 구매확률↓, 명절/학기/계약월 freq↑\n"
        "5) 계절성·프로모·광고모델 반영 시뮬레이션\n"
        "6) 생산/재고/캠페인/ROI 의사결정\n"
        "7) 런칭 후 모니터링·피드백 루프"
    )

def build_single_turn_prompt(product: ProductConfig, n_personas: int) -> str:
    cat_hint = normalize_category_guess(product) or product.category
    ctx = category_context(cat_hint)
    features_txt = f"- 제품 특성: {product.features}" if product.features else "- 제품 특성: (제공 없음)"

    g = f"""
너는 『2025 동원×KAIST AI Competition』 참가 연구팀의 시뮬레이션 책임자이다.
동원그룹의 사내 생성형 챗봇 '동원GPT' 경험을 바탕으로,
페르소나 기반 월별 수요 예측 모델을 설계·실험한다.

경영학·경제학 관점에서 마진, 재고회전, 프로모션 ROI, 글로벌 진출 가능성을 고려한다.
아래 제품에 대해 '싱글 턴(single turn)'으로 소비자/바이어 페르소나를 정확히 {n_personas}명 생성하라.
각 페르소나는 최소 10개 속성과 속성별 영향 가중치(0~1), 월별 구매빈도(12, 합=1.0), 구매확률(%)을 제공한다.

페르소나는 개인소비자뿐 아니라 B2B(식자재/급식/편의점/대형마트 바이어), 공공기관 구매담당, 해외 바이어도 포함한다.
- B2B/기관은 대량구매 특성상 weight는 상대적으로 높이고, 구매확률은 낮게 설정하라.
- B2B/기관의 월별 빈도는 명절/학기/계약 갱신 시점에 뾰족하게(frequency↑) 배분하라.
- 개인소비자는 리뷰·가격·프로모션·광고모델의 영향을 더 크게 받는다.

제품 정보:
- 제품명: {product.product_name}
- 카테고리: {product.category or "미상"}
- 가격(원): {product.price if product.price is not None else "미상"}
{features_txt}

카테고리 맥락:
{ctx}

""".strip()

    ftxt = (product.features or "")
    if "광고모델" in ftxt:
        g += "\n- 광고모델 효과: 20~30대 여성·수도권에서 구매확률↑(약 1.2배)로 반영."
    if ("SNS" in ftxt) or ("바이럴" in ftxt):
        g += "\n- SNS 바이럴: 출시 초 1~3개월 빈도↑(약 1.15배)로 반영."
    if ("저당" in ftxt) or ("락토프리" in ftxt):
        g += "\n- 건강 트렌드: 20~40대 건강 관심층 재구매율↑(약 1.1배)로 반영."

    g += "\n\n" + enterprise_flow_prompt() + "\n"
    g += "반드시 함수 호출(personas 인자)로만 응답하라."
    return g

# -----------------------------
# 5) OpenAI function calling
# -----------------------------
def persona_schema(n_personas: int) -> dict:
    return {
        "type":"object",
        "properties":{
            "personas":{
                "type":"array","minItems":n_personas,"maxItems":n_personas,
                "items":{
                    "type":"object",
                    "properties":{
                        "name":{"type":"string"},
                        "weight":{"type":"number"},
                        "attributes":{"type":"object"},
                        "attribute_weights":{"type":"object"},
                        "purchase_probability_pct":{"type":"number"},
                        "monthly_purchase_frequency":{"type":"array","minItems":12,"maxItems":12,"items":{"type":"number"}}
                    },
                    "required":["name","weight","attributes","attribute_weights","purchase_probability_pct","monthly_purchase_frequency"]
                }
            }
        },
        "required":["personas"]
    }

def call_openai_with_tools(prompt: str, api_key: str, n_personas: int,
                           model: str="gpt-4o-mini", temperature: float=0.1,
                           timeout: int=120, max_retries: int=2) -> dict:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type":"application/json","Authorization":f"Bearer {api_key}"}
    tools = [{
        "type":"function",
        "function":{"name":"submit_personas","description":"Return personas as strict JSON only.",
                    "parameters": persona_schema(n_personas)}
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
    last_err=None
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
            last_err=e; time.sleep(1.1)
    raise RuntimeError(f"OpenAI tool-call 실패: {last_err}")

# -----------------------------
# 6) Offline dummy personas
# -----------------------------
def offline_dummy_personas(n: int, category: Optional[str], seed: int=42) -> Dict[str, Any]:
    rng = random.Random(seed); np_rng = np.random.default_rng(seed)
    persons = []
    w = np.array([rng.random()+0.1 for _ in range(n)], dtype=float); w = w/w.sum()
    cfg = category_bias(category); p_lo, p_hi = cfg["prob_range"]; bias = cfg["bias"]

    for i in range(n):
        buyer_type = rng.choices(["개인","기업","공공","해외"], weights=[0.72,0.18,0.06,0.04])[0]
        age = rng.randint(22,65) if buyer_type=="개인" else rng.randint(28,60)
        ch_pref = rng.choice(["ON","OFF","MIXED"])
        attrs = {
            "buyer_type": buyer_type,
            "age": age,
            "gender": rng.choice(["male","female"]) if buyer_type=="개인" else "n/a",
            "income_band": rng.choice(["<2M","2-4M","4-6M","6-8M","8M+"]) if buyer_type=="개인" else "org",
            "org_size": rng.choice(["소","중","대"]) if buyer_type!="개인" else "n/a",
            "contract_cycle": rng.choice(["분기","반기","연간"]) if buyer_type!="개인" else "n/a",
            "region_kr": rng.choice(["서울","경기","인천","부산","대구","광주","대전","울산","세종","강원","충북","충남","전북","전남","경북","경남","제주"]),
            "channel": "B2C" if buyer_type=="개인" else "B2B",
            "channel_preference": ch_pref,
            "price_sensitivity": round(rng.random(),3),
            "promotion_sensitivity": round(rng.random(),3),
            "brand_loyalty": round(rng.random(),3),
            "innovation_seeking": round(rng.random(),3),
            "review_dependence": round(rng.random(),3),
            "sustainability_preference": round(rng.random(),3),
            "risk_aversion": round(rng.random(),3),
        }
        # weight/prob 편향
        if buyer_type!="개인": w[i] *= 1.3
        prob = rng.uniform(p_lo, p_hi) * (0.8 if buyer_type!="개인" else 1.0)
        prob = max(5.0, min(90.0, prob))

        keys = list(attrs.keys())
        aw = np_rng.random(len(keys)); aw = aw / aw.sum()

        def clamp01(x): return float(max(0.0, min(1.0, x)))
        for k,v in bias.items():
            if k in attrs and isinstance(attrs[k], (int,float)):
                direction, delta = v
                attrs[k] = clamp01(float(attrs[k]) + (delta if direction=="up" else -delta))
        if bias.get("channel_preference")==("up_ON",0.05) and attrs["channel_preference"]!="ON":
            if rng.random()<0.25: attrs["channel_preference"]="ON"

        m = np.abs(np_rng.normal(loc=1.0, scale=0.2, size=12))
        if buyer_type!="개인":
            for idx in [0,1,2,8,9]: m[idx] *= 1.25
        m = m / m.sum()

        persons.append({
            "name": f"Persona_{i+1}",
            "weight": float(round(w[i],6)),
            "attributes": attrs,
            "attribute_weights": {k: float(round(v,4)) for k,v in zip(keys, aw)},
            "purchase_probability_pct": float(round(prob,2)),
            "monthly_purchase_frequency": [float(round(x,6)) for x in m.tolist()]
        })
    tot = sum(p["weight"] for p in persons)
    for p in persons: p["weight"] = float(p["weight"]/tot)
    return {"personas": persons}

# -----------------------------
# 7) Simulation (with lifts)
# -----------------------------
def simulate_monthly_units(personas: List[Persona],
                           base_units: float = 5000.0,
                           category_seasonality: Optional[Dict[int, float]] = None,
                           promo_lift_by_month: Optional[Dict[int, float]] = None,
                           ab_lift_first3: float = 1.0,
                           vr_lift_all: float = 1.0,
                           export_lift_all: float = 1.0,
                           service_level: float = 0.85,
                           mc_runs: int = 800,
                           seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    months = 12
    default_seasonality = {1:1.05,2:1.08,3:0.98,4:0.97,5:0.99,6:0.98,7:1.00,8:1.02,9:1.07,10:1.03,11:1.02,12:1.10}
    if category_seasonality is None: category_seasonality = default_seasonality
    if promo_lift_by_month is None: promo_lift_by_month = {m:1.0 for m in range(1,13)}

    sl_factor = 1.0 + max(0.0, (service_level - 0.80)) * 0.25

    def k2mon(k:int)->int: return (6+k) if k<=6 else (k-6)  # 1→7월 ... 12→6월

    totals = np.zeros((mc_runs, months), dtype=float)
    for run in range(mc_runs):
        month_units = np.zeros(months, dtype=float)
        for p in personas:
            prob = p.purchase_probability_pct/100.0
            freq = np.array(p.monthly_purchase_frequency, dtype=float)
            expected = p.weight * base_units * prob * freq
            noise = np.clip(rng.lognormal(mean=0.0, sigma=0.15, size=months), 0.5, 2.0)
            expected *= noise
            month_units += expected
        for k in range(1, months+1):
            m = k2mon(k)
            mult = category_seasonality.get(m,1.0) * promo_lift_by_month.get(m,1.0)
            if k<=3: mult *= ab_lift_first3
            mult *= vr_lift_all * export_lift_all * sl_factor
            month_units[k-1] *= mult
        totals[run,:] = month_units
    return np.mean(totals, axis=0)

# -----------------------------
# 8) Smoothing / Scaling helpers
# -----------------------------
def moving_average(vec: np.ndarray, window: int) -> np.ndarray:
    if window<=1: return vec
    pad = window//2
    ext = np.pad(vec, (pad,pad), mode="edge")
    out = np.convolve(ext, np.ones(window)/window, mode="valid")
    return out[:len(vec)]

def apply_scales(monthly: np.ndarray,
                 global_scale: float,
                 sku_scale: float,
                 month_scale: Dict[int, float]) -> np.ndarray:
    s = np.ones_like(monthly) * float(global_scale) * float(sku_scale)
    for k in range(1, 13):
        s[k-1] *= float(month_scale.get(k, 1.0))
    return monthly * s

# -----------------------------
# 9) Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", type=str, required=True)              # product_info1.csv
    ap.add_argument("--sample_submission", type=str, required=True)     # sample_submission.csv
    ap.add_argument("--out_csv", type=str, required=True)

    # Backends
    ap.add_argument("--llm_backend", type=str, default="none", choices=["none","openai"])
    ap.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY",""))
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.1)

    # Simulation core
    ap.add_argument("--personas_per_product", type=int, default=60)
    ap.add_argument("--mc_runs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default="")   # "41,42,43" → 멀티시드 앙상블

    # Lifts
    ap.add_argument("--category_seasonality_json", type=str, default="")
    ap.add_argument("--promo_lift_json", type=str, default="")
    ap.add_argument("--promo_lift_per_sku_json", type=str, default="")
    ap.add_argument("--ab_json", type=str, default="")
    ap.add_argument("--vr_json", type=str, default="")
    ap.add_argument("--export_json", type=str, default="")
    ap.add_argument("--service_level", type=float, default=0.85)

    # Early boost + scales + smoothing
    ap.add_argument("--early3_boost_factor", type=float, default=1.08)  # 광고/SNS SKU 초반 3개월 가중
    ap.add_argument("--global_scale", type=float, default=1.0)
    ap.add_argument("--sku_scale_json", type=str, default="")
    ap.add_argument("--month_scale_json", type=str, default="")
    ap.add_argument("--smooth_ma", type=int, default=1)                 # 이동평균 창(3 추천)

    # Optional evaluation
    ap.add_argument("--actuals_csv", type=str, default="")
    ap.add_argument("--best_private_score", type=float, default=0.20)   # 환산점수 계산용 가정
    ap.add_argument("--score_exponent_N", type=float, default=3.0)

    args = ap.parse_args()

    log_path = os.path.splitext(args.out_csv)[0] + ".log"
    log_append(log_path, f"START v10 llm={args.llm_backend} model={args.openai_model} personas={args.personas_per_product} mc={args.mc_runs} seeds={args.seeds or args.seed}")

    # Load inputs
    prod_df = pd.read_csv(args.products)
    sub_df  = pd.read_csv(args.sample_submission)

    # Ensure floats for submission columns
    month_cols = [c for c in sub_df.columns if c.startswith("months_since_launch_")]
    for c in month_cols:
        sub_df[c] = pd.to_numeric(sub_df[c], errors="coerce").astype(float)

    # Optional JSONs
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

    ab_plan, vr_store, export_markets = {}, {}, {}
    if args.ab_json and os.path.exists(args.ab_json):
        with open(args.ab_json, "r", encoding="utf-8") as f: ab_plan = json.load(f)
    if args.vr_json and os.path.exists(args.vr_json):
        with open(args.vr_json, "r", encoding="utf-8") as f: vr_store = json.load(f)
    if args.export_json and os.path.exists(args.export_json):
        with open(args.export_json, "r", encoding="utf-8") as f: export_markets = json.load(f)

    sku_scale = {}
    if args.sku_scale_json and os.path.exists(args.sku_scale_json):
        with open(args.sku_scale_json, "r", encoding="utf-8") as f:
            sku_scale = {k: float(v) for k,v in json.load(f).items()}

    month_scale = {}
    if args.month_scale_json and os.path.exists(args.month_scale_json):
        with open(args.month_scale_json, "r", encoding="utf-8") as f:
            month_scale = {int(k): float(v) for k,v in json.load(f).items()}

    out = sub_df.copy()
    product_names = list(out["product_name"].astype(str).values)

    # Seeds for ensemble
    if args.seeds.strip():
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed]

    # ========= per product loop =========
    for p_name in product_names:
        t0 = time.time()
        print(f"[INFO] Processing {p_name} ...", flush=True)
        log_append(log_path, f"PROCESS {p_name} start")

        meta = prod_df.loc[prod_df["product_name"] == p_name].head(1)
        if len(meta)==0:
            product = ProductConfig(product_name=p_name)
        else:
            row = meta.iloc[0].to_dict()
            if all(k in row for k in ["category_level_1","category_level_2","category_level_3"]):
                cats = [str(row.get("category_level_1","")).strip(),
                        str(row.get("category_level_2","")).strip(),
                        str(row.get("category_level_3","")).strip()]
                cats = [c for c in cats if c]
                category = " > ".join(cats) if cats else None
            else:
                category = row.get("category", None)
            features = str(row.get("product_feature","")).strip() if "product_feature" in row else None
            price = float(row["price"]) if "price" in row and pd.notna(row["price"]) else None
            base_units = float(row["base_units"]) if "base_units" in row and pd.notna(row["base_units"]) else 5000.0
            product = ProductConfig(p_name, category, price, base_units, features)

        # Ensemble accumulation
        ensemble_vec = np.zeros(12, dtype=float)

        for sd in seeds:
            random.seed(sd); np.random.seed(sd)

            # Prompt
            prompt = build_single_turn_prompt(product, args.personas_per_product)

            # Personas
            if args.llm_backend=="openai":
                if not args.openai_api_key:
                    log_append(log_path, f"{p_name} API_KEY_MISSING → dummy")
                    persona_json = offline_dummy_personas(args.personas_per_product, product.category, seed=sd)
                else:
                    try:
                        persona_json = call_openai_with_tools(
                            prompt, api_key=args.openai_api_key,
                            n_personas=args.personas_per_product,
                            model=args.openai_model, temperature=args.temperature
                        )
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
                                "temperature": float(args.temperature),
                                "max_tokens": 7000
                            }
                            r = requests.post(url, headers=headers, json=payload, timeout=120); r.raise_for_status()
                            data = r.json(); content = data["choices"][0]["message"].get("content") or "{}"
                            persona_json = parse_personas_json(content)
                            log_append(log_path, f"{p_name} fallback_json_object OK")
                        except Exception as pe:
                            log_append(log_path, f"{p_name} OPENAI_FAIL → dummy : {pe}")
                            persona_json = offline_dummy_personas(args.personas_per_product, product.category, seed=sd)
            else:
                persona_json = offline_dummy_personas(args.personas_per_product, product.category, seed=sd)

            plist = persona_json.get("personas", [])
            if not isinstance(plist, list) or len(plist)==0:
                plist = offline_dummy_personas(args.personas_per_product, product.category, seed=sd)["personas"]

            # Normalize
            weights = np.array([max(0.0, float(p.get("weight", 0.0))) for p in plist], dtype=float)
            if weights.sum() <= 0: weights = np.ones(len(plist), dtype=float)
            weights = weights / weights.sum()

            personas: List[Persona] = []
            for i, pj in enumerate(plist):
                mf = np.array(pj.get("monthly_purchase_frequency", [1]*12), dtype=float)
                mf = np.maximum(mf, 0.0); 
                if mf.sum()<=0: mf[:] = 1.0
                mf = mf / mf.sum()

                attrs = dict(pj.get("attributes", {}))
                raw_aw = pj.get("attribute_weights", {})
                aw_dict = coerce_attribute_weights(raw_aw, attrs)

                personas.append(Persona(
                    name=str(pj.get("name", f"Persona_{i+1}")),
                    weight=float(weights[i]) if i<len(weights) else float(1.0/len(plist)),
                    attributes=attrs,
                    attribute_weights=aw_dict,
                    monthly_purchase_frequency=mf.tolist(),
                    purchase_probability_pct=float(pj.get("purchase_probability_pct", 30.0))
                ))

            # Early 3-month boost for ad/SNS SKUs
            if product.features and (("SNS" in product.features) or ("바이럴" in product.features) or ("광고모델" in product.features)):
                for px in personas:
                    mf = np.array(px.monthly_purchase_frequency, dtype=float)
                    mf[:3] = mf[:3] * float(args.early3_boost_factor)
                    mf = mf / mf.sum()
                    px.monthly_purchase_frequency = mf.tolist()
                log_append(log_path, f"{p_name} early3 x{args.early3_boost_factor} (seed={sd})")

            # Merge promotions
            local_promo = dict(promo_lift or {})
            if p_name in promo_lift_per_sku:
                try:
                    sku_map = {int(k): float(v) for k,v in promo_lift_per_sku[p_name].items()}
                    for m,v in sku_map.items():
                        local_promo[m] = local_promo.get(m,1.0) * v
                except Exception as e:
                    log_append(log_path, f"{p_name} promo_merge_fail {e}")

            # AB Lift
            ab_lift = 1.0
            try:
                a = ab_plan.get(p_name, ab_plan.get("default", {}))
                if isinstance(a, dict):
                    click = float(a.get("click_rate", 0.0))
                    conv  = float(a.get("conversion_rate", 0.0))
                    ab_lift = 1.0 + min(0.3, click*0.5 + conv*1.5)
            except Exception as e:
                log_append(log_path, f"{p_name} ab_parse_fail {e}")

            # VR Lift
            vr_lift = 1.0
            try:
                vrc = vr_store.get(p_name, vr_store.get("default", {}))
                if isinstance(vrc, dict):
                    vis = float(vrc.get("shelf_visibility", 1.0))
                    eye = float(vrc.get("eye_tracking_lift", 1.0))
                    vr_lift = max(0.7, min(1.5, vis*eye))
            except Exception as e:
                log_append(log_path, f"{p_name} vr_parse_fail {e}")

            # Export Lift
            export_lift = 1.0
            try:
                vals=[]
                for mk,cfg in (export_markets or {}).items():
                    if isinstance(cfg, dict) and "lift" in cfg:
                        vals.append(float(cfg["lift"]))
                if vals: export_lift = float(np.mean(vals))
            except Exception as e:
                log_append(log_path, f"{p_name} export_parse_fail {e}")

            # Simulate
            monthly = simulate_monthly_units(
                personas=personas, base_units=product.base_units,
                category_seasonality=category_seasonality,
                promo_lift_by_month=local_promo,
                ab_lift_first3=ab_lift, vr_lift_all=vr_lift,
                export_lift_all=export_lift, service_level=args.service_level,
                mc_runs=args.mc_runs, seed=sd
            )

            # Scaling
            sku_mul = float(sku_scale.get(p_name, 1.0))
            monthly = apply_scales(monthly, args.global_scale, sku_mul, month_scale)

            # Smoothing(장기 안정성 ↑ → Private Score 개선 방향)
            if args.smooth_ma>1:
                monthly = moving_average(monthly, args.smooth_ma)

            ensemble_vec += monthly

        # Average over seeds
        ensemble_vec /= float(len(seeds))

        # Fill submission row
        for k in range(12):
            col = f"months_since_launch_{k+1}"
            if col in out.columns:
                out.loc[out["product_name"]==p_name, col] = float(round(max(0.0, ensemble_vec[k]), 2))

        dt = time.time() - t0
        print(f"[INFO] Done {p_name} ({dt:.1f}s)", flush=True)
        log_append(log_path, f"PROCESS {p_name} done {dt:.1f}s base_units={product.base_units}")

    # Save submission
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[OK] submission saved: {args.out_csv}")
    log_append(log_path, f"SAVED {args.out_csv}")

    # Optional evaluation on provided actuals
    if args.actuals_csv and os.path.exists(args.actuals_csv):
        act = pd.read_csv(args.actuals_csv)
        merged = pd.merge(out, act, on="product_name", suffixes=("_pred","_true"))
        smapes=[]
        for _, r in merged.iterrows():
            yp, yt = [], []
            for k in range(12):
                yp.append(float(r[f"months_since_launch_{k+1}_pred"]))
                yt.append(float(r[f"months_since_launch_{k+1}_true"]))
            smapes.append(smape(np.array(yt), np.array(yp)))
        overall = float(np.mean(smapes)) if smapes else float("nan")
        conv_score = private_leaderboard_score(args.best_private_score, overall, args.score_exponent_N)
        print(f"[INFO] SMAPE over 12 months: {overall:.4f} | Converted Private Score~: {conv_score:.3f}")
        log_append(log_path, f"SMAPE {overall:.4f} Converted {conv_score:.3f}")

if __name__ == "__main__":
    main()
