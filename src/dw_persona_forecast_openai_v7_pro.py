#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DW Persona Forecast — v7_pro (1등 목표 튜닝)
- 화자/역할 프롬프트: 동원참치 마케팅 → 본사 재고관리 30년차 부장 관점
- B2C + B2B(해외/국내 기업, 공공기관) 구매자 세그먼트 반영
- OpenAI function-calling 강제(JSON 깨짐 방지)
- product_info 확장(price, base_units); base_units로 스케일링
- 프로모션 승수(promo_lift.json), (선택) 카테고리 계절성 파일
- 최적 권장: personas_per_product=80, mc_runs=800
"""

import os, sys, json, time, argparse, random, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()  # OPENAI_API_KEY

# ---------------- Metric ----------------
def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom > 0
    value = np.zeros_like(denom, dtype=float)
    value[mask] = 2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
    return np.mean(value) * 100.0

# --------------- Data classes -----------
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

# --------------- Helpers ----------------
def norm(s: Optional[str]) -> str:
    return (s or "").lower()

def get_category_context(category: Optional[str]) -> str:
    c = norm(category); ctx = []
    ctx.append("- 공통: 설(1-2월), 추석(9-10월), 연말(12월) 수요 상승 가능성.")
    ctx.append("- 11월 코리아세일페스타·블랙프라이데이 등 판촉 강화.")
    ctx.append("- 학사 일정: 3·9월 개학/개강, 1-2·7-8월 방학 → 가정 내 소비 패턴 변화.")
    if any(k in c for k in ["yogurt","요거트","그릭","건강","헬스","발효유"]):
        ctx += ["- 다이어트/단백질 트렌드(1월, 5-6월) 수요↑.", "- 냉장 신선·고단백 메시지 중요."]
    if any(k in c for k in ["음료","drink","커피","coffee","rtd","cup"]):
        ctx += ["- 기온 민감: 7-8월 피크.", "- 편의점/자판기·1+1 프로모션 영향 큼."]
    if any(k in c for k in ["간편식","즉석","축산캔","rte","스낵","라면"]):
        ctx += ["- 방학/시험/직장인 수요 스윙.", "- 온라인 장보기/구독 재구매."]
    if any(k in c for k in ["참치","캔","통조림","오일","조미료","액상조미료","참기름","tuna"]):
        ctx += ["- 원자재/운임 변동 → 가격/프로모션 민감.", "- 명절·선물세트 스파이크."]
    return "\n".join(ctx)

def tune_persona_ranges_by_category(category: Optional[str]) -> Dict[str, Any]:
    c = norm(category)
    cfg = {"prob_range": (18, 55), "bias": {}}
    if any(k in c for k in ["요거트","yogurt","발효유","건강","헬스"]):
        cfg["prob_range"] = (22, 60); cfg["bias"] = {"price_sensitivity":("down",0.1),"brand_loyalty":("up",0.1)}
    elif any(k in c for k in ["음료","drink","커피","coffee","rtd","cup"]):
        cfg["prob_range"] = (25, 65); cfg["bias"] = {"promotion_sensitivity":("up",0.1),"channel_preference":("up_ON",0.05)}
    elif any(k in c for k in ["간편식","즉석","rte","라면","스낵","축산캔"]):
        cfg["prob_range"] = (20, 58); cfg["bias"] = {"review_dependence":("up",0.08)}
    elif any(k in c for k in ["참치","캔","통조림","오일","조미료","액상조미료","참기름"]):
        cfg["prob_range"] = (18, 52); cfg["bias"] = {"price_sensitivity":("up",0.08),"brand_loyalty":("up",0.06)}
    return cfg

# -------- JSON fallback (safety) --------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I); s = re.sub(r"\s*```$", "", s)
    return s.strip()

def repair_json_string(s: str) -> str:
    s = _strip_code_fences(s)
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start: s = s[start:end+1]
    s = re.sub(r'\}\s*\{', '},{', s)
    s = re.sub(r'([0-9\]\}])\s*("(?=[^"]+"\s*:))', r'\1,\2', s)
    s = re.sub(r",\s*([\}\]])", r"\1", s)
    try: json.loads(s); return s
    except: pass
    s = re.sub(r"(\s*)'([^']+)'\s*:", r'\1"\2":', s)
    s = re.sub(r':\s*\'([^\'\\]*(?:\\.[^\'\\]*)*)\'', r': "\1"', s)
    s = re.sub(r",\s*([\}\]])", r"\1", s)
    return s

def parse_personas_json(raw_text: str) -> dict:
    raw = _strip_code_fences(raw_text)
    try: return json.loads(raw)
    except: pass
    fixed = repair_json_string(raw_text)
    return json.loads(fixed)

# --------------- Prompt/schema -----------
def build_persona_schema(n_personas: int) -> dict:
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
                        "monthly_purchase_frequency":{
                            "type":"array","minItems":12,"maxItems":12,"items":{"type":"number"}
                        }
                    },
                    "required":["name","weight","attributes","attribute_weights","purchase_probability_pct","monthly_purchase_frequency"],
                    "additionalProperties":True
                }
            }
        },
        "required":["personas"],"additionalProperties":False
    }

def normalize_category_guess(product: ProductConfig) -> str:
    text = f"{product.product_name} {product.features or ''}".lower()
    if any(k in text for k in ["참기름","sesame","goma","참유"]): return "오일/참기름"
    return product.category or ""

def build_single_turn_prompt(product: ProductConfig, n_personas: int) -> str:
    """
    화자: 동원참치 마케팅 부서 경력 후 본사 재고관리로 이동한 30년차 부장.
    목표: 1등. 재고회전/프로모션/광고모델/납품계약까지 고려한 정교한 수요예측.
    B2C와 B2B(국내/해외 기업, 공공기관) 구매자 모두 포함.
    """
    cat_hint = normalize_category_guess(product) or product.category
    cat_ctx = get_category_context(cat_hint)
    features_txt = f"- 제품 특성: {product.features}" if product.features else "- 제품 특성: (제공 없음)"

    guidance = f"""
당신은 동원그룹에서 마케팅을 거쳐 본사 재고 관리 부서로 이동한 30년차 부장입니다. 목표는 리더보드 1위를 달성하는 실전형 예측입니다.
아래 제품에 대해 '싱글 턴(single turn)'으로 **정확히 {n_personas}명**의 페르소나를 생성하세요.
- **B2C와 B2B(국내/해외 기업, 공공기관 조달 담당자)**를 모두 포함합니다.
- 각 페르소나는 최소 **10개 이상의 속성**과 **속성별 영향 가중치(0~1)**를 포함합니다.
- **월별 구매 빈도 패턴(12개월: 7월→다음해 6월, 합=1.0)**과 **구매 확률(%)**을 제시합니다.
- **광고모델·SNS·저당/락토프리·명절/세일페스타·학사일정** 등 맥락을 반영합니다.
- B2B의 경우 **납품/입찰 주기(분기/반기/연말 집행), 대량구매(벌크), 계약 유지·이탈 확률**을 속성과 빈도 패턴에 반영합니다.

제품 정보:
- 제품명: {product.product_name}
- 카테고리: {product.category or "미상"}
- 가격(원): {product.price if product.price is not None else "미상"}
{features_txt}

카테고리 맥락:
{cat_ctx}

세그먼트 가이드(권장 분포):
- buyer_type ∈ {{B2C, B2B_기업_국내, B2B_기업_해외, B2B_공공}}.
- 기본 비중: B2C 70~85%, B2B_기업 10~20%(국내/해외 합), B2B_공공 5~10%.
- B2B는 **분기/연말 집중(1·4·7·10월, 12월)**, B2C는 **시즌·프로모션**에 민감.

출력은 **함수 호출 인자(personas)**로만 반환하세요. 설명 텍스트를 쓰지 마세요.
""".strip()

    # features 기반 보강
    ftxt = product.features or ""
    if "광고모델" in ftxt:
        guidance += "\n- 광고모델 효과: 20~30대 여성·수도권에서 구매확률 1.2배 가중."
    if "SNS" in ftxt or "바이럴" in ftxt:
        guidance += "\n- SNS 바이럴: 출시 후 1~3개월 구매 빈도 1.15배."
    if "저당" in ftxt or "락토프리" in ftxt:
        guidance += "\n- 저당/락토프리: 20~40대 건강 관심층 재구매율 1.1배."

    guidance += "\n반드시 도구 호출로만(personas 인자) 답하세요."
    return guidance

# --------------- OpenAI call ------------
def call_openai_with_tools(prompt: str, api_key: str,
                           n_personas: int,
                           model: str = "gpt-4o-mini",
                           temperature: float = 0.2,
                           timeout: int = 120,
                           max_retries: int = 2) -> dict:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type":"application/json","Authorization":f"Bearer {api_key}"}
    tools = [{
        "type":"function",
        "function":{
            "name":"submit_personas",
            "description":"Return personas as strict JSON only.",
            "parameters":build_persona_schema(n_personas)
        }
    }]
    payload = {
        "model": model,
        "temperature": float(temperature),
        "messages": [
            {"role":"system","content":"Return output ONLY via the function call `submit_personas`. Do not write prose."},
            {"role":"user","content": prompt}
        ],
        "tools": tools,
        "tool_choice": {"type":"function","function":{"name":"submit_personas"}},
        "max_tokens": 7000
    }
    last_err = None
    for _ in range(max_retries+1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            tcalls = data["choices"][0]["message"].get("tool_calls") or []
            if not tcalls:
                content = data["choices"][0]["message"].get("content") or ""
                return parse_personas_json(content)
            args_str = tcalls[0]["function"].get("arguments","{}")
            return json.loads(args_str)
        except Exception as e:
            last_err = e; time.sleep(1.2)
    raise RuntimeError(f"OpenAI tool-call 실패: {last_err}")

# ------- Offline dummy (B2C/B2B 포함) ----
def offline_dummy_personas(n: int, category: Optional[str], seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed); np_rng = np.random.default_rng(seed)
    personas = []
    w = np.array([rng.random()+0.1 for _ in range(n)], dtype=float); w = w/w.sum()
    cfg = tune_persona_ranges_by_category(category); p_lo, p_hi = cfg["prob_range"]; bias = cfg["bias"]

    # 권장 분포: B2C 75%, B2B_기업 18%, B2B_공공 7% (대략)
    space = ["B2C"]*75 + ["B2B_기업_국내"]*9 + ["B2B_기업_해외"]*9 + ["B2B_공공"]*7

    for i in range(n):
        buyer_type = rng.choice(space)
        age = rng.randint(22, 65) if buyer_type=="B2C" else rng.randint(28, 60)
        ch_pref = rng.choice(["ON","OFF","MIXED"])

        attrs = {
            "buyer_type": buyer_type,
            "age": age,
            "gender": rng.choice(["male","female"]) if buyer_type=="B2C" else "n/a",
            "job_title": rng.choice(["담당","대리","과장","차장","부장"]) if buyer_type!="B2C" else "n/a",
            "org_type": ("개인" if buyer_type=="B2C" else ("기업" if "기업" in buyer_type else "공공")),
            "income_band": rng.choice(["<2M","2-4M","4-6M","6-8M","8M+"]) if buyer_type=="B2C" else "n/a",
            "region_kr": rng.choice(["서울","경기","인천","부산","대구","광주","대전","울산","세종","강원","충북","충남","전북","전남","경북","경남","제주"]),
            "channel_preference": ch_pref,
            "price_sensitivity": round(rng.random(),3),
            "promotion_sensitivity": round(rng.random(),3),
            "brand_loyalty": round(rng.random(),3),
            "innovation_seeking": round(rng.random(),3),
            "review_dependence": round(rng.random(),3),
            "sustainability_preference": round(rng.random(),3),
            # B2B 전용 속성
            "contract_cycle": rng.choice(["월별","분기","반기","연말"]) if buyer_type!="B2C" else "n/a",
            "bulk_purchase": rng.choice([0,1]) if buyer_type!="B2C" else 0,
            "tender_seasonality": "강함" if buyer_type=="B2B_공공" else ("보통" if "기업" in buyer_type else "약함"),
            "preferred_pack": rng.choice(["소포장","중포장","대포장"]),
            "risk_aversion": round(rng.random(),3),
        }

        # bias 반영
        def clamp01(x): return float(max(0.0, min(1.0, x)))
        for k,(direction,delta) in bias.items():
            if k in attrs and isinstance(attrs[k], (int,float)):
                attrs[k] = clamp01(float(attrs[k]) + (delta if direction=="up" else -delta))
        if bias.get("channel_preference") == ("up_ON",0.05) and attrs["channel_preference"]!="ON":
            if rng.random()<0.25: attrs["channel_preference"]="ON"

        # 속성 가중치
        keys = list(attrs.keys()); aw = np_rng.random(len(keys)); aw = aw/aw.sum()

        # 월별 패턴
        m = np.abs(np_rng.normal(loc=1.0, scale=0.2, size=12))
        # B2B는 분기·연말 벌크 집중(1,4,7,10,12월)
        if buyer_type!="B2C":
            bumps = {1:1.3,4:1.25,7:1.25,10:1.25,12:1.4}
            for idx in range(12):
                cal_m = (6+(idx+1)) if (idx+1)<=6 else ((idx+1)-6)  # 1→7월 … 12→6월
                m[idx] *= bumps.get(cal_m, 1.0)
        m = m/m.sum()

        # B2B는 확률 낮고 변동 큼, B2C는 중간~높음
        if buyer_type=="B2C":
            prob = rng.uniform(max(p_lo, 24), min(p_hi, 62))
        elif buyer_type=="B2B_공공":
            prob = rng.uniform(14, 36)
        else:  # B2B 기업
            prob = rng.uniform(16, 42)

        personas.append({
            "name": f"Persona_{i+1}_{buyer_type}",
            "weight": float(round(w[i],6)),
            "attributes": attrs,
            "attribute_weights": {k: float(round(v,4)) for k,v in zip(keys, aw)},
            "purchase_probability_pct": float(round(prob,2)),
            "monthly_purchase_frequency": [float(round(x,6)) for x in m.tolist()]
        })
    return {"personas": personas}

# --------------- Simulation --------------
def simulate_monthly_units(personas: List[Persona],
                           base_units: float = 5000.0,
                           category_seasonality: Optional[Dict[int, float]] = None,
                           promo_lift_by_month: Optional[Dict[int, float]] = None,
                           mc_runs: int = 800,
                           seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    months = 12
    default_seasonality = {1:1.05,2:1.08,3:0.98,4:0.97,5:0.99,6:0.98,7:1.00,8:1.02,9:1.07,10:1.03,11:1.02,12:1.10}
    if category_seasonality is None: category_seasonality = default_seasonality
    if promo_lift_by_month is None: promo_lift_by_month = {m:1.0 for m in range(1,13)}

    def k_to_calendar_month(k: int) -> int:  # 1→7 … 12→6
        return (6 + k) if k <= 6 else (k - 6)

    totals = np.zeros((mc_runs, months), dtype=float)
    for run in range(mc_runs):
        month_units = np.zeros(months, dtype=float)
        for p in personas:
            prob = p.purchase_probability_pct / 100.0
            freq = np.array(p.monthly_purchase_frequency, dtype=float)
            expected = p.weight * base_units * prob * freq
            # 잡음: B2B(공공/기업)는 변동성 상향
            sigma = 0.15
            if str(p.attributes.get("buyer_type","")).startswith("B2B"):
                sigma = 0.22 if p.attributes.get("buyer_type")=="B2B_공공" else 0.18
            noise = np.clip(rng.lognormal(mean=0.0, sigma=sigma, size=months), 0.5, 2.2)
            expected *= noise
            month_units += expected
        for k in range(1, months+1):
            cal_m = k_to_calendar_month(k)
            month_units[k-1] *= category_seasonality.get(cal_m,1.0) * promo_lift_by_month.get(cal_m,1.0)
        totals[run,:] = month_units
    return np.mean(totals, axis=0)

# -------------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", type=str, required=True)
    ap.add_argument("--sample_submission", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--llm_backend", type=str, default="openai", choices=["none","openai"])
    ap.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY",""))
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--personas_per_product", type=int, default=80)  # ★ 권장값
    ap.add_argument("--mc_runs", type=int, default=800)             # ★ 권장값
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--category_seasonality_json", type=str, default="")
    ap.add_argument("--promo_lift_json", type=str, default="")
    ap.add_argument("--actuals_csv", type=str, default="")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    prod_df = pd.read_csv(args.products)
    sub_df  = pd.read_csv(args.sample_submission)
    month_cols = [c for c in sub_df.columns if c.startswith("months_since_launch_")]
    for c in month_cols:
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
