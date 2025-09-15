"""
Dongwon - KR Persona Generator (Single-turn LLM, 10+ attrs, Internal Taxonomy)
- 한국 지역(region_kr), 자취(living_alone), 결혼(married) 포함
- 회사 내부 분류(채널 코드, 소득 구간, 교육 레벨) 강제
- product_info.csv를 RAG 요약으로 투입 → 현실성 강화
- 싱글턴 프롬프트 1회로 N개 JSON 생성 (대회 규정 준수)
- 사후 검증/보정: 스키마/값 범위/내부 코드 매핑/다양성 점수

필요 패키지: openai, python-dotenv, pandas, numpy
실행: python app_persona_kr.py
결과: personas.json, personas_preview.csv
"""

import os, re, json, random
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# =========================[ 설정 ]=========================
N_PERSONAS = 24                   # 생성할 페르소나 수
PRODUCT_INFO = "product_info.csv" # 첨부하신 최신 파일명
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# LLM 세팅
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 가 없습니다. .env 를 확인하세요.")
client = OpenAI(api_key=OPENAI_API_KEY)

# FT 모델이 있으면 사용, 없으면 4o-mini
MODEL = os.getenv("FT_MODEL_ID") or "gpt-4o-mini"

# =========================[ 내부 분류(회사 표준) ]=========================
# 1) 채널 코드 (예시)
CHANNEL_CODE_MAP = {
    "ON": "온라인(자사몰/오픈마켓/퀵커머스)",
    "OF": "오프라인(대형마트/편의점/SSM)",
    "OM": "옴니(온라인+오프라인)"
}
CHANNEL_ALLOWED = set(CHANNEL_CODE_MAP.keys())

# 2) 소득 구간 (세전/월 or 연 기준 내부 표준 예시 — 구간 라벨만 사용)
#    내부 보고서 표준 라벨을 강제하도록 함
INCOME_BUCKETS_KR = [
    "월<200만원", "월200-300", "월300-400", "월400-600", "월600-800", "월800-1000", "월>1000",
    # 또는 연 기준이 필요하면 아래로 교체
    # "연<2000만원", "연2000-3000", "연3000-4000", "연4000-6000", "연6000-8000", "연8000-1억", "연>1억"
]

# 3) 교육 레벨 (내부 표준)
EDU_LEVELS_KR = [
    "고졸이하", "전문대", "대졸", "석사", "박사"
]

# 4) 한국 지역 (광역)
REGION_KR = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산",
    "세종", "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
]

# =========================[ 제품 속성 요약(RAG) ]=========================
# 제품 속성 매핑(파일 컬럼명에 맞게 필요 시 수정)
ATTR_MAP = {
    "product_name": "product_name",
    "price":        "price",
    "promo":        "promo_flag",
    "channel":      "channel",
    "size":         "size_g",
    "brand_tier":   "brand_tier",
    "category":     "category",
}

def summarize_product_info_for_prompt(df: pd.DataFrame) -> str:
    """
    product_info.csv를 요약해 프롬프트에 삽입 (토큰 절약)
    - 문자형: 상위 level 빈도
    - 수치형: 요약통계
    """
    lines = []
    for label, col in ATTR_MAP.items():
        if label == "product_name":
            continue
        if col not in df.columns:
            continue
        if df[col].dtype == "O":
            vc = df[col].astype(str).value_counts().head(10)
            lines.append(f"{col}: " + ", ".join([f"{k}({v})" for k,v in vc.items()]))
        else:
            desc = df[col].describe().to_dict()
            lines.append(
                f"{col}: min={desc.get('min')}, q25={desc.get('25%')}, "
                f"median={desc.get('50%')}, q75={desc.get('75%')}, max={desc.get('max')}"
            )
    return "\n".join(lines)

# =========================[ 스키마 정의 ]=========================
# (논문 인구통계 + 한국 로컬 + 민감도/행동 + 메모) → 기본만 18+개
REQUIRED_KEYS = [
    # 논문 인구통계 축
    "gender", "age", "race_ethnicity", "politics", "income", "education",
    # 한국 로컬
    "region_kr", "living_alone", "married",
    # 기본 행동/채널/가치
    "channel_code",            # 내부 코드: ON/OF/OM
    "price_sensitivity", "promotion_sensitivity", "brand_loyalty",
    "innovation_seeking", "review_dependence", "sustainability_preference",
    "risk_aversion", "baseline_purchase_frequency",
    # 메모
    "product_fit_note"
]

# 확장 속성(더 풍부한 이질성 표현) — 10개 이상 자유 확장
OPTIONAL_KEYS = [
    "health_consciousness", "eco_packaging_preference", "package_size_preference",
    "taste_preference", "time_pressure", "convenience_store_usage", "social_influence",
    "kids_presence", "cooking_frequency", "dietary_restriction", "digital_literacy",
    "ad_avoidance", "coupon_usage", "brand_switch_tendency", "novelty_fatigue",
    "outside_option_bias", "bulk_buying_preference", "shelf_life_importance",
    "freshness_importance", "delivery_sensitivity", "store_loyalty", "ugc_creation_level"
]

# 0~1 클램프 대상
CLAMP01_KEYS = {
    "price_sensitivity","promotion_sensitivity","brand_loyalty","innovation_seeking",
    "review_dependence","sustainability_preference","risk_aversion",
    "baseline_purchase_frequency","health_consciousness","eco_packaging_preference",
    "package_size_preference","time_pressure","convenience_store_usage","social_influence",
    "digital_literacy","ad_avoidance","coupon_usage","brand_switch_tendency","novelty_fatigue",
    "outside_option_bias","bulk_buying_preference","shelf_life_importance","freshness_importance",
    "delivery_sensitivity","store_loyalty","ugc_creation_level"
}

def _clamp01(x, default=0.5):
    try:
        return float(np.clip(float(x), 0.0, 1.0))
    except Exception:
        return default

def _validate_and_fix(p: Dict[str,Any]) -> Dict[str,Any]:
    """내부 코드/값 범위/누락 채우기 등 사후 보정"""
    # 기본값
    defaults = {
        "gender":"other", "age":"25-34", "race_ethnicity":"unspecified",
        "politics":"moderate", "income":"월300-400", "education":"대졸",
        "region_kr":"서울", "living_alone": False, "married": False,
        "channel_code":"OM", "product_fit_note":"general fit",
        "baseline_purchase_frequency": 0.2,
        "price_sensitivity":0.5, "promotion_sensitivity":0.5, "brand_loyalty":0.5,
        "innovation_seeking":0.5, "review_dependence":0.5, "sustainability_preference":0.5,
        "risk_aversion":0.5
    }
    for k,v in defaults.items():
        if k not in p:
            p[k] = v

    # 내부 코드 강제
    # channel_code → ON/OF/OM 중 하나
    code_in = str(p.get("channel_code","OM")).upper()
    if code_in not in CHANNEL_ALLOWED:
        # 자연어로 온 경우 간단 매핑
        raw = str(code_in).lower()
        if "online" in raw or "온라인" in raw: p["channel_code"] = "ON"
        elif "offline" in raw or "오프라인" in raw or "마트" in raw or "편의점" in raw: p["channel_code"] = "OF"
        else: p["channel_code"] = "OM"

    # income을 내부 구간 라벨 중 하나로 스냅
    if p.get("income") not in INCOME_BUCKETS_KR:
        # 간단 휴리스틱
        txt = str(p.get("income","")).replace(" ", "")
        mapped = None
        for bucket in INCOME_BUCKETS_KR:
            rng = bucket.replace(" ", "")
            if any(s in txt for s in rng.split("-")):
                mapped = bucket; break
        p["income"] = mapped or "월300-400"

    # education 스냅
    if p.get("education") not in EDU_LEVELS_KR:
        txt = str(p.get("education","")).lower()
        if "고" in txt: p["education"] = "고졸이하"
        elif "전문" in txt: p["education"] = "전문대"
        elif "석" in txt: p["education"] = "석사"
        elif "박" in txt: p["education"] = "박사"
        else: p["education"] = "대졸"

    # region_kr 스냅
    if p.get("region_kr") not in REGION_KR:
        # 간단한 포함 매칭
        txt = str(p.get("region_kr","")).strip()
        found = next((r for r in REGION_KR if r in txt), None)
        p["region_kr"] = found or "서울"

    # Boolean 보정
    def to_bool(v):
        if isinstance(v, bool): return v
        s = str(v).strip().lower()
        return s in ["1","y","yes","true","t","자취","기혼","married","혼인"]

    p["living_alone"] = bool(p.get("living_alone", False))
    p["married"] = bool(p.get("married", False))

    # 0~1 클램프
    for k in CLAMP01_KEYS:
        if k in p:
            p[k] = _clamp01(p[k])

    return p

def _vec_for_diversity(p: Dict[str,Any]) -> np.ndarray:
    """다양성 점수용 간단 임베딩"""
    vec_keys = list(CLAMP01_KEYS) + ["living_alone","married"]
    v = []
    for k in vec_keys:
        if k in ["living_alone","married"]:
            v.append(1.0 if bool(p.get(k, False)) else 0.0)
        else:
            v.append(float(p.get(k, 0.5)))
    return np.array(v, dtype=float)

def _diversity_score(personas: List[Dict[str,Any]]) -> float:
    if len(personas) < 2: return 0.0
    V = np.stack([_vec_for_diversity(p) for p in personas])
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    dists = []
    for i in range(len(V)):
        for j in range(i+1, len(V)):
            dists.append(1 - float(V[i] @ V[j]))
    return float(np.mean(dists)) if dists else 0.0

# =========================[ LLM 싱글턴 호출 ]=========================
def ask_single_turn(prompt: str, temperature=0.9, max_tokens=6000) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()
def build_persona_prompt(n: int, product_summary: str) -> str:
    """
    강화 프롬프트: 맛/저당/광고 민감도 분포를 '강제'하여 다양성 확장.
    내부 표준(채널/소득/교육) 라벨을 그대로 사용.
    """
    # 내부 표준 라벨(파일 상단 선언된 상수 사용 가정)
    from __main__ import CHANNEL_CODE_MAP, INCOME_BUCKETS_KR, EDU_LEVELS_KR, REGION_KR
    ch_codes = ", ".join([f"{k}={v}" for k,v in CHANNEL_CODE_MAP.items()])
    income_str = ", ".join(INCOME_BUCKETS_KR)
    edu_str = ", ".join(EDU_LEVELS_KR)
    region_str = ", ".join(REGION_KR)

    return f"""
당신은 FMCG 시장조사 전문가입니다. 한국 시장을 반영한 소비자 페르소나 {n}개를
JSON 배열로만 생성하세요(코드블록/설명 금지).

[필수 스키마]
- snake_case 키, 각 페르소나는 10개 이상 속성
- 필수 키:
  gender, age, race_ethnicity, politics, income, education,
  region_kr, living_alone, married,
  channel_code,                        # 내부 코드: ON/OF/OM
  price_sensitivity, promotion_sensitivity, brand_loyalty,
  innovation_seeking, review_dependence, sustainability_preference,
  risk_aversion, baseline_purchase_frequency,
  taste_preference,                    # "spicy" | "vanilla" | "latte" | "neutral"
  low_sugar_preference,                # 0~1 (저당 선호)
  product_fit_note, name

[라벨/값 제약]
- channel_code ∈ {{ON, OF, OM}} (설명 금지)
- income ∈ [{income_str}]
- education ∈ [{edu_str}]
- region_kr ∈ [{region_str}]
- 민감도/확률형 변수는 0~1 (소수점)

[분포 강제 규칙 — 꼭 지키세요]
- taste_preference 할당(총 {n}명 기준):
  - spicy:    ~{max(1, n//6)}명
  - vanilla:  ~{max(1, n//6)}명
  - latte:    ~{max(1, n//6)}명
  - neutral:  나머지
- low_sugar_preference:
  - 상위 그룹(0.7~1.0): ~{max(1, n//4)}명
  - 중간(0.4~0.7):      ~{max(1, n//2)}명
  - 하위(0.0~0.4):      나머지
- promotion_sensitivity:
  - 고(0.7~1.0)와 저(0.0~0.3)가 각각 전체의 ≥{max(1, n//5)}명 포함되도록
- price_sensitivity(가격 민감)과 brand_loyalty(브랜드 충성)는 역상관이 되도록
  (극단 값끼리 동시 출현 금지; 다양한 조합 배치)
- channel_code 분포: ON≈OF≈OM(±1명 허용)
- married/living_alone는 True/False가 각각 전체의 ≥{max(1, n//4)}명 포함되도록
- 소득/교육/연령/지역은 최대한 고르게(과도한 쏠림 금지)
[제품 속성 요약(RAG)]
{product_summary}

오직 JSON만 출력하세요.
"""

# def build_persona_prompt(n: int, product_summary: str) -> str:
#     """
#     싱글턴 프롬프트: 내부 분류 준수/한국 로컬 속성/최소 10개 이상/JSON only
#     """
#     req = ", ".join(REQUIRED_KEYS)
#     opt = ", ".join(OPTIONAL_KEYS)
#     ch_codes = ", ".join([f"{k}={v}" for k,v in CHANNEL_CODE_MAP.items()])
#     income_str = ", ".join(INCOME_BUCKETS_KR)
#     edu_str = ", ".join(EDU_LEVELS_KR)
#     region_str = ", ".join(REGION_KR)

#     return f"""
# 당신은 FMCG 시장조사 전문가입니다. 한국 시장을 반영한 소비자 페르소나 {n}개를
# JSON 배열로만 생성하세요(코드블록/설명 금지).

# 요구사항(반드시 준수):
# - 각 페르소나는 최소 10개 이상의 속성을 가져야 하며 snake_case 키로 작성합니다.
# - 필수 키: {req}
# - 선택 키(가능한 많이 포함): {opt}
# - 한국 지역(region_kr)은 다음 중 하나만 사용: [{region_str}]
# - 결혼 여부(married)와 자취 여부(living_alone)는 불리언으로 표현하세요.
# - 채널은 내부 코드 channel_code 로만 표기: [{ch_codes}]  # 예: "ON","OF","OM"
# - 소득(income)은 내부 표준 구간 라벨 중 하나만 사용: [{income_str}]
# - 교육(education)은 내부 표준 라벨 중 하나만 사용: [{edu_str}]
# - 가격/프로모/채널/지속가능성/리뷰의존/혁신성향 등의 민감도는 0~1 범위의 실수.
# - 각 페르소나에는 "name"을 포함(예: "건강중시 프리미엄형").
# - 집단 다양성 극대화(성별/연령/소득/교육/정치/채널/민감도 조합이 중복되지 않도록 설계).

# [제품 속성 요약(RAG)]
# {product_summary}

# 오직 JSON만 출력하세요.
# """

def make_personas_kr(n: int = N_PERSONAS) -> List[Dict[str,Any]]:
    # 제품 요약
    if not os.path.exists(PRODUCT_INFO):
        raise FileNotFoundError(f"{PRODUCT_INFO} 를 찾을 수 없습니다.")
    dfp = pd.read_csv(PRODUCT_INFO)
    summary = summarize_product_info_for_prompt(dfp)

    prompt = build_persona_prompt(n, summary)
    txt = ask_single_turn(prompt, temperature=0.95, max_tokens=7000)

    # JSON 배열 추출
    m = re.search(r"\[.*\]", txt, re.S)
    raw = m.group(0) if m else txt
    personas = json.loads(raw)

    # 보정/검증
    cleaned = []
    for p in personas:
        p = _validate_and_fix(p)
        if "name" not in p:
            p["name"] = f"persona_{len(cleaned)+1}"
        # 최소 10개 속성 (name 포함하면 보통 19+)
        if len(p.keys()) >= 10:
            cleaned.append(p)

    # 다양성 체크 → 필요 시 약간의 분산(로컬 후처리, 재질의 없음)
    div = _diversity_score(cleaned)
    if div < 0.15:
        for p in cleaned:
            for k in CLAMP01_KEYS:
                if k in p:
                    p[k] = float(np.clip(p[k] + np.random.normal(0, 0.03), 0, 1))

    return cleaned[:n]

# =========================[ 실행부 ]=========================
if __name__ == "__main__":
    personas = make_personas_kr(N_PERSONAS)
    print(f"[INFO] personas generated: {len(personas)}, diversity={_diversity_score(personas):.3f}")

    # 저장: JSON + 미리보기 CSV
    with open("personas.json", "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)

    # 주요 필드만 csv 미리보기
    cols = [
        "name","gender","age","race_ethnicity","politics","income","education",
        "region_kr","living_alone","married","channel_code",
        "price_sensitivity","promotion_sensitivity","brand_loyalty","innovation_seeking",
        "review_dependence","sustainability_preference","risk_aversion",
        "baseline_purchase_frequency","product_fit_note"
    ]
    # 없는 컬럼은 건너뛰고 출력
    rows = []
    for p in personas:
        rows.append({k: p.get(k, "") for k in cols})
    pd.DataFrame(rows).to_csv("personas_preview.csv", index=False, encoding="utf-8-sig")

    print("[OK] Saved personas.json, personas_preview.csv")
