"""
predict_pipeline_v2.py
──────────────────────────────────────────────────────────────────────────────
Dongwon x KAIST 대회 (텍스트 기반) 제출 파이프라인 v2

입력 (새 스키마에 맞춤)
- personas.json             : app_persona_kr.py 결과 (싱글턴, 10개+ 속성)
- product_info.csv          : [product_name, product_feature, category_level_1,2,3]
- sample_submission.csv     : 제출 템플릿
- (선택) early_actuals.csv  : 출시 후 1~2개월 실판매 (베이지안 보정용)

출력
- submission.csv            : 제출 형식 (product_name + months_since_launch_1..12)

개선점 (v1 → v2)
1) price/promo/channel 등 수치 컬럼 없이도 작동하도록 설계.
2) product_feature에서
   - 건강/단백질/저나트륨/락토프리/매콤/참기름/프리미엄 등 키워드를 태깅
   - “6-8월 광고” 같은 월 정보를 파싱하여 제품별 월별 프로모 가중치 반영
3) LLM 옵션:
   - 싱글턴 1회로 15개 제품을 구조화(JSON) (attrs + ad_months + inferred flags)
   - 키가 없으면 휴리스틱 키워드 파서로 대체
4) 계층형 베이지안 보정:
   - 카테고리(최상위 L1) × 제품 두 단계의 감마 승수로 스케일 보정
5) 안전한 제출 파이프라인 + 매우 상세한 로깅

──────────────────────────────────────────────────────────────────────────────
"""

import os, re, json, math, random
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ============== 설정 ==============
PERSONA_JSON         = "personas.json"
PRODUCT_INFO_CSV     = "product_info.csv"      # 현재 제공 스키마
SUBMISSION_TEMPLATE  = "sample_submission.csv"
EARLY_ACTUALS_CSV    = "early_actuals.csv"     # 선택

OUTPUT_CSV           = "submission.csv"

USE_LLM              = True    # .env 키 없으면 자동 False
MODEL                = os.getenv("FT_MODEL_ID") or "gpt-4o-mini"
SEED                 = 42
random.seed(SEED); np.random.seed(SEED)

# 시장/보정 파라미터
BASE_TAM             = 120_000     # 월간 잠재 수요(초깃값) — 조정 가능
BASE_OUTSIDE         = 0.45        # 비구매 비율 평균
A0_CAT, B0_CAT       = 0.5, 0.5    # 카테고리(상위) 감마 사전 (작게 → 민감)
A0_PRD, B0_PRD       = 1.0, 1.0    # 제품(하위) 감마 사전

# L1 카테고리별 기본 시즌성(가벼운 상식 규칙; 필요시 내부 데이터로 교체)
CATEGORY_SEASON_RULES = {
    # 우유/요거트/커피: 여름(6-9) 약간 상향, 겨울 약간 하향
    "우유류": {"up":[6,7,8,9], "down":[12,1,2]},
    # 조미소스: 명절(1~2, 9~10) 상향
    "조미소스": {"up":[1,2,9,10], "down":[]},
    # 참치/축산캔: 겨울 저장식·편의식 수요(12~2) 약간 상향
    "참치": {"up":[12,1,2], "down":[6,7,8]},
    "축산캔": {"up":[12,1,2], "down":[6,7,8]},
}

# ============== 공통 유틸 ==============
def month_cols_from_submission(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("months_since_launch_")]

def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = (x - x.max()) / max(temp, 1e-9)
    e = np.exp(z)
    return e / (e.sum() + 1e-9)

def clamp01(x, default=0.5):
    try:
        return float(np.clip(float(x), 0.0, 1.0))
    except Exception:
        return default

# ============== 데이터 로드 ==============
def load_personas(path: str) -> List[Dict[str,Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_products(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["product_name","product_feature","category_level_1","category_level_2","category_level_3"]
    miss = [c for c in expected if c not in df.columns]
    if miss:
        raise ValueError(f"product_info.csv 에 다음 컬럼이 필요합니다: {miss}")
    return df

# ============== LLM 준비 ==============
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    USE_LLM = False   # 키 없으면 자동 비활성화

if USE_LLM:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

def ask_llm_single_turn(prompt: str, temperature=0.5, max_tokens=3000) -> str:
    if not USE_LLM:
        raise RuntimeError("USE_LLM=False (API 키 미설정)")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# ============== 텍스트 → 구조화 (제품 속성/광고월) ==============

# 휴리스틱 키워드 사전(LLM 미사용 시)
KEYWORDS = {
    "protein": ["고단백","단백질","protein"],
    "lactose_free": ["락토프리","유당zero","유당 ZERO","lactose","유당없"],
    "low_sodium": ["저나트륨","나트륨 낮"],
    "spicy": ["매콤","매운"],
    "sesame_oil": ["참기름"],
    "premium": ["프리미엄","고급","명가"],
    "ads": ["광고","TV","YouTube","SNS","엘리베이터"],  # 단어가 있으면 광고 존재 신호
}

def parse_months_from_korean(text: str) -> List[int]:
    """
    '6-8월', '2~6월', '7월' 같은 표현에서 월 리스트를 추출 (1~12)
    - 대회 기간: 2024.07 ~ 2025.06 (12개월)
    - 텍스트의 달은 '달력상 1~12'로 해석한 뒤, 12개월 윈도우에 매핑
    """
    t = text.replace(" ", "")
    months = set()

    # 범위형: 6-8월, 2~6월
    for m1, m2 in re.findall(r"(\d{1,2})\s*[-~]\s*(\d{1,2})\s*월", t):
        a, b = int(m1), int(m2)
        if 1 <= a <= 12 and 1 <= b <= 12:
            if a <= b:
                for k in range(a, b+1): months.add(k)
            else:
                # 11-2월 같은 wrap-around 케이스
                for k in list(range(a,13)) + list(range(1,b+1)): months.add(k)

    # 단일형: 7월, 12월 등
    for m in re.findall(r"(\d{1,2})월", t):
        k = int(m)
        if 1 <= k <= 12: months.add(k)

    return sorted(list(months))

def rule_extract_product(product_row: pd.Series) -> Dict[str,Any]:
    """
    LLM 미사용 시: product_feature에서 간단 키워드 태깅 + 광고월 파싱
    반환:
      {
        "product_name": ...,
        "flags": {"protein":1/0, "lactose_free":..., "low_sodium":..., "spicy":..., "sesame_oil":..., "premium":...},
        "ad_months": [list of 1..12],
        "category_l1": ...,
      }
    """
    name = str(product_row["product_name"])
    feat = str(product_row["product_feature"])
    cat1 = str(product_row["category_level_1"])

    flags = {k: 0.0 for k in ["protein","lactose_free","low_sodium","spicy","sesame_oil","premium"]}
    for key, words in KEYWORDS.items():
        for w in words:
            if w.lower() in feat.lower():
                if key == "ads":
                    # 광고 키워드만 있으면 ad_months가 없을 때 약한 보너스 (월 파싱은 따로)
                    pass
                else:
                    flags[key] = 1.0
                break

    ad_months = parse_months_from_korean(feat)
    return {"product_name": name, "flags": flags, "ad_months": ad_months, "category_l1": cat1}

def llm_extract_all(df: pd.DataFrame) -> List[Dict[str,Any]]:
    """
    LLM 싱글턴 1회로 15개 제품을 구조화(JSON 배열)로 추출.
    각 항목:
      - product_name
      - flags: {protein, lactose_free, low_sodium, spicy, sesame_oil, premium} (0~1)
      - ad_months: [1..12]  (광고/프로모가 집중된 달)
      - category_l1
    """
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "product_name": str(r["product_name"]),
            "product_feature": str(r["product_feature"]),
            "category_level_1": str(r["category_level_1"]),
            "category_level_2": str(r["category_level_2"]),
            "category_level_3": str(r["category_level_3"]),
        })
    table_json = json.dumps(rows, ensure_ascii=False)

    prompt = f"""
아래는 제품 설명 테이블입니다(JSON). 각 제품에 대해 다음 정보를 JSON 배열로만 출력하세요.

필드:
- product_name: 문자열
- flags: 객체 (다음 키를 0~1로 추정) {{
    "protein","lactose_free","low_sodium","spicy","sesame_oil","premium"
  }}
- ad_months: [정수] (1~12 중, 광고/프로모 집중 월; 예: "6-8월" → [6,7,8])
- category_l1: 문자열 (category_level_1과 동일)

입력(JSON):
{table_json}

오직 JSON만 출력하세요 (코드블록/설명 금지).
"""
    txt = ask_llm_single_turn(prompt, temperature=0.2, max_tokens=4000)
    m = re.search(r"\[.*\]", txt, re.S)
    raw = m.group(0) if m else txt
    data = json.loads(raw)
    # 간단 검증/보정
    for d in data:
        d.setdefault("flags", {})
        for k in ["protein","lactose_free","low_sodium","spicy","sesame_oil","premium"]:
            d["flags"][k] = clamp01(d["flags"].get(k, 0.0))
        d.setdefault("ad_months", [])
        d["ad_months"] = [int(x) for x in d["ad_months"] if 1 <= int(x) <= 12]
        d["category_l1"] = d.get("category_l1") or ""
    return data

def build_product_struct(df: pd.DataFrame) -> Dict[str,Dict[str,Any]]:
    """
    df(product_info) → name→구조화 딕셔너리
    USE_LLM=True 이면 llm_extract_all, 아니면 rule_extract_product
    """
    if USE_LLM:
        try:
            ex = llm_extract_all(df)
        except Exception as e:
            print("[WARN] LLM 구조화 실패 → 휴리스틱으로 대체:", e)
            ex = [rule_extract_product(r) for _,r in df.iterrows()]
    else:
        ex = [rule_extract_product(r) for _,r in df.iterrows()]

    # name → object
    out = {d["product_name"]: d for d in ex}
    return out

# ============== 페르소나 × 제품 효용 ==============
def persona_product_utility(persona: Dict[str,Any],
                            prod: Dict[str,Any],
                            cat1: str,
                            month_idx: int) -> float:
    """
    단일 페르소나/제품/월에 대한 효용.
    - flags 가중치: health_consciousness(단백질/저나트륨/락토프리), taste_preference(spicy), brand_loyalty(premium)
    - promotion_sensitivity × (해당 월이 ad_months에 포함되면 보너스)
    - category_season_rules 로 L1 카테고리 시즌 보정
    - price는 없으므로 가격 항은 0
    """
    f = prod["flags"]
    # (1) 기능/건강/취향
    health = float(persona.get("health_consciousness", 0.5))
    brand = float(persona.get("brand_loyalty", 0.5))
    promo_sens = float(persona.get("promotion_sensitivity", 0.5))
    risk = float(persona.get("risk_aversion", 0.5))

    u = 0.0
    # 건강 관련
    u += 0.25 * health * (0.6*f["protein"] + 0.4*f["low_sodium"])
    u += 0.15 * health * f["lactose_free"]
    # 취향: spicy 키가 있으면 taste_preference가 "spicy"일 때 보너스 (없으면 중립)
    taste = str(persona.get("taste_preference","")).lower()
    if "spicy" in taste or "매운" in taste:
        u += 0.10 * f["spicy"]
    else:
        # 매운맛 비선호 가정: 위험회피 높은 경우 약한 패널티
        u -= 0.05 * f["spicy"] * risk

    # 참기름/풍미(고소): brand_loyalty · sustainability_preference 약간 반영(기호적)
    sust = float(persona.get("sustainability_preference", 0.5))
    u += 0.06 * f["sesame_oil"] * (0.5*brand + 0.5*sust)

    # 프리미엄: 브랜드 충성 + 낮은 가격민감
    price_sens = float(persona.get("price_sensitivity", 0.5))
    u += 0.12 * f["premium"] * (0.6*brand + 0.4*(1-price_sens))

    # (2) 광고/프로모 월 보너스
    month_G = month_idx_to_calendar(month_idx)  # 1~12
    if month_G in prod["ad_months"]:
        u += 0.10 * promo_sens

    # (3) 카테고리 시즌 보정
    u += 0.06 * category_season_bump(cat1, month_G)

    return float(u)

def category_season_bump(cat1: str, m: int) -> float:
    """
    L1 카테고리 시즌 보정 (간단한 규칙):
      up 월이면 +1, down 월이면 -1 → 0.5 스케일
    """
    rule = CATEGORY_SEASON_RULES.get(cat1, {})
    score = 0.0
    if m in rule.get("up", []): score += 1.0
    if m in rule.get("down", []): score -= 1.0
    return 0.5 * score

def month_idx_to_calendar(idx: int) -> int:
    """
    대회 윈도우: 2024.07(=idx0) ~ 2025.06(=idx11)
    calendar month 반환(1~12)
    """
    # idx: 0..11 → calendar month: 7..12,1..6
    m = ((7 + idx - 1) % 12) + 1
    return m

# ============== 혼합 확률 및 월별 시뮬 ==============
def mix_probabilities(df: pd.DataFrame,
                      prod_struct: Dict[str,Dict[str,Any]],
                      personas: List[Dict[str,Any]],
                      month_idx: int) -> np.ndarray:
    """
    해당 월에 대해 페르소나 혼합 확률 벡터(제품별) 생성
    """
    names = df["product_name"].tolist()
    cat1s = df["category_level_1"].tolist()

    # 제품별 효용(페르소나 평균)
    U = np.zeros(len(names))
    for j, (nm, c1) in enumerate(zip(names, cat1s)):
        pu = 0.0
        prod = prod_struct[nm]
        for p in personas:
            pu += persona_product_utility(p, prod, c1, month_idx)
        U[j] = pu / max(1, len(personas))

    p = softmax(U, temp=1.0)
    return p  # shape (n_products,)

def month_factors(n_months: int) -> np.ndarray:
    """
    전체 시장(TAM) 계절 요인 — 가벼운 사인 파형
    """
    t = np.arange(n_months)
    season = 1.0 + 0.08*np.sin(2*math.pi*t/12 - math.pi/6)
    return season

def simulate_year(df: pd.DataFrame,
                  prod_struct: Dict[str,Dict[str,Any]],
                  personas: List[Dict[str,Any]],
                  n_months: int = 12,
                  base_tam: float = BASE_TAM,
                  base_out: float = BASE_OUTSIDE) -> Dict[str, np.ndarray]:
    """
    12개월 예측: 제품별 월 수량 벡터 반환
    """
    names = df["product_name"].tolist()
    pred = {nm: np.zeros(n_months) for nm in names}
    factors = month_factors(n_months)

    for m in range(n_months):
        tam = base_tam * factors[m]
        outside = np.clip(base_out + 0.03*np.sin(2*math.pi*m/12), 0.2, 0.7)
        p = mix_probabilities(df, prod_struct, personas, month_idx=m)
        units = tam * (1 - outside) * p
        for j, nm in enumerate(names):
            pred[nm][m] = units[j]
    return pred

# ============== 계층형 베이지안 보정 (카테고리→제품) ==============
def gamma_scale(y_sum: float, yh_sum: float, A0: float, B0: float) -> float:
    """
    m ~ Gamma(A0,B0),  y ~ Poisson(m * y_hat)
    E[m | data] = (A0 + y_sum) / (B0 + yh_sum)
    """
    return (A0 + y_sum) / (B0 + yh_sum + 1e-9)

def hierarchical_calibration(pred: Dict[str,np.ndarray],
                             df: pd.DataFrame,
                             early_path: str = EARLY_ACTUALS_CSV,
                             A0_cat: float = A0_CAT, B0_cat: float = B0_CAT,
                             A0_prd: float = A0_PRD, B0_prd: float = B0_PRD) -> Dict[str,np.ndarray]:
    """
    1차: 카테고리(L1)별로 총합 보정
    2차: 제품별 보정
    - early_actuals.csv: product_name + months_since_launch_1, months_since_launch_2
    """
    if not os.path.exists(early_path):
        return pred

    act = pd.read_csv(early_path)
    # 사용할 월 수(K=1 또는 2)
    K = sum([c in act.columns for c in ["months_since_launch_1","months_since_launch_2"]])
    K = max(1, min(K, 2))
    cols = [f"months_since_launch_{i}" for i in range(1, K+1)]

    # ── 1) 카테고리 단 보정
    pred_scaled = {k: v.copy() for k,v in pred.items()}
    for cat in df["category_level_1"].unique():
        names = df.loc[df["category_level_1"]==cat, "product_name"].tolist()
        # 실제/예측 합계
        y_sum = 0.0; yh_sum = 0.0
        for nm in names:
            if nm in act["product_name"].astype(str).values:
                y = act.loc[act["product_name"].astype(str)==nm, cols].values.astype(float).ravel().sum()
            else:
                y = 0.0
            y_sum  += y
            yh_sum += pred[nm][:K].sum()
        m_cat = gamma_scale(y_sum, yh_sum, A0_cat, B0_cat)
        for nm in names:
            pred_scaled[nm] = pred_scaled[nm] * m_cat

    # ── 2) 제품 단 보정
    for nm in pred_scaled.keys():
        if nm in act["product_name"].astype(str).values:
            y = act.loc[act["product_name"].astype(str)==nm, cols].values.astype(float).ravel().sum()
            yh = pred_scaled[nm][:K].sum()
            m_prd = gamma_scale(y, yh, A0_prd, B0_prd)
            pred_scaled[nm] = pred_scaled[nm] * m_prd

    return pred_scaled

# ============== 메인 ==============
def main():
    # (0) 입력 로드
    personas = load_personas(PERSONA_JSON)
    df = load_products(PRODUCT_INFO_CSV)
    sub = pd.read_csv(SUBMISSION_TEMPLATE)
    month_cols = month_cols_from_submission(sub)
    n_months = len(month_cols)

    print(f"[INFO] products={len(df)}, months={n_months}, personas={len(personas)}, USE_LLM={USE_LLM}")

    # (1) 제품 텍스트/카테고리 → 구조화
    prod_struct = build_product_struct(df)

    # (2) 12개월 예측(텍스트 기반 효용/혼합)
    pred = simulate_year(df, prod_struct, personas, n_months, BASE_TAM, BASE_OUTSIDE)

    # (3) (선택) 계층형 베이지안 보정 (카테고리→제품)
    pred2 = hierarchical_calibration(pred, df, EARLY_ACTUALS_CSV, A0_CAT, B0_CAT, A0_PRD, B0_PRD)

    # (4) 제출 파일 채우기
    out = sub.copy()
    for nm in df["product_name"].astype(str).tolist():
        if nm not in out["product_name"].astype(str).values:
            print(f"[WARN] 템플릿에 없는 제품: {nm} (스킵)")
            continue
        out.loc[out["product_name"].astype(str)==nm, month_cols] = np.round(pred2[nm]).astype(int)

    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] saved {OUTPUT_CSV} shape={out.shape}")

if __name__ == "__main__":
    main()
