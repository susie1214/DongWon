"""
predict_pipeline.py
──────────────────────────────────────────────────────────────────────────────
Dongwon 신제품 예측 — (페르소나 + LLM/RAG WTP + 로짓 + 베이지안 보정) → 제출파일 생성

입력 파일
- personas.json            : app_persona_kr.py 로 생성한 페르소나(싱글턴, 10개+ 속성)
- product_info.csv         : 제품 속성(가격/채널/프로모/규격/브랜드티어/카테고리 등)
- sample_submission.csv    : 대회 제출 템플릿 (product_name + months_since_launch_1..12)
- (선택) early_actuals.csv : 출시 후 1~2개월 실판매 (베이지안 보정용)

출력 파일
- submission.csv           : 대회 제출 형식으로 저장

환경
- .env : OPENAI_API_KEY=sk-...  (있으면 LLM 사용, 없으면 MOCK 휴리스틱 사용)

개요
1) 페르소나 로드 → 특성(가격/프로모/채널/브랜드/지속가능성/리뷰/혁신 등 민감도) 확인
2) RAG(제품요약) + (선택) LLM 1회 호출로 속성 WTP/가중치(JSON) 추정
   - weights:   범주형 속성(level별 0~1 점수)
   - weights_numeric: 수치형 속성(size 등)의 min/max/direction/weight
   - price_slope: 가격 기울기(음수)
3) 각 페르소나별 효용(Utility) 계산 → softmax로 선택확률 → 페르소나 혼합
   - 페르소나 특성으로 개별 조정 (가격민감도·프로모민감도·채널 코드·브랜드충성 등)
4) 월별 시즌/프로모 요인 + TAM × (1 - Outside)로 월별 수량 환산
5) (선택) early_actuals.csv가 있으면 1~2개월로 베이지안 스케일 보정 (Gamma-Poisson)
6) sample_submission.csv 포맷에 맞춰 submission.csv 저장

주의
- LLM 사용 비용/토큰을 줄이기 위해 속성 요약만 전달하는 싱글턴 1회 호출 구조
- LLM 미사용 시에도 휴리스틱으로 동작하여 로컬 테스트 가능
──────────────────────────────────────────────────────────────────────────────
"""

import os, re, json, math, random
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ============================== 설정 =============================== #

# (필수) 파일 경로
PERSONA_JSON         = "personas.json"
PRODUCT_INFO_CSV     = "product_info.csv"       # 최신 파일명
SUBMISSION_TEMPLATE  = "sample_submission.csv"
EARLY_ACTUALS_CSV    = "early_actuals.csv"      # 선택(없어도 됨)
OUTPUT_CSV           = "submission.csv"

# (옵션) 실행 파라미터
USE_LLM           = True     # .env에 키가 없으면 자동으로 False로 동작
MODEL              = os.getenv("FT_MODEL_ID") or "gpt-4o-mini"
N_SIMS             = 60      # 월별 몬테카를로 반복(확률 안정화)
SEED               = 42
random.seed(SEED); np.random.seed(SEED)

# 시장/보정 파라미터
BASE_TAM           = 100_000    # 월간 잠재 수요(초깃값) — 필요 시 조정/캘리브레이션
BASE_OUTSIDE       = 0.45       # 비구매 비율 평균 (프롬프트 문구/프로모에 민감)
A0, B0             = 1.0, 1.0   # 베이지안 사전(감마) — 보수/공격 튜닝 (A0,B0↑→보수적)

# 제품 속성 매핑(데이터 컬럼명에 맞게 필요 시 수정)
ATTR_MAP = {
    "product_name": "product_name",
    "price":        "price",
    "promo":        "promo_flag",
    "channel":      "channel",      # product_info.csv의 채널 텍스트 (예: online/offline/omni)
    "size":         "size_g",
    "brand_tier":   "brand_tier",   # premium/mainstream 등
    "category":     "category",
}

# 내부 채널 코드(페르소나에는 ON/OF/OM) ↔ 제품 채널 텍스트 매핑
CHANNEL_TEXT_TO_CODE = {
    "online": "ON", "온라인": "ON",
    "offline":"OF", "오프라인":"OF", "마트":"OF", "편의점":"OF",
    "omni":  "OM", "옴니": "OM", "mixed":"OM"
}

# ========================= 공통 유틸/LLM ========================== #

def month_cols_from_submission(df: pd.DataFrame) -> List[str]:
    """제출 템플릿에서 'months_since_launch_*' 컬럼을 자동 탐지"""
    return [c for c in df.columns if c.startswith("months_since_launch_")]

def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = (x - x.max()) / max(temp, 1e-9)
    e = np.exp(z)
    return e / (e.sum() + 1e-9)

# LLM 준비
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    USE_LLM = False  # 키 없으면 자동으로 LLM 비활성화

if USE_LLM:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

def ask_llm_single_turn(prompt: str, temperature: float = 0.7, max_tokens: int = 1800) -> str:
    """싱글턴 LLM 호출 (키가 없으면 예외 대신 휴리스틱으로 대체)"""
    if not USE_LLM:
        raise RuntimeError("USE_LLM=False 상태입니다(.env 키가 없거나 비활성화).")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# ========================= 1) 데이터 로드 ========================= #

def load_personas(path: str) -> List[Dict[str,Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 를 찾을 수 없습니다. (먼저 app_persona_kr.py 실행)")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("personas.json 형식이 올바르지 않습니다 (JSON 배열 필요)")
    return data

def load_products(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 를 찾을 수 없습니다.")
    df = pd.read_csv(path)
    # 필수 컬럼 체크
    for key, col in ATTR_MAP.items():
        if col not in df.columns:
            print(f"[WARN] product_info.csv에 '{col}' 컬럼이 없습니다. ({key})")
    return df

def summarize_products_for_prompt(df: pd.DataFrame) -> str:
    """
    RAG 컨텍스트로 넣을 '속성 요약' 생성.
    - 범주형: 상위 수준 빈도
    - 수치형: 요약 통계
    """
    lines = []
    for key, col in ATTR_MAP.items():
        if key == "product_name" or col not in df.columns: 
            continue
        if df[col].dtype == "O":
            vc = df[col].astype(str).value_counts().head(8)
            lines.append(f"{col}: " + ", ".join([f"{k}({v})" for k,v in vc.items()]))
        else:
            desc = df[col].describe().to_dict()
            lines.append(
                f"{col}: min={desc.get('min')}, q25={desc.get('25%')}, "
                f"median={desc.get('50%')}, q75={desc.get('75%')}, max={desc.get('max')}"
            )
    return "\n".join(lines)

def summarize_personas_for_prompt(personas: List[Dict[str,Any]]) -> str:
    """
    페르소나들의 핵심 분포를 간단 요약(토큰 절약).
    인구통계/민감도 평균치를 LLM에 힌트로 제공.
    """
    def mean_of(key, default=0.5):
        vals = [p.get(key, default) for p in personas if isinstance(p.get(key, default), (int,float))]
        return round(float(np.mean(vals)) if vals else default, 3)

    # 텍스트 분포는 상위 항목만
    def topk_of(key, k=5):
        vc = pd.Series([str(p.get(key,"")) for p in personas]).value_counts().head(k)
        return ", ".join([f"{a}({b})" for a,b in vc.items()])

    lines = []
    lines.append(f"gender: {topk_of('gender')}")
    lines.append(f"age: {topk_of('age')}")
    lines.append(f"region_kr: {topk_of('region_kr')}")
    lines.append(f"channel_code: {topk_of('channel_code')}")
    for key in ["price_sensitivity","promotion_sensitivity","brand_loyalty",
                "innovation_seeking","review_dependence","sustainability_preference",
                "risk_aversion","baseline_purchase_frequency"]:
        lines.append(f"{key}_mean: {mean_of(key)}")
    return "\n".join(lines)

# ================= 2) LLM/RAG: WTP/속성 가중치 추정 ================= #

def estimate_attribute_weights(df_prod: pd.DataFrame, personas: List[Dict[str,Any]]) -> Dict[str,Any]:
    """
    싱글턴 1회로 속성 가중치 추정(JSON):
    {
      "weights": { "channel": {"online":0.55, "offline":0.45}, "brand_tier":{"premium":0.6,...}, ... },
      "weights_numeric": { "size_g": {"min":80,"max":180,"direction":"+","weight":0.2} },
      "price_slope": -0.0003
    }
    - USE_LLM=True이면 LLM에 RAG 요약 + 페르소나 분포 요약을 주고 받음
    - USE_LLM=False이면 간단 MOCK(균등+노이즈) 가중치 사용
    """
    # (1) 속성/페르소나 요약
    prod_summary = summarize_products_for_prompt(df_prod)
    persona_summary = summarize_personas_for_prompt(personas)

    if USE_LLM:
        prompt = f"""
동원 신제품의 속성별 선호 가중치를 JSON으로만 생성하세요.
- "weights": 범주형 속성(level별 0~1 점수)
- "weights_numeric": 수치형 속성(min/max/direction/weight)
- "price_slope": 가격 기울기(음수; 절대값↑ → 가격 민감↑)
참고: 아래는 제품 속성 요약(RAG)과 페르소나 집합의 분포 요약입니다.

[제품 요약]
{prod_summary}

[페르소나 요약(평균/상위)]
{persona_summary}

오직 JSON만 출력하세요. 코드블록/설명 금지.
예시 스키마:
{{ "weights": {{ "channel": {{"online":0.55,"offline":0.45}} }},
   "weights_numeric": {{ "size_g": {{"min":80,"max":200,"direction":"+","weight":0.2}} }},
   "price_slope": -0.0003 }}
"""
        txt = ask_llm_single_turn(prompt, temperature=0.6, max_tokens=1600)
        m = re.search(r"\{.*\}", txt, re.S)
        raw = m.group(0) if m else txt
        try:
            parsed = json.loads(raw)
        except Exception as e:
            print("[WARN] LLM JSON 파싱 실패 → 휴리스틱 대체:", e)
            USE_LLM_LOCAL = False
            parsed = None
    else:
        parsed = None

    # (2) LLM 실패 또는 미사용 → 휴리스틱 가중치
    if not parsed:
        parsed = {"weights":{}, "weights_numeric":{}, "price_slope": -0.0003}
        # 범주형: 값 상위빈도로 level 추출 → 균등+노이즈
        for key, col in ATTR_MAP.items():
            if key == "product_name" or col not in df_prod.columns:
                continue
            if df_prod[col].dtype == "O":
                levels = df_prod[col].astype(str).value_counts().head(6).index.tolist()
                w = np.clip(np.random.normal(1.0, 0.15, len(levels)), 0.3, 1.7)
                w = (w / w.sum()).tolist()
                parsed["weights"][col] = {lv: float(wi) for lv, wi in zip(levels, w)}
        # 수치형: size_g만 예시로 선호 증가(+)
        if ATTR_MAP["size"] in df_prod.columns:
            s = df_prod[ATTR_MAP["size"]].dropna().astype(float)
            if len(s) > 0:
                parsed["weights_numeric"][ATTR_MAP["size"]] = {
                    "min": float(s.min()), "max": float(s.max()),
                    "direction": "+", "weight": 0.2
                }
        parsed.setdefault("price_slope", -0.0003)

    # 필수 필드 채우기
    parsed.setdefault("weights", {})
    parsed.setdefault("weights_numeric", {})
    parsed.setdefault("price_slope", -0.0003)
    return parsed

# ============== 3) 페르소나별 효용/확률 계산 ============== #

def channel_text_to_code(txt: str) -> str:
    t = str(txt).strip().lower()
    for k, v in CHANNEL_TEXT_TO_CODE.items():
        if k in t: return v
    return "OM"

def numeric_effect(x: float, spec: Dict[str,Any]) -> float:
    """weights_numeric 스펙(min/max/direction/weight)을 0~1 효과값으로 변환"""
    try:
        vmin = float(spec.get("min", 0.0)); vmax = float(spec.get("max", 1.0))
        w = float(spec.get("weight", 0.2))
        direction = spec.get("direction", "+")
        if vmax <= vmin: return 0.0
        norm = np.clip((float(x) - vmin) / (vmax - vmin), 0.0, 1.0)
        return (norm if direction == "+" else 1.0 - norm) * w
    except Exception:
        return 0.0

def utility_for_persona(row: pd.Series,
                        W: Dict[str,Any],
                        persona: Dict[str,Any]) -> float:
    """
    단일 제품(row)에 대해 특정 페르소나(persona)의 효용 점수 U를 계산.
    구성:
      U_base     : LLM/RAG 기반 속성 가중치(범주형 level + 수치형)
      U_channel  : 페르소나 채널코드 선호와 제품 채널 매칭 보너스
      U_brand    : brand_tier와 price/brand 민감도 상호작용
      U_promo    : promo_flag × promotion_sensitivity
      U_price    : price_slope × price × (1 + price_sensitivity)  (민감할수록 불이익↑)
    """
    u = 0.0
    # (1) 범주형 level 가중치
    for key, col in ATTR_MAP.items():
        if key in ["product_name", "price"] or col not in row.index: 
            continue
        if row.index.dtype == "O":
            pass
        if col in W["weights"]:
            lv = str(row[col])
            if lv in W["weights"][col]:
                u += float(W["weights"][col][lv])

    # (2) 수치형 가중치 (예: size_g)
    for col, spec in (W.get("weights_numeric") or {}).items():
        if col in row.index:
            try:
                u += numeric_effect(float(row[col]), spec)
            except Exception:
                pass

    # (3) 채널 매칭 보너스
    p_code = str(persona.get("channel_code","OM")).upper()
    prod_code = channel_text_to_code(row.get(ATTR_MAP["channel"], "OM"))
    if p_code == prod_code:
        u += 0.10  # 매칭 보너스 (필요 시 조정)

    # (4) 브랜드/가격 상호작용
    brand_tier = str(row.get(ATTR_MAP["brand_tier"], "")).lower()
    price_sens = float(persona.get("price_sensitivity", 0.5))
    brand_loyal = float(persona.get("brand_loyalty", 0.5))
    if "premium" in brand_tier:
        u += 0.05 * (1 - price_sens) + 0.05 * brand_loyal
    else:
        u += 0.03 * (price_sens)  # 메인스트림일수록 가성비 지향 보너스

    # (5) 프로모 상호작용
    promo_flag = float(row.get(ATTR_MAP["promo"], 0))
    promo_sens = float(persona.get("promotion_sensitivity", 0.5))
    u += 0.10 * promo_flag * promo_sens

    # (6) 가격 항 (음수)
    base_price_slope = float(W.get("price_slope", -0.0003))
    # 가격 민감도가 클수록(→1) 더 가파르게 벌점
    slope = base_price_slope * (0.5 + price_sens)
    price = float(row.get(ATTR_MAP["price"], 0.0))
    u += slope * price

    return float(u)

def mix_probabilities_over_personas(df_prod: pd.DataFrame,
                                    W: Dict[str,Any],
                                    personas: List[Dict[str,Any]],
                                    temp: float = 1.0) -> np.ndarray:
    """
    제품 × 페르소나 효용 → 소프트맥스로 확률화 → 페르소나 혼합
    - 페르소나 가중치는 baseline_purchase_frequency 평균으로 가중(없으면 균등)
    반환: 제품별 선택확률 벡터(shape = [n_products])
    """
    n_prod = len(df_prod)
    n_pers = len(personas)
    utils = np.zeros((n_pers, n_prod))

    for i, p in enumerate(personas):
        row_utils = []
        for _, row in df_prod.iterrows():
            row_utils.append(utility_for_persona(row, W, p))
        row_utils = np.array(row_utils)
        # 개별 페르소나 softmax
        prob_i = softmax(row_utils, temp=temp)
        utils[i, :] = prob_i

    # 페르소나 가중치 (기본: baseline_purchase_frequency 평균)
    weights = np.array([float(p.get("baseline_purchase_frequency", 0.2)) for p in personas])
    if weights.sum() <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()

    # 혼합
    mix = weights @ utils
    mix = mix / (mix.sum() + 1e-9)
    return mix  # shape: (n_prod,)

# ============== 4) 월별 요인, TAM/Outside, 시뮬레이션 ============== #

def month_factors(n_months: int) -> np.ndarray:
    """시즌성(사인) × 프로모 스파이크(임의 3개월)를 조합"""
    t = np.arange(n_months)
    season = 1.0 + 0.08*np.sin(2*math.pi*t/12 - math.pi/6)
    promo = np.ones(n_months); promo[[2,5,10]] = 1.12
    return season * promo

def simulate_month(df_prod: pd.DataFrame,
                   personas: List[Dict[str,Any]],
                   W: Dict[str,Any],
                   tam_units: float,
                   outside_share: float,
                   sims: int = N_SIMS) -> np.ndarray:
    """
    월별 시뮬레이션:
      1) 페르소나 혼합 확률 벡터 p (제품별)
      2) 수량 = TAM × (1 - Outside) × p
    반복(sims) 평균은 모델 내 랜덤요소가 있을 때 안정화 용도(여기선 1회와 유사)
    """
    agg = np.zeros(len(df_prod))
    for _ in range(max(1, sims)):
        p = mix_probabilities_over_personas(df_prod, W, personas, temp=1.0)
        units = tam_units * (1 - outside_share) * p
        agg += units
    agg /= max(1, sims)
    return agg  # 제품별 월 수량

# ============== 5) 베이지안 스케일 보정 (1~2개월) ============== #

def gamma_scale(y_true: np.ndarray, y_hat: np.ndarray,
                A0: float = A0, B0: float = B0) -> float:
    """
    y ~ Poisson(m * y_hat),  m ~ Gamma(A0, B0)   (rate-parameterization)
    posterior mean: E[m | data] = (A0 + sum y) / (B0 + sum y_hat)
    """
    return (A0 + float(np.sum(y_true))) / (B0 + float(np.sum(y_hat)) + 1e-9)

# ============================= 메인 ============================= #

def main():
    # 0) 입력 로드
    personas = load_personas(PERSONA_JSON)
    df_prod = load_products(PRODUCT_INFO_CSV)
    sub = pd.read_csv(SUBMISSION_TEMPLATE)
    month_cols = month_cols_from_submission(sub)
    n_months = len(month_cols)
    if n_months == 0:
        raise ValueError("sample_submission.csv 에 months_since_launch_* 컬럼이 없습니다.")

    prod_names = df_prod[ATTR_MAP["product_name"]].astype(str).tolist()

    print(f"[INFO] products={len(prod_names)}, months={n_months}, personas={len(personas)}, USE_LLM={USE_LLM}")

    # 1) 속성 WTP/가중치 추정 (LLM 싱글턴 or 휴리스틱)
    W = estimate_attribute_weights(df_prod, personas)
    print("[INFO] price_slope=", W.get("price_slope"))

    # 2) 월별 예측
    factors = month_factors(n_months)
    pred = {name: np.zeros(n_months) for name in prod_names}

    for m in range(n_months):
        tam = BASE_TAM * factors[m]
        # Outside는 계절로 약간 변동 (예시)
        outside = np.clip(BASE_OUTSIDE + 0.03*np.sin(2*math.pi*m/12), 0.2, 0.7)
        units = simulate_month(df_prod, personas, W, tam, outside, sims=N_SIMS)
        for j, nm in enumerate(prod_names):
            pred[nm][m] = units[j]

    # 3) (선택) 1~2개월 베이지안 보정
    if os.path.exists(EARLY_ACTUALS_CSV):
        act = pd.read_csv(EARLY_ACTUALS_CSV)
        # 첫 1~2개월만 자동 탐지
        K = sum([c in act.columns for c in ["months_since_launch_1","months_since_launch_2"]])
        K = max(1, min(K, 2))
        use_cols = [f"months_since_launch_{i}" for i in range(1, K+1)]
        for nm in prod_names:
            if nm in act["product_name"].astype(str).values:
                y_true = act.loc[act["product_name"].astype(str)==nm, use_cols].values.astype(float).ravel()
                y_hat  = pred[nm][:K]
                m_hat  = gamma_scale(y_true, y_hat, A0=A0, B0=B0)
                pred[nm] = pred[nm] * m_hat
                print(f"[CAL] {nm}: m_hat={m_hat:.3f}")

    # 4) 제출 파일 채우기 (정수 반올림)
    out = sub.copy()
    for nm in prod_names:
        if nm not in out["product_name"].astype(str).values:
            # 템플릿에 없으면 스킵(또는 추가 로직으로 append 가능)
            print(f"[WARN] submission 템플릿에 없는 제품: {nm} (스킵)")
            continue
        out.loc[out["product_name"].astype(str)==nm, month_cols] = np.round(pred[nm]).astype(int)

    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] saved {OUTPUT_CSV} shape={out.shape}")

    # 5) (선택) SMAPE 계산 훅 — 정답이 있을 경우만 사용
    # def smape(y_true, y_pred):
    #     mask = y_true > 0
    #     yt = y_true[mask]; yp = y_pred[mask]
    #     return float(np.mean(np.abs(yt-yp) / ((np.abs(yt)+np.abs(yp))/2 + 1e-9)))
    # if os.path.exists("ground_truth.csv"):
    #     gt = pd.read_csv("ground_truth.csv")
    #     s_public  = smape(gt[month_cols[:6]].values,  out[month_cols[:6]].values)
    #     s_private = smape(gt[month_cols].values,      out[month_cols].values)
    #     print(f"[SMAPE] public={s_public:.4f}, private={s_private:.4f}")

if __name__ == "__main__":
    main()
