"""
predict_pipeline_v3.py  (auto-template)
──────────────────────────────────────────────────────────────────────────────
- v3 파이프라인에 '옵션 파일 템플릿 자동 생성'을 추가
- product_info.csv 로드 시 아래 파일이 없으면 템플릿을 자동 생성:
    • early_actuals_template.csv   # product_name, months_since_launch_1..2
    • tam_overrides_template.csv   # category_level_1, month_1..month_12 (배수)
- 사용자는 템플릿에 값을 채워 'early_actuals.csv' / 'tam_overrides.csv'로 저장하면 됨
"""

import os, re, json, math, random
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ========================== 하이퍼파라미터 ==========================
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# 파일 경로
PERSONA_JSON         = "personas.json"
PRODUCT_INFO_CSV     = "product_info.csv"
SUBMISSION_TEMPLATE  = "sample_submission.csv"
EARLY_ACTUALS_CSV    = "early_actuals.csv"      # 선택(없으면 템플릿만 생성)
TAM_OVERRIDES_CSV    = "tam_overrides.csv"      # 선택(없으면 템플릿만 생성)

OUTPUT_CSV           = "submission.csv"

# LLM
USE_LLM              = True       # .env 키 없으면 자동 False
MODEL                = os.getenv("FT_MODEL_ID") or "gpt-4o-mini"

# 시장/보정/광고 관련
BASE_TAM             = 120_000    # 월간 시장수요 기본치
BASE_OUTSIDE         = 0.45       # 비구매 비율
CAT_TAM_AD_BOOST     = 1.08       # 광고월 카테고리 TAM 배수
AD_BONUS_PER_PERSONA = 0.14       # u += AD_BONUS_PER_PERSONA * promo_sens
EMA_ALPHA            = 0.35       # EMA 스무딩
WINSOR_LOW, WINSOR_HIGH = 0.02, 0.98

# 베이지안(감마-포아송) 계층형 보정 강도 (작을수록 민감)
A0_CAT, B0_CAT       = 0.3, 0.3   # 카테고리
A0_PRD, B0_PRD       = 0.8, 0.8   # 제품

# 하드 오버라이드: 초기 1~K개월 실측으로 덮기 (0/1/2)
HARD_OVERRIDE_K      = 1

# 시즌 규칙 (L1/L2/L3 키 모두 적용 가능; 없으면 무시)
CATEGORY_SEASON_RULES = {
    "우유류": {"up":[6,7,8,9], "down":[12,1,2]},
    "조미소스": {"up":[1,2,9,10], "down":[]},
    "참치": {"up":[12,1,2], "down":[6,7,8]},
    "축산캔": {"up":[12,1,2], "down":[6,7,8]},
    "발효유": {"up":[6,7,8,9], "down":[12,1,2], "bump":0.10},
    "호상-중대용량": {"up":[6,7,8,9], "down":[12,1,2], "bump":0.20},
    "액상조미료": {"up":[1,2,9,10], "down":[], "bump":0.10},
}

# 모델 파워(간단 예시)
MODEL_POWER = {"안유진": 0.25}

# ========================== 공통 유틸 ==========================
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

def month_idx_to_calendar(idx: int) -> int:
    # 대회 윈도우: 2024.07(=idx0) ~ 2025.06(=idx11)
    return ((7 + idx - 1) % 12) + 1  # 1..12

# ========================== 데이터 로드/LLM ==========================
def load_personas(path: str) -> List[Dict[str,Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_products(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ["product_name","product_feature","category_level_1","category_level_2","category_level_3"]
    miss = [c for c in need if c not in df.columns]
    if miss: raise ValueError(f"product_info.csv 컬럼 누락: {miss}")
    return df

# --- ⬇️ 추가: 옵션 템플릿 자동 생성 --------------------------------
def ensure_optional_templates(df_products: pd.DataFrame) -> None:
    """
    폴더에 early_actuals.csv / tam_overrides.csv 가 없을 경우,
    각 템플릿 파일을 자동 생성한다.
    - early_actuals_template.csv : product_name + months_since_launch_1..2 (0으로 채움)
    - tam_overrides_template.csv : category_level_1 + month_1..12 (1.0으로 채움)
    사용자는 값을 채운 후 파일명을 각각 early_actuals.csv / tam_overrides.csv 로 저장하면 됨.
    """
    # early_actuals
    if not os.path.exists(EARLY_ACTUALS_CSV) and not os.path.exists("early_actuals_template.csv"):
        prod_names = df_products["product_name"].astype(str).unique()
        ea = pd.DataFrame({"product_name": prod_names})
        ea["months_since_launch_1"] = 0
        ea["months_since_launch_2"] = 0
        ea.to_csv("early_actuals_template.csv", index=False, encoding="utf-8-sig")
        print("[INFO] created early_actuals_template.csv (초기 1~2개월 실판매 템플릿)")

    # tam_overrides
    if not os.path.exists(TAM_OVERRIDES_CSV) and not os.path.exists("tam_overrides_template.csv"):
        cats = df_products["category_level_1"].astype(str).unique()
        to = pd.DataFrame({"category_level_1": cats})
        for m in range(1, 13):
            to[f"month_{m}"] = 1.0
        to.to_csv("tam_overrides_template.csv", index=False, encoding="utf-8-sig")
        print("[INFO] created tam_overrides_template.csv (카테고리/월별 TAM 배수 템플릿)")
# -------------------------------------------------------------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    USE_LLM = False
if USE_LLM:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

def ask_llm_single_turn(prompt: str, temperature=0.2, max_tokens=4000) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# ========================== 텍스트 → 구조화 ==========================
KEYWORDS = {
    "protein": ["고단백","단백질","protein"],
    "lactose_free": ["락토프리","유당 zero","유당zero","lactose"],
    "low_sodium": ["저나트륨","나트륨 낮"],
    "low_sugar": ["저당","당 줄인","low sugar","무가당"],
    "spicy": ["매콤","매운"],
    "vanilla": ["바닐라"],
    "latte": ["라떼","카페라떼"],
    "sesame_oil": ["참기름"],
    "premium": ["프리미엄","명가","고급"],
    "tv_ad": ["TV","티비","방송광고"],
    "social_ad": ["YouTube","SNS","유튜브","소셜"],
    "elevator_ad": ["엘리베이터"],
    "festival": ["명절","추석","설"],
}

def parse_months_from_text(text: str) -> List[int]:
    t = text.replace(" ", "")
    months = set()
    for m1,m2 in re.findall(r"(\d{1,2})\s*[-~]\s*(\d{1,2})\s*월", t):
        a,b = int(m1), int(m2)
        if 1<=a<=12 and 1<=b<=12:
            months.update(range(a,b+1) if a<=b else list(range(a,13))+list(range(1,b+1)))
    for m in re.findall(r"(\d{1,2})월", t):
        k = int(m); 1<=k<=12 and months.add(k)
    return sorted(months)

def rule_extract_product(r: pd.Series) -> Dict[str,Any]:
    name = str(r["product_name"])
    feat = str(r["product_feature"]).lower()
    cat1,cat2,cat3 = str(r["category_level_1"]), str(r["category_level_2"]), str(r["category_level_3"])
    flags = {k:0.0 for k in ["protein","lactose_free","low_sodium","low_sugar","spicy","vanilla","latte",
                             "sesame_oil","premium","tv_ad","social_ad","elevator_ad","festival","model_star_power"]}
    for k,words in KEYWORDS.items():
        for w in words:
            if w.lower() in feat: flags[k]=1.0; break
    for model_name, power in MODEL_POWER.items():
        if model_name in r["product_feature"]:
            flags["model_star_power"] = power
    ad_months = parse_months_from_text(r["product_feature"])
    return {"product_name":name, "flags":flags, "ad_months":ad_months, "cat1":cat1, "cat2":cat2, "cat3":cat3}

def llm_extract_all(df: pd.DataFrame) -> List[Dict[str,Any]]:
    rows = [{"product_name":x["product_name"], "product_feature":x["product_feature"],
             "category_level_1":x["category_level_1"], "category_level_2":x["category_level_2"],
             "category_level_3":x["category_level_3"]} for _,x in df.iterrows()]
    tab = json.dumps(rows, ensure_ascii=False)
    prompt = f"""
아래 JSON 테이블의 각 제품에 대해 다음 스키마로 JSON 배열만 출력하세요.
- product_name
- flags: {{
  "protein","lactose_free","low_sodium","low_sugar","spicy","vanilla","latte",
  "sesame_oil","premium","tv_ad","social_ad","elevator_ad","festival","model_star_power"
  }}  # 0~1
- ad_months: [정수 1..12]
- cat1, cat2, cat3: category_level_1/2/3 그대로
입력:
{tab}
오직 JSON만 출력(코드블록/설명 금지).
"""
    txt = ask_llm_single_turn(prompt, temperature=0.2, max_tokens=5000)
    m = re.search(r"\[.*\]", txt, re.S)
    raw = m.group(0) if m else txt
    data = json.loads(raw)
    for d in data:
        d.setdefault("flags", {})
        for k in ["protein","lactose_free","low_sodium","low_sugar","spicy","vanilla","latte",
                  "sesame_oil","premium","tv_ad","social_ad","elevator_ad","festival","model_star_power"]:
            d["flags"][k] = clamp01(d["flags"].get(k,0.0))
        d.setdefault("ad_months", [])
        d["ad_months"] = [int(x) for x in d["ad_months"] if 1<=int(x)<=12]
        for k in ["cat1","cat2","cat3"]:
            d[k] = d.get(k,"")
    return data

def build_product_struct(df: pd.DataFrame) -> Dict[str,Dict[str,Any]]:
    if USE_LLM:
        try:
            ex = llm_extract_all(df)
        except Exception as e:
            print("[WARN] LLM 구조화 실패 → 휴리스틱 사용:", e)
            ex = [rule_extract_product(r) for _,r in df.iterrows()]
    else:
        ex = [rule_extract_product(r) for _,r in df.iterrows()]
    return {d["product_name"]: d for d in ex}

# ========================== 시즌/광고/효용 ==========================
def category_season_bump(cat_keys: List[str], cal_month: int) -> float:
    bump = 0.0
    for key in cat_keys:
        rule = CATEGORY_SEASON_RULES.get(key, None)
        if not rule: continue
        base = float(rule.get("bump", 0.06))
        if cal_month in rule.get("up", []):   bump += base
        if cal_month in rule.get("down", []): bump -= base
    return bump

def persona_product_utility(persona: Dict[str,Any], prod: Dict[str,Any], month_idx: int) -> float:
    f = prod["flags"]; cal_m = month_idx_to_calendar(month_idx)
    health = float(persona.get("health_consciousness", 0.5))
    taste  = str(persona.get("taste_preference","")).lower()
    brand  = float(persona.get("brand_loyalty", 0.5))
    promo  = float(persona.get("promotion_sensitivity", 0.5))
    price_sens = float(persona.get("price_sensitivity", 0.5))
    risk   = float(persona.get("risk_aversion", 0.5))

    u = 0.0
    u += 0.20 * health * (0.35*f["protein"] + 0.25*f["low_sodium"] + 0.20*f["lactose_free"] + 0.20*f["low_sugar"])

    spicy_like = ("spicy" in taste) or ("매운" in taste)
    coffee_like = ("latte" in taste) or ("coffee" in taste) or ("커피" in taste)
    vanilla_like = ("vanilla" in taste) or ("바닐라" in taste)
    if spicy_like:   u += 0.10 * f["spicy"]
    else:            u -= 0.05 * f["spicy"] * risk
    if coffee_like:  u += 0.08 * f["latte"]
    if vanilla_like: u += 0.06 * f["vanilla"]

    sust = float(persona.get("sustainability_preference", 0.5))
    u += 0.05 * f["sesame_oil"] * (0.5*brand + 0.5*sust)
    u += 0.13 * f["premium"] * (0.6*brand + 0.4*(1-price_sens))

    if cal_m in prod["ad_months"]:
        ad_weight = 1.0 + 0.15*f["tv_ad"] + 0.12*f["social_ad"] + 0.08*f["elevator_ad"]
        u += AD_BONUS_PER_PERSONA * promo * ad_weight

    u += 0.12 * f["model_star_power"]
    u += category_season_bump([prod["cat1"], prod["cat2"], prod["cat3"]], cal_m)
    return float(u)

def month_factors_base(n_months: int) -> np.ndarray:
    t = np.arange(n_months)
    return 1.0 + 0.08*np.sin(2*math.pi*t/12 - math.pi/6)

def load_tam_overrides(path: str, df: pd.DataFrame) -> Dict[Tuple[str,int], float]:
    if not os.path.exists(path):
        return {}
    src = pd.read_csv(path)
    out = {}
    for _, r in src.iterrows():
        cat1 = str(r["category_level_1"])
        for m in range(1,13):
            col = f"month_{m}"
            if col in src.columns:
                try: out[(cat1, m)] = float(r[col])
                except: pass
    return out

def simulate_year(df: pd.DataFrame, prod_struct: Dict[str,Dict[str,Any]],
                  personas: List[Dict[str,Any]], tam_overrides, n_months=12,
                  base_tam=BASE_TAM, base_out=BASE_OUTSIDE) -> Dict[str,np.ndarray]:
    names = df["product_name"].tolist()
    cat1s = df["category_level_1"].tolist()
    pred = {nm: np.zeros(n_months) for nm in names}
    base = month_factors_base(n_months)

    for m in range(n_months):
        tam = base_tam * base[m]
        cal_m = month_idx_to_calendar(m)

        # 광고월 카테고리 TAM boost(평균 반영)
        cat_boosts = []
        for nm,c1 in zip(names,cat1s):
            if cal_m in prod_struct[nm]["ad_months"]:
                cat_boosts.append(CAT_TAM_AD_BOOST)
        if cat_boosts:
            tam *= float(np.mean(cat_boosts))

        # 수동 오버라이드(있을 때만)
        if tam_overrides:
            factors = [tam_overrides.get((c1, cal_m), 1.0) for c1 in cat1s]
            tam *= float(np.mean(factors))

        outside = np.clip(base_out + 0.03*np.sin(2*math.pi*m/12), 0.18, 0.72)

        U = np.zeros(len(names))
        for j,nm in enumerate(names):
            prod = prod_struct[nm]
            U[j] = np.mean([persona_product_utility(p, prod, m) for p in personas])
        p = softmax(U, temp=1.0)
        units = tam * (1 - outside) * p
        for j,nm in enumerate(names):
            pred[nm][m] = units[j]
    return pred

# ========================== 보정/스무딩 ==========================
def gamma_scale(y_sum: float, yh_sum: float, A0: float, B0: float) -> float:
    return (A0 + y_sum) / (B0 + yh_sum + 1e-9)

def hierarchical_calibration(pred: Dict[str,np.ndarray], df: pd.DataFrame,
                             early_path: str = EARLY_ACTUALS_CSV,
                             A0_cat: float = A0_CAT, B0_cat: float = B0_CAT,
                             A0_prd: float = A0_PRD, B0_prd: float = B0_PRD,
                             hard_override_k: int = HARD_OVERRIDE_K) -> Dict[str,np.ndarray]:
    if not os.path.exists(early_path):
        return pred
    act = pd.read_csv(early_path)
    K = sum([c in act.columns for c in ["months_since_launch_1","months_since_launch_2"]])
    K = max(1, min(K, 2))
    cols = [f"months_since_launch_{i}" for i in range(1, K+1)]
    out = {k: v.copy() for k,v in pred.items()}

    # 카테고리 보정
    for cat in df["category_level_1"].unique():
        names = df.loc[df["category_level_1"]==cat,"product_name"].tolist()
        y_sum=yh_sum=0.0
        for nm in names:
            y = act.loc[act["product_name"].astype(str)==nm, cols].values.astype(float).ravel().sum() if nm in act["product_name"].astype(str).values else 0.0
            y_sum += y; yh_sum += out[nm][:K].sum()
        m_cat = gamma_scale(y_sum, yh_sum, A0_cat, B0_cat)
        for nm in names: out[nm] *= m_cat

    # 제품 보정 + 하드 오버라이드
    for nm in out.keys():
        if nm in act["product_name"].astype(str).values:
            y = act.loc[act["product_name"].astype(str)==nm, cols].values.astype(float).ravel().sum()
            yh = out[nm][:K].sum()
            m_prd = gamma_scale(y, yh, A0_prd, B0_prd)
            out[nm] *= m_prd
            if hard_override_k > 0:
                yrow = act.loc[act["product_name"].astype(str)==nm, cols].values.astype(float).ravel()
                for i in range(min(hard_override_k, K)):
                    out[nm][i] = yrow[i]
    return out

def ema_smooth(x: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    y = np.zeros_like(x, dtype=float); y[0] = x[0]
    for i in range(1, len(x)): y[i] = alpha*x[i] + (1-alpha)*y[i-1]
    return y

def winsorize(x: np.ndarray, low=0.02, high=0.98) -> np.ndarray:
    lo = np.quantile(x, low); hi = np.quantile(x, high)
    return np.clip(x, lo, hi)

def smooth_all(pred: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
    out = {}
    for k,v in pred.items():
        out[k] = ema_smooth(winsorize(v, WINSOR_LOW, WINSOR_HIGH), EMA_ALPHA)
    return out

# ========================== 메인 ==========================
def main():
    personas = load_personas(PERSONA_JSON)
    df = load_products(PRODUCT_INFO_CSV)

    # ⬇️ 여기서 템플릿 자동 생성 (없을 때만 만듦)
    ensure_optional_templates(df)

    sub = pd.read_csv(SUBMISSION_TEMPLATE)
    month_cols = month_cols_from_submission(sub); n_months = len(month_cols)
    print(f"[INFO] products={len(df)}, months={n_months}, personas={len(personas)}, USE_LLM={USE_LLM}")

    prod_struct = build_product_struct(df)
    tam_overrides = load_tam_overrides(TAM_OVERRIDES_CSV, df)

    pred = simulate_year(df, prod_struct, personas, tam_overrides, n_months, BASE_TAM, BASE_OUTSIDE)
    pred_cal = hierarchical_calibration(pred, df, EARLY_ACTUALS_CSV, A0_CAT, B0_CAT, A0_PRD, B0_PRD, HARD_OVERRIDE_K)
    pred_smooth = smooth_all(pred_cal)

    out = sub.copy()
    for nm in df["product_name"].astype(str).tolist():
        if nm not in out["product_name"].astype(str).values:
            print(f"[WARN] 템플릿에 없는 제품: {nm} (스킵)")
            continue
        out.loc[out["product_name"].astype(str)==nm, month_cols] = np.round(pred_smooth[nm]).astype(int)

    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] saved {OUTPUT_CSV} shape={out.shape}")

if __name__ == "__main__":
    main()
