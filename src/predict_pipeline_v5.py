"""
predict_pipeline_v5.py  (경로/파일 커스텀 적용판)

변경 사항(요청 반영)
- EARLY_ACTUALS_CSV       = "early_actuals_template.csv"
- TAM_OVERRIDES_CSV       = "tam_overrides_template.csv"
- EXOG_CSV                = "exogenous_overrides.csv"
- OUTPUT_CSV              = "submission6.csv"

기능 요약
1) 외생변수(exogenous_overrides.csv) 주입: 카테고리×월 가격/프로모션%로 TAM/효용 조정
2) TAM 학습: Simulated Annealing + 좌표하강 혼합 (공개월 교차검증)
3) 카테고리별 커스텀 효용 자동화
4) 감마-포아송 계층 보정 + 초기 K개월 하드오버라이드
5) Winsorize + EMA 스무딩 + naive 블렌딩
6) 제품명 정규화 병합(전각/반각/개행/공백 불일치 방지)
"""

import os, re, json, math, random, time, unicodedata
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ================= 기본 설정 =================
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ----- 경로/파일 (요청 반영) -----
PERSONA_JSON            = "personas.json"
PRODUCT_INFO_CSV        = "product_info.csv"
SUBMISSION_TEMPLATE     = "sample_submission.csv"

EARLY_ACTUALS_CSV       = "early_actuals_template.csv"   # 실제 값 파일(옵션; 템플릿 이름 그대로 사용)
TAM_OVERRIDES_CSV       = "tam_overrides_template.csv"   # 수동 보정(옵션; 템플릿 이름 그대로 사용)
EXOG_CSV                = "exogenous_overrides.csv"      # 외생변수(옵션: 가격/프로모션%)

OUTPUT_CSV              = "submission6.csv"
LEARNED_TAM_CSV         = "tam_overrides_learned_v5.csv"

# LLM 사용(제품 플래그 추출 시) — 키 없으면 자동 False
USE_LLM = True
MODEL   = os.getenv("FT_MODEL_ID") or "gpt-4o-mini"

# 시장/보정/광고/스무딩/런치커브/블렌딩
BASE_TAM             = 120_000
BASE_OUTSIDE         = 0.45
CAT_TAM_AD_BOOST     = 1.08
AD_BONUS_PER_PERSONA = 0.14
EMA_ALPHA            = 0.35
WINSOR_LOW, WINSOR_HIGH = 0.02, 0.98
A0_CAT, B0_CAT       = 0.3, 0.3
A0_PRD, B0_PRD       = 0.8, 0.8
HARD_OVERRIDE_K      = 1
BASS_P, BASS_Q       = 0.03, 0.38
BLEND_WEIGHT         = 0.20

# 공개 6개월 인덱스 및 K-fold (3-fold)
PUBLIC_MONTH_IDX     = [0,1,2,3,4,5]
KFOLD_SPLITS         = [( [2,3,4,5], [0,1] ),
                        ( [0,1,4,5], [2,3] ),
                        ( [0,1,2,3], [4,5] )]

# 카테고리 시즌 룰
CATEGORY_SEASON_RULES = {
    "우유류": {"up":[6,7,8,9], "down":[12,1,2]},
    "조미소스": {"up":[1,2,9,10], "down":[]},
    "참치": {"up":[12,1,2], "down":[6,7,8]},
    "축산캔": {"up":[12,1,2], "down":[6,7,8]},
    "발효유": {"up":[6,7,8,9], "down":[12,1,2], "bump":0.10},
    "호상-중대용량": {"up":[6,7,8,9], "down":[12,1,2], "bump":0.20},
    "액상조미료": {"up":[1,2,9,10], "down":[], "bump":0.10},
}

# 광고 모델 파워(예시)
MODEL_POWER = {"안유진": 0.25}

# 카테고리 커스텀 효용 가중
CATEGORY_CUSTOM_WEIGHTS = {
    "발효유":            {"protein": +0.05, "low_sugar": +0.05},
    "호상-중대용량":     {"protein": +0.05, "low_sugar": +0.05},
    "참치":             {"pantry_household": +0.04, "pantry_car": +0.02, "pantry_workingmom": +0.03},
    "축산캔":            {"pantry_household": +0.04, "pantry_car": +0.02, "pantry_workingmom": +0.03},
    "액상조미료":        {"sesame": +0.03},
    "조미소스":          {"sesame": +0.03},
}

# ========= 유틸 =========
def month_cols_from_submission(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("months_since_launch_")]

def softmax(x, temp=1.0):
    z = (x - x.max()) / max(temp,1e-9); e=np.exp(z); return e/(e.sum()+1e-9)

def clamp01(x, default=0.5):
    try: return float(np.clip(float(x),0.0,1.0))
    except: return default

def month_idx_to_calendar(idx:int)->int:
    # (0→7월, 11→6월) 형태의 캘린더월(1~12)
    return ((7 + idx - 1) % 12) + 1

def smape(y_true,y_pred):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
    num=np.abs(y_true-y_pred); den=(np.abs(y_true)+np.abs(y_pred))/2.0
    mask=(y_true>0)
    if mask.sum()==0: return 100.0
    return 100.0*np.mean(num[mask]/np.clip(den[mask],1e-9,None))

def bass_curve(M:int,p=BASS_P,q=BASS_Q):
    t=np.arange(1,M+1,dtype=float)
    F=(1-np.exp(-(p+q)*t))/(1+(q/p)*np.exp(-(p+q)*t)+1e-9)
    w=np.clip(np.diff(np.concatenate([[0.0],F])),0,1)
    return w/(w.sum()+1e-9)

def _norm_name(s: str) -> str:
    """제품명 정규화(전각/반각/개행/앞뒤공백 제거)"""
    return unicodedata.normalize("NFKC", str(s)).strip().replace("\n","").replace("\r","")

# ========= 데이터 로드 & 템플릿 생성 =========
def load_personas(path:str)->List[Dict[str,Any]]:
    with open(path,"r",encoding="utf-8") as f:
        arr=json.load(f)
    for p in arr:
        mw=p.get("month_weights")
        if not isinstance(mw,list) or len(mw)!=12:
            mw=[1/12.0]*12
        else:
            s=sum(max(0.0,float(x)) for x in mw)
            mw=[(max(0.0,float(x))/s) if s>0 else 1/12.0 for x in mw]
        p["month_weights"]=mw
        p["weight"]=float(p.get("weight",1.0))
        # 확장 속성 기본값(호환성)
        p.setdefault("socioeconomic_tier","middle")
        p.setdefault("family_stage","single")
        p["region_birth_rate_index"]=float(p.get("region_birth_rate_index",0.5))
        p["household_size"]=int(p.get("household_size",2))
        p["working_mom"]=bool(p.get("working_mom",False))
        p["car_ownership"]=bool(p.get("car_ownership",True))
    return arr

def load_products(path:str)->pd.DataFrame:
    df=pd.read_csv(path)
    need=["product_name","product_feature","category_level_1","category_level_2","category_level_3"]
    miss=[c for c in need if c not in df.columns]
    if miss: raise ValueError(f"product_info.csv missing: {miss}")
    return df

def ensure_optional_templates(df_products: pd.DataFrame):
    """
    템플릿 없으면 생성.
    (현재 EARLY_ACTUALS_CSV, TAM_OVERRIDES_CSV가 템플릿 파일명을 가리키므로
     없는 경우 기본 템플릿을 생성해 둔다.)
    """
    # early_actuals 템플릿
    if not os.path.exists(EARLY_ACTUALS_CSV):
        ea=pd.DataFrame({"product_name":df_products["product_name"].astype(str).unique()})
        ea["months_since_launch_1"]=0; ea["months_since_launch_2"]=0
        ea.to_csv(EARLY_ACTUALS_CSV, index=False, encoding="utf-8-sig")
        print(f"[INFO] created {EARLY_ACTUALS_CSV}")

    # tam_overrides 템플릿
    if not os.path.exists(TAM_OVERRIDES_CSV):
        cats=df_products["category_level_1"].astype(str).unique()
        to=pd.DataFrame({"category_level_1":cats})
        for m in range(1,13): to[f"month_{m}"]=1.0
        to.to_csv(TAM_OVERRIDES_CSV, index=False, encoding="utf-8-sig")
        print(f"[INFO] created {TAM_OVERRIDES_CSV}")

    # 외생변수 템플릿
    if not os.path.exists(EXOG_CSV) and not os.path.exists("exogenous_overrides_template.csv"):
        cats=df_products["category_level_1"].astype(str).unique()
        ex=pd.DataFrame({"category_level_1":cats})
        for m in range(1,13):
            ex[f"price_{m}"]=100.0    # 100 = 기준, 110 = +10%
            ex[f"promo_{m}"]=100.0
        ex.to_csv("exogenous_overrides_template.csv", index=False, encoding="utf-8-sig")
        print("[INFO] created exogenous_overrides_template.csv")

# ========= LLM 제품 플래그 추출(옵션) =========
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY: USE_LLM=False
if USE_LLM:
    from openai import OpenAI
    client=OpenAI(api_key=OPENAI_API_KEY)

KEYWORDS={
    "protein":["고단백","단백질","protein"],
    "lactose_free":["락토프리","유당 zero","유당zero","lactose"],
    "low_sodium":["저나트륨","나트륨 낮"],
    "low_sugar":["저당","당 줄인","low sugar","무가당"],
    "spicy":["매콤","매운"],
    "vanilla":["바닐라"],
    "latte":["라떼","카페라떼"],
    "sesame_oil":["참기름"],
    "premium":["프리미엄","명가","고급"],
    "tv_ad":["TV","티비","방송광고"],
    "social_ad":["YouTube","SNS","유튜브","소셜"],
    "elevator_ad":["엘리베이터"],
    "festival":["명절","추석","설"],
}

def parse_months_from_text(text:str)->List[int]:
    t=text.replace(" ",""); months=set()
    for m1,m2 in re.findall(r"(\d{1,2})\s*[-~]\s*(\d{1,2})\s*월",t):
        a,b=int(m1),int(m2)
        if 1<=a<=12 and 1<=b<=12:
            months.update(range(a,b+1) if a<=b else list(range(a,13))+list(range(1,b+1)))
    for m in re.findall(r"(\d{1,2})월",t):
        k=int(m); 1<=k<=12 and months.add(k)
    return sorted(months)

def rule_extract_product(r: pd.Series)->Dict[str,Any]:
    name=str(r["product_name"]); feat=str(r["product_feature"]).lower()
    cat1,cat2,cat3=str(r["category_level_1"]),str(r["category_level_2"]),str(r["category_level_3"])
    flags={k:0.0 for k in ["protein","lactose_free","low_sodium","low_sugar","spicy","vanilla","latte",
                            "sesame_oil","premium","tv_ad","social_ad","elevator_ad","festival","model_star_power"]}
    for k,words in KEYWORDS.items():
        for w in words:
            if w.lower() in feat: flags[k]=1.0; break
    for model_name,power in MODEL_POWER.items():
        if model_name in r["product_feature"]: flags["model_star_power"]=power
    ad_months=parse_months_from_text(r["product_feature"])
    return {"product_name":name,"flags":flags,"ad_months":ad_months,"cat1":cat1,"cat2":cat2,"cat3":cat3}

def llm_extract_all(df: pd.DataFrame)->List[Dict[str,Any]]:
    rows=[{"product_name":x["product_name"],"product_feature":x["product_feature"],
           "category_level_1":x["category_level_1"],"category_level_2":x["category_level_2"],
           "category_level_3":x["category_level_3"]} for _,x in df.iterrows()]
    tab=json.dumps(rows,ensure_ascii=False)
    prompt=f"""
다음 JSON 테이블의 각 제품을 분석해, 아래 형식의 JSON 객체만 출력:
{{"data":[{{"product_name":..,"flags":{{"protein":0..1,"lactose_free":0..1,"low_sodium":0..1,"low_sugar":0..1,"spicy":0..1,"vanilla":0..1,"latte":0..1,"sesame_oil":0..1,"premium":0..1,"tv_ad":0..1,"social_ad":0..1,"elevator_ad":0..1,"festival":0..1,"model_star_power":0..1}},"ad_months":[1..12],"cat1":"..","cat2":"..","cat3":".."}},...]}}
설명/코드펜스 금지.
입력:
{tab}
"""
    kwargs=dict(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0.2, max_tokens=6000)
    try:
        resp=client.chat.completions.create(**kwargs, response_format={"type":"json_object"})
    except TypeError:
        resp=client.chat.completions.create(**kwargs)
    txt=resp.choices[0].message.content.strip()

    def _clean(t):
        t=t.replace("\u201c",'"').replace("\u201d",'"').replace("\u2018",'"').replace("\u2019",'"')
        t=re.sub(r",\s*([}\]])", r"\1", t); return t
    s=txt.find("{"); e=txt.rfind("}")
    obj=json.loads(_clean(txt[s:e+1]))
    data=obj.get("data", obj)

    out=[]
    for d in data:
        d.setdefault("flags",{})
        for k in ["protein","lactose_free","low_sodium","low_sugar","spicy","vanilla","latte",
                  "sesame_oil","premium","tv_ad","social_ad","elevator_ad","festival","model_star_power"]:
            d["flags"][k]=clamp01(d["flags"].get(k,0.0))
        d.setdefault("ad_months",[])
        d["ad_months"]=[int(x) for x in d["ad_months"] if 1<=int(x)<=12]
        for k in ["cat1","cat2","cat3"]: d[k]=d.get(k,"")
        out.append(d)
    return out

def build_product_struct(df: pd.DataFrame)->Dict[str,Dict[str,Any]]:
    if USE_LLM:
        try:
            ex=llm_extract_all(df)
        except Exception as e:
            print("[WARN] LLM 실패 → 규칙 사용:", e)
            ex=[rule_extract_product(r) for _,r in df.iterrows()]
    else:
        ex=[rule_extract_product(r) for _,r in df.iterrows()]
    return {d["product_name"]:d for d in ex}

# ========= 외생변수 로드 =========
def load_exogenous(path:str, df:pd.DataFrame):
    """
    exogenous_overrides.csv (옵션)
    - columns: category_level_1, price_1..12(%) , promo_1..12(%)
    - 100 = 기준, 110 = +10%, 90 = -10%
    """
    if not os.path.exists(path): return {}
    ex=pd.read_csv(path)
    out={}
    for _,r in ex.iterrows():
        cat=str(r["category_level_1"])
        for m in range(1,13):
            price=float(r.get(f"price_{m}",100.0))/100.0
            promo=float(r.get(f"promo_{m}",100.0))/100.0
            out[(cat,m)]={"price":price,"promo":promo}
    return out

# ========= 시즌/효용 =========
def category_season_bump(cat_keys:List[str], cal_month:int)->float:
    bump=0.0
    for key in cat_keys:
        rule=CATEGORY_SEASON_RULES.get(key,None)
        if not rule: continue
        base=float(rule.get("bump",0.06))
        if cal_month in rule.get("up",[]):   bump+=base
        if cal_month in rule.get("down",[]): bump-=base
    return bump

def apply_custom_weights(cat_keys:List[str], contrib:Dict[str,float])->float:
    """
    카테고리 커스텀 가중을 한 바구니로 모아 스칼라로 반환
    contrib 키:
      protein/low_sugar/sesame/pantry_household/pantry_car/pantry_workingmom
    """
    gain=0.0
    for key in cat_keys:
        w=CATEGORY_CUSTOM_WEIGHTS.get(key,{})
        for k,v in w.items():
            gain += v*contrib.get(k,0.0)
    return gain

def persona_product_utility(persona:Dict[str,Any], prod:Dict[str,Any], month_idx:int)->float:
    f=prod["flags"]; cal_m=month_idx_to_calendar(month_idx)
    health=float(persona.get("health_consciousness",0.5))
    taste=str(persona.get("taste_preference","")).lower()
    brand=float(persona.get("brand_loyalty",0.5))
    promo=float(persona.get("promotion_sensitivity",0.5))
    price_sens=float(persona.get("price_sensitivity",0.5))
    risk=float(persona.get("risk_aversion",0.5))
    month_w=float(persona.get("month_weights",[1/12.0]*12)[cal_m-1])

    socio=str(persona.get("socioeconomic_tier","middle")).lower()
    fam=str(persona.get("family_stage","single")).lower()
    birthI=float(persona.get("region_birth_rate_index",0.5))
    hh=int(persona.get("household_size",2))
    working_mom=bool(persona.get("working_mom",False))
    car_owner=bool(persona.get("car_ownership",True))

    u=0.0
    # 건강/영양
    base_health = (0.35*f["protein"]+0.25*f["low_sodium"]+0.20*f["lactose_free"]+0.20*f["low_sugar"])
    u += 0.20*health*base_health

    # 취향
    spicy_like=("spicy" in taste) or ("매운" in taste)
    coffee_like=("latte" in taste) or ("coffee" in taste) or ("커피" in taste)
    vanilla_like=("vanilla" in taste) or ("바닐라" in taste)
    if spicy_like:   u+=0.10*f["spicy"]
    else:            u-=0.05*f["spicy"]*risk
    if coffee_like:  u+=0.08*f["latte"]
    if vanilla_like: u+=0.06*f["vanilla"]

    # 참기름/브랜드/지속가능
    sust=float(persona.get("sustainability_preference",0.5))
    u+=0.05*f["sesame_oil"]*(0.5*brand+0.5*sust)

    # 프리미엄 × 사회계층
    socio_premium={"low":-0.06,"middle":0.0,"high":0.08}.get(socio,0.0)
    u+=0.13*f["premium"]*(0.6*brand+0.4*(1-price_sens))+socio_premium

    # 광고월
    if cal_m in prod["ad_months"]:
        ad_w=1.0+0.15*f["tv_ad"]+0.12*f["social_ad"]+0.08*f["elevator_ad"]
        u+=AD_BONUS_PER_PERSONA*promo*ad_w

    # 모델파워 & 시즌
    u+=0.12*f["model_star_power"]
    cat_keys=[prod["cat1"],prod["cat2"],prod["cat3"]]
    u+=category_season_bump(cat_keys, cal_m)

    # 유제품 계열: 신생아/영유아 가정 + 출산율 높은 지역
    is_dairy=any(k in ["우유류","발효유","호상-중대용량"] for k in cat_keys)
    if is_dairy:
        kid_stage = 1.0 if fam in ["new_parent","young_kids"] else 0.0
        birth_push = (birthI-0.5)*2.0
        u += 0.10*(0.6*kid_stage+0.4*birth_push)

    # 저장/편의식 계열: 대가족/차량/워킹맘
    is_pantry = any(k in ["조미소스","액상조미료","참치","축산캔"] for k in cat_keys)
    pantry_household = max(0.0,(hh-2)/5.0) if is_pantry else 0.0
    pantry_car       = 1.0 if (is_pantry and car_owner) else 0.0
    pantry_working   = 1.0 if (is_pantry and working_mom) else 0.0
    if is_pantry:
        u += 0.06*pantry_household
        if car_owner: u += 0.02
    if working_mom:
        u += 0.05*f["latte"]
        if is_pantry: u += 0.03

    # 카테고리별 커스텀 가중 자동 적용
    custom_gain = apply_custom_weights(
        cat_keys,
        {
            "protein": base_health*0.5,
            "low_sugar": f["low_sugar"],
            "sesame": f["sesame_oil"],
            "pantry_household": pantry_household,
            "pantry_car": pantry_car,
            "pantry_workingmom": pantry_working,
        }
    )
    u += custom_gain

    # 개인 월 패턴
    u += 0.20*month_w
    return float(u)

def month_factors_base(n:int)->np.ndarray:
    t=np.arange(n); return 1.0+0.08*np.sin(2*math.pi*t/12 - math.pi/6)

def load_tam_overrides(path:str, df:pd.DataFrame)->Dict[Tuple[str,int],float]:
    if not os.path.exists(path): return {}
    src=pd.read_csv(path); out={}
    for _,r in src.iterrows():
        cat1=str(r["category_level_1"])
        for m in range(1,13):
            col=f"month_{m}"
            if col in src.columns:
                try: out[(cat1,m)]=float(r[col])
                except: pass
    return out

# ========= 시뮬레이션 =========
def simulate_year(df:pd.DataFrame, prod_struct:Dict[str,Dict[str,Any]],
                  personas:List[Dict[str,Any]], tam_overrides, exog_dict,
                  n_months=12, base_tam=BASE_TAM, base_out=BASE_OUTSIDE)->Dict[str,np.ndarray]:
    names=df["product_name"].tolist()
    cat1s=df["category_level_1"].tolist()
    pred={nm:np.zeros(n_months) for nm in names}
    base=month_factors_base(n_months)
    launch_w=bass_curve(n_months,BASS_P,BASS_Q)
    P_weights=np.array([float(p.get("weight",1.0)) for p in personas]); P_weights/= (P_weights.sum()+1e-9)

    for m in range(n_months):
        tam=base_tam*base[m]; cal_m=month_idx_to_calendar(m)

        # 광고월 평균 부스트
        boosts=[CAT_TAM_AD_BOOST for nm,c1 in zip(names,cat1s) if cal_m in prod_struct[nm]["ad_months"]]
        if boosts: tam*=float(np.mean(boosts))

        # TAM 오버라이드(카테고리×월)
        if tam_overrides:
            factors=[tam_overrides.get((c1,cal_m),1.0) for c1 in cat1s]; tam*=float(np.mean(factors))

        # 외생변수: 프로모션%는 TAM에, 가격%는 outside/효용에 반영
        if exog_dict:
            promo_fs=[exog_dict.get((c1,cal_m),{"promo":1.0}).get("promo",1.0) for c1 in cat1s]
            tam *= float(np.mean(promo_fs))

        # 가격%에 따라 외부/효용 미세 조정 (가격↑ → outside↑, 효용↓)
        outside=np.clip(base_out+0.03*np.sin(2*math.pi*m/12),0.18,0.72)
        price_fs=[exog_dict.get((c1,cal_m),{"price":1.0}).get("price",1.0) for c1 in cat1s] if exog_dict else [1.0]
        price_effect=float(np.mean(price_fs))
        outside = np.clip(outside * (0.98 + 0.2*(price_effect-1.0)), 0.15, 0.80)  # ±0.2 범위 영향

        # 효용→점유율
        U=np.zeros(len(names))
        for j,nm in enumerate(names):
            prod=prod_struct[nm]
            up=np.array([persona_product_utility(p,prod,m) for p in personas])
            # 가격 높은 달 → 효용 감쇠
            U[j]=float((up*P_weights).sum()) * (1.0 - 0.10*(price_effect-1.0))

        p=softmax(U,1.0)
        units=tam*(1-outside)*p
        units*=launch_w[m]
        for j,nm in enumerate(names): pred[nm][m]=units[j]
    return pred

# ========= 보정/스무딩/블렌딩 =========
def gamma_scale(y_sum,yh_sum,A0,B0): return (A0+y_sum)/(B0+yh_sum+1e-9)

def hierarchical_calibration(pred, df, early_path=EARLY_ACTUALS_CSV,
                             A0_cat=A0_CAT,B0_cat=B0_CAT,A0_prd=A0_PRD,B0_prd=B0_PRD,
                             hard_override_k=HARD_OVERRIDE_K):
    if not os.path.exists(early_path): return pred
    act=pd.read_csv(early_path)
    K=sum([c in act.columns for c in ["months_since_launch_1","months_since_launch_2"]]); K=max(1,min(K,2))
    cols=[f"months_since_launch_{i}" for i in range(1, K+1)]
    out={k:v.copy() for k,v in pred.items()}

    # 카테고리 보정
    for cat in df["category_level_1"].unique():
        names=df.loc[df["category_level_1"]==cat,"product_name"].tolist()
        y_sum=yh_sum=0.0
        for nm in names:
            if nm in act["product_name"].astype(str).values:
                y=act.loc[act["product_name"].astype(str)==nm, cols].values.astype(float).ravel().sum()
            else: y=0.0
            y_sum+=y; yh_sum+=out[nm][:K].sum()
        m_cat=gamma_scale(y_sum,yh_sum,A0_cat,B0_cat)
        for nm in names: out[nm]*=m_cat

    # 제품 보정 + 하드오버라이드
    for nm in out.keys():
        if nm in act["product_name"].astype(str).values:
            y=act.loc[act["product_name"].astype(str)==nm, cols].values.astype(float).ravel().sum()
            yh=out[nm][:K].sum()
            m_prd=gamma_scale(y,yh,A0_prd,B0_prd); out[nm]*=m_prd
            if hard_override_k>0:
                yrow=act.loc[act["product_name"].astype(str)==nm, cols].values.astype(float).ravel()
                for i in range(min(hard_override_k,K)): out[nm][i]=yrow[i]
    return out

def ema_smooth(x,alpha=EMA_ALPHA):
    y=np.zeros_like(x,dtype=float); y[0]=x[0]
    for i in range(1,len(x)): y[i]=alpha*x[i]+(1-alpha)*y[i-1]
    return y

def winsorize(x,low=WINSOR_LOW,high=WINSOR_HIGH):
    lo=np.quantile(x,low); hi=np.quantile(x,high); return np.clip(x,lo,hi)

def smooth_all(pred):
    return {k: ema_smooth(winsorize(v), EMA_ALPHA) for k,v in pred.items()}

def seasonal_naive_baseline(n): w=np.ones(n); return w/w.sum()

def blend_predictions(pred_sim, n_months, blend=BLEND_WEIGHT):
    out={}
    for nm,arr in pred_sim.items():
        base=seasonal_naive_baseline(n_months)*arr.sum()
        out[nm]=(1.0-blend)*arr+blend*base
    return out

# ========= TAM 학습(좌표하강 + Simulated Annealing) =========
def fit_tam_overrides_SA(df, prod_struct, personas, exog_dict,
                         base_tam, base_outside, month_cols,
                         public_month_idx, init_W=None,
                         cat_key="category_level_1", max_iter=60, seed=SEED):
    rng=np.random.default_rng(seed)
    if not os.path.exists(EARLY_ACTUALS_CSV):
        print("[WARN] no early_actuals_template.csv → skip TAM learning"); return init_W or {}

    act=pd.read_csv(EARLY_ACTUALS_CSV)
    cats=df[cat_key].unique().tolist(); M=len(month_cols)
    W={(c,m+1):(init_W.get((c,m+1),1.0) if init_W else 1.0) for c in cats for m in range(M)}

    def run(Wdict):
        return simulate_year(df,prod_struct,personas,tam_overrides=Wdict,exog_dict=exog_dict,
                             n_months=M, base_tam=base_tam, base_out=base_outside)

    def metric(pred, eval_month_idx):
        y_true=[]; y_pred=[]
        for _,r in df.iterrows():
            nm=str(r["product_name"]); pv=[pred[nm][m] for m in eval_month_idx]
            if nm in act["product_name"].astype(str).values:
                cols=[f"months_since_launch_{m+1}" for m in eval_month_idx if f"months_since_launch_{m+1}" in act.columns]
                tv=act.loc[act["product_name"].astype(str)==nm, cols].values.astype(float).ravel()
                if len(tv)<len(pv): tv=np.pad(tv,(0,len(pv)-len(tv)),constant_values=0)
            else: tv=np.zeros(len(pv))
            y_true+=list(tv); y_pred+=list(pv)
        return smape(np.array(y_true),np.array(y_pred))

    # K-fold 교차검증
    Ws=[]; scores=[]
    for fold,(train_idx, valid_idx) in enumerate(KFOLD_SPLITS, start=1):
        W_fold=W.copy()
        T0, Tmin = 1.0, 0.02
        for it in range(max_iter):
            # 1) 좌표하강 스텝
            improved=0
            for c in cats:
                for cal_m in [m+1 for m in train_idx]:
                    key=(c,cal_m); base=W_fold[key]; loc_best=base
                    current_pred = run(W_fold); current_m = metric(current_pred, valid_idx)

                    for factor in [0.85,0.93,1.07,1.15]:
                        W_fold[key]=max(0.2,min(2.5,base*factor))
                        mtr=metric(run(W_fold), valid_idx)
                        if mtr < current_m:
                            current_m=mtr; loc_best=W_fold[key]; improved+=1
                    W_fold[key]=loc_best

            # 2) SA 무작위 제안
            T = T0*( (Tmin/T0) ** (it/max_iter) )
            for _ in range(len(cats)):
                c = rng.choice(cats)
                cal_m = rng.choice([m+1 for m in train_idx])
                key=(c,cal_m); old=W_fold[key]
                sigma=0.15*T
                prop = max(0.2, min(2.5, old * math.exp(rng.normal(0, sigma))))
                pred_old=run(W_fold); m_old=metric(pred_old, valid_idx)
                W_fold[key]=prop
                pred_new=run(W_fold); m_new=metric(pred_new, valid_idx)
                if (m_new < m_old) or (rng.random() < math.exp( -(m_new-m_old)/(T+1e-9) )):
                    pass
                else:
                    W_fold[key]=old

            if it%10==0:
                val = metric(run(W_fold), valid_idx)
                print(f"[TAM-SA] fold={fold} it={it:02d} valid_sMAPE={val:.4f} (improved:{improved})")

        final_val = metric(run(W_fold), valid_idx)
        Ws.append(W_fold); scores.append(final_val)
        print(f"[TAM-SA] fold={fold} done. valid_sMAPE={final_val:.4f}")

    # fold 가중 평균 (성능 역수 가중)
    inv = np.array([1.0/max(s,1e-6) for s in scores]); inv /= inv.sum()
    W_avg={}
    for key in Ws[0].keys():
        W_avg[key]=float(np.sum([inv[i]*Ws[i][key] for i in range(len(Ws))]))
    print("[TAM-SA] blended across folds.")
    return W_avg

# ========================= 메인 =========================
def main():
    personas=load_personas(PERSONA_JSON)
    df=load_products(PRODUCT_INFO_CSV)
    ensure_optional_templates(df)

    sub=pd.read_csv(SUBMISSION_TEMPLATE)
    month_cols=month_cols_from_submission(sub); n_months=len(month_cols)
    print(f"[INFO] products={len(df)}, months={n_months}, personas={len(personas)}, USE_LLM={USE_LLM}")

    prod_struct=build_product_struct(df)
    tam_overrides=load_tam_overrides(TAM_OVERRIDES_CSV, df)
    exog=load_exogenous(EXOG_CSV, df)

    # TAM 학습 (공개월 교차검증 기반)
    learned=fit_tam_overrides_SA(df,prod_struct,personas,exog,
                                 BASE_TAM,BASE_OUTSIDE,month_cols,
                                 PUBLIC_MONTH_IDX,init_W=tam_overrides)

    # ---------- 시뮬레이션 ----------
    pred=simulate_year(df,prod_struct,personas,tam_overrides=learned or tam_overrides,
                       exog_dict=exog, n_months=n_months,base_tam=BASE_TAM,base_out=BASE_OUTSIDE)

    # ---------- 보정/블렌딩/스무딩 ----------
    pred_cal=hierarchical_calibration(pred,df,EARLY_ACTUALS_CSV,
                                      A0_CAT,B0_CAT,A0_PRD,B0_PRD,HARD_OVERRIDE_K)
    pred_blend=blend_predictions(pred_cal,n_months,blend=BLEND_WEIGHT)
    pred_smooth=smooth_all(pred_blend)

    # ---------- 안전 대입(제품명 정규화 병합) ----------
    out = sub.copy()
    out["pn_key"] = out["product_name"].map(_norm_name)
    df["pn_key"]  = df["product_name"].map(_norm_name)

    rows = []
    for nm in df["product_name"].astype(str).tolist():
        key = _norm_name(nm)
        if nm not in pred_smooth:
            print(f"[WARN] 예측 dict에 없는 제품: {nm}")
            continue
        rows.append([key] + list(np.round(pred_smooth[nm]).astype(int)))
    pred_df = pd.DataFrame(rows, columns=["pn_key"] + month_cols)

    merged = out.merge(pred_df, on="pn_key", how="left", suffixes=("","_pred"))

    filled = 0
    for c in month_cols:
        if c + "_pred" in merged.columns:
            filled += merged[c + "_pred"].notna().sum()
            merged[c] = merged[c + "_pred"].fillna(merged[c]).astype(int)

    print(f"[INFO] 대입 성공 건수(행×월 합계): {filled}")

    merged.drop(columns=[c for c in merged.columns if c.endswith("_pred")] + ["pn_key"], inplace=True)
    merged.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] saved {OUTPUT_CSV} shape={merged.shape}")

    # ---------- 학습된 TAM 계수 저장 ----------
    if learned:
        cats=sorted({k[0] for k in learned.keys()})
        to=pd.DataFrame({"category_level_1":cats})
        for m in range(1,13): to[f"month_{m}"]=[learned.get((c,m),1.0) for c in cats]
        to.to_csv(LEARNED_TAM_CSV,index=False,encoding="utf-8-sig")
        print(f"[OK] saved {LEARNED_TAM_CSV}")

    # ---------- 디버그 로그 ----------
    tot_pred = sum([float(np.sum(v)) for v in pred_smooth.values()])
    print("[DEBUG] total predicted units:", round(tot_pred,2))
    print("[DEBUG] unique products in df/sub:", df['product_name'].nunique(), "/", sub['product_name'].nunique())

if __name__=="__main__":
    main()
