"""
predict_pipeline_v4.py
- chat.completions + JSON 강제(가능한 경우) / 실패 시 규칙 기반
- 페르소나 확장 필드(socioeconomic_tier, family_stage, region_birth_rate_index,
  household_size, working_mom, car_ownership)를 효용 함수에 반영
- TAM 오버라이드 학습(공개월 sMAPE 최소화) + 하향식/상향식 보정 + 블렌딩 + 스무딩
"""

import os, re, json, math, random
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ========================== 설정 ==========================
SEED = 42
random.seed(SEED); np.random.seed(SEED)

PERSONA_JSON         = "personas.json"
PRODUCT_INFO_CSV     = "product_info.csv"
SUBMISSION_TEMPLATE  = "sample_submission.csv"
EARLY_ACTUALS_CSV    = "early_actuals_template.csv"
TAM_OVERRIDES_CSV    = "tam_overrides_template.csv"
OUTPUT_CSV           = "submission5.csv"

USE_LLM = True
MODEL   = os.getenv("FT_MODEL_ID") or "gpt-4o-mini"

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

CATEGORY_SEASON_RULES = {
    "우유류": {"up":[6,7,8,9], "down":[12,1,2]},
    "조미소스": {"up":[1,2,9,10], "down":[]},
    "참치": {"up":[12,1,2], "down":[6,7,8]},
    "축산캔": {"up":[12,1,2], "down":[6,7,8]},
    "발효유": {"up":[6,7,8,9], "down":[12,1,2], "bump":0.10},
    "호상-중대용량": {"up":[6,7,8,9], "down":[12,1,2], "bump":0.20},
    "액상조미료": {"up":[1,2,9,10], "down":[], "bump":0.10},
}
MODEL_POWER = {"안유진": 0.25}

# ======================= 유틸/함수 ========================
def month_cols_from_submission(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("months_since_launch_")]

def softmax(x, temp=1.0):
    z = (x - x.max()) / max(temp,1e-9); e=np.exp(z); return e/(e.sum()+1e-9)

def clamp01(x, default=0.5):
    try: return float(np.clip(float(x),0.0,1.0))
    except: return default

def month_idx_to_calendar(idx:int)->int:
    return ((7 + idx - 1) % 12) + 1

def smape(y_true,y_pred):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
    num=np.abs(y_true-y_pred); den=(np.abs(y_true)+np.abs(y_pred))/2.0
    mask=(y_true>0); return 100.0*np.mean(num[mask]/np.clip(den[mask],1e-9,None))

def bass_curve(M:int,p=BASS_P,q=BASS_Q):
    t=np.arange(1,M+1,dtype=float)
    F=(1-np.exp(-(p+q)*t))/(1+(q/p)*np.exp(-(p+q)*t)+1e-9)
    w=np.clip(np.diff(np.concatenate([[0.0],F])),0,1)
    return w/(w.sum()+1e-9)

# =============== 데이터 로드 & 템플릿 생성 =================
def load_personas(path:str)->List[Dict[str,Any]]:
    with open(path,"r",encoding="utf-8") as f:
        arr=json.load(f)
    for p in arr:
        # month_weights 정규화
        mw=p.get("month_weights")
        if not isinstance(mw,list) or len(mw)!=12:
            mw=[1/12.0]*12
        else:
            s=sum(max(0.0,float(x)) for x in mw)
            mw=[(max(0.0,float(x))/s) if s>0 else 1/12.0 for x in mw]
        p["month_weights"]=mw
        p["weight"]=float(p.get("weight",1.0))
        # 새 필드 기본값
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
    if not os.path.exists(EARLY_ACTUALS_CSV) and not os.path.exists("early_actuals_template.csv"):
        ea=pd.DataFrame({"product_name":df_products["product_name"].astype(str).unique()})
        ea["months_since_launch_1"]=0; ea["months_since_launch_2"]=0
        ea.to_csv("early_actuals_template.csv",index=False,encoding="utf-8-sig")
        print("[INFO] created early_actuals_template.csv")
    if not os.path.exists(TAM_OVERRIDES_CSV) and not os.path.exists("tam_overrides_template.csv"):
        cats=df_products["category_level_1"].astype(str).unique()
        to=pd.DataFrame({"category_level_1":cats})
        for m in range(1,13): to[f"month_{m}"]=1.0
        to.to_csv("tam_overrides_template.csv",index=False,encoding="utf-8-sig")
        print("[INFO] created tam_overrides_template.csv")

# ==================== LLM 제품 특징 추출 ====================
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
    # JSON 강제 지원/비지원 모두 대비
    kwargs=dict(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0.2, max_tokens=6000)
    try:
        resp=client.chat.completions.create(**kwargs, response_format={"type":"json_object"})
    except TypeError:
        resp=client.chat.completions.create(**kwargs)
    txt=resp.choices[0].message.content.strip()

    # 안전 파싱
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

# ================== 시즌/효용/시뮬레이션 ==================
def category_season_bump(cat_keys:List[str], cal_month:int)->float:
    bump=0.0
    for key in cat_keys:
        rule=CATEGORY_SEASON_RULES.get(key,None)
        if not rule: continue
        base=float(rule.get("bump",0.06))
        if cal_month in rule.get("up",[]):   bump+=base
        if cal_month in rule.get("down",[]): bump-=base
    return bump

def persona_product_utility(persona:Dict[str,Any], prod:Dict[str,Any], month_idx:int)->float:
    f=prod["flags"]; cal_m=month_idx_to_calendar(month_idx)
    health=float(persona.get("health_consciousness",0.5))
    taste=str(persona.get("taste_preference","")).lower()
    brand=float(persona.get("brand_loyalty",0.5))
    promo=float(persona.get("promotion_sensitivity",0.5))
    price_sens=float(persona.get("price_sensitivity",0.5))
    risk=float(persona.get("risk_aversion",0.5))
    month_w=float(persona.get("month_weights",[1/12.0]*12)[cal_m-1])

    # 새 페르소나 변수
    socio=str(persona.get("socioeconomic_tier","middle")).lower()
    fam=str(persona.get("family_stage","single")).lower()
    birthI=float(persona.get("region_birth_rate_index",0.5))
    hh=int(persona.get("household_size",2))
    working_mom=bool(persona.get("working_mom",False))
    car_owner=bool(persona.get("car_ownership",True))

    u=0.0
    # 건강/저당/락토프리
    u+=0.20*health*(0.35*f["protein"]+0.25*f["low_sodium"]+0.20*f["lactose_free"]+0.20*f["low_sugar"])

    # 취향
    spicy_like=("spicy" in taste) or ("매운" in taste)
    coffee_like=("latte" in taste) or ("coffee" in taste) or ("커피" in taste)
    vanilla_like=("vanilla" in taste) or ("바닐라" in taste)
    if spicy_like:   u+=0.10*f["spicy"]
    else:            u-=0.05*f["spicy"]*risk
    if coffee_like:  u+=0.08*f["latte"]
    if vanilla_like: u+=0.06*f["vanilla"]

    # 참기름(브랜드/지속가능)
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
    u+=category_season_bump([prod["cat1"],prod["cat2"],prod["cat3"]], cal_m)

    # 가족/출산율 → 우유/발효유 계열 가산
    cat_keys=[prod["cat1"],prod["cat2"],prod["cat3"]]
    is_dairy=any(k in ["우유류","발효유","호상-중대용량"] for k in cat_keys)
    if is_dairy:
        kid_stage = 1.0 if fam in ["new_parent","young_kids"] else 0.0
        birth_push = (birthI-0.5)*2.0
        u += 0.10*(0.6*kid_stage+0.4*birth_push)

    # 대가족/차량보유/워킹맘 → 저장식품/편의식/커피 RTD 선호
    is_pantry = any(k in ["조미소스","액상조미료","참치","축산캔"] for k in cat_keys)
    if is_pantry:
        u += 0.06*((hh-2)/5.0)  # 가족 큰 편일수록 가산
        if car_owner: u += 0.02
    # 워킹맘이면 라떼/편의식 선호
    if working_mom:
        u += 0.05*f["latte"]
        if is_pantry: u += 0.03

    # 개인 월 패턴
    u+=0.20*month_w
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

def simulate_year(df:pd.DataFrame, prod_struct:Dict[str,Dict[str,Any]],
                  personas:List[Dict[str,Any]], tam_overrides, n_months=12,
                  base_tam=BASE_TAM, base_out=BASE_OUTSIDE)->Dict[str,np.ndarray]:
    names=df["product_name"].tolist()
    cat1s=df["category_level_1"].tolist()
    pred={nm:np.zeros(n_months) for nm in names}
    base=month_factors_base(n_months)
    launch_w=bass_curve(n_months,BASS_P,BASS_Q)
    P_weights=np.array([float(p.get("weight",1.0)) for p in personas]); P_weights/= (P_weights.sum()+1e-9)

    for m in range(n_months):
        tam=base_tam*base[m]; cal_m=month_idx_to_calendar(m)
        boosts=[CAT_TAM_AD_BOOST for nm,c1 in zip(names,cat1s) if cal_m in prod_struct[nm]["ad_months"]]
        if boosts: tam*=float(np.mean(boosts))
        if tam_overrides:
            factors=[tam_overrides.get((c1,cal_m),1.0) for c1 in cat1s]; tam*=float(np.mean(factors))
        outside=np.clip(base_out+0.03*np.sin(2*math.pi*m/12),0.18,0.72)

        U=np.zeros(len(names))
        for j,nm in enumerate(names):
            prod=prod_struct[nm]
            up=np.array([persona_product_utility(p,prod,m) for p in personas])
            U[j]=float((up*P_weights).sum())
        p=softmax(U,1.0)
        units=tam*(1-outside)*p
        units*=launch_w[m]
        for j,nm in enumerate(names): pred[nm][m]=units[j]
    return pred

# ================= 보정/블렌딩/스무딩 =================
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

# =========== TAM 오버라이드 학습(공개월 sMAPE) ===========
def fit_tam_overrides_by_smape(df, prod_struct, personas, base_tam, base_outside,
                               month_cols, public_month_idx, init_W=None,
                               cat_key="category_level_1", max_iter=50):
    if not os.path.exists(EARLY_ACTUALS_CSV):
        print("[WARN] no early_actuals.csv → skip TAM fitting"); return init_W or {}
    act=pd.read_csv(EARLY_ACTUALS_CSV)
    cats=df[cat_key].unique().tolist(); M=len(month_cols)
    W={(c,m+1):(init_W.get((c,m+1),1.0) if init_W else 1.0) for c in cats for m in range(M)}

    def run(Wdict):
        return simulate_year(df,prod_struct,personas,tam_overrides=Wdict,n_months=M,
                             base_tam=base_tam,base_out=base_outside)

    def metric(pred):
        y_true=[]; y_pred=[]
        for _,r in df.iterrows():
            nm=str(r["product_name"]); pv=[pred[nm][m] for m in public_month_idx]
            if nm in act["product_name"].astype(str).values:
                cols=[f"months_since_launch_{m+1}" for m in public_month_idx if f"months_since_launch_{m+1}" in act.columns]
                tv=act.loc[act["product_name"].astype(str)==nm, cols].values.astype(float).ravel()
                if len(tv)<len(pv): tv=np.pad(tv,(0,len(pv)-len(tv)),constant_values=0)
            else: tv=np.zeros(len(pv))
            y_true+=list(tv); y_pred+=list(pv)
        return smape(np.array(y_true),np.array(y_pred))

    best=metric(run(W))
    for it in range(max_iter):
        improved=0
        for c in cats:
            for cal_m in [m+1 for m in public_month_idx]:
                key=(c,cal_m); base=W[key]; loc_best=base; loc_metric=best
                for factor in [0.85,0.93,1.07,1.15]:
                    W[key]=max(0.2,min(2.5,base*factor))
                    mtr=metric(run(W))
                    if mtr<loc_metric: loc_metric=mtr; loc_best=W[key]
                if loc_best!=base:
                    W[key]=loc_best; best=loc_metric; improved+=1
        print(f"[TAM-FIT] iter={it:02d} improved={improved} smape={best:.4f}")
        if improved==0: break
    return W

# ============================ 메인 ============================
def main():
    personas=load_personas(PERSONA_JSON)
    df=load_products(PRODUCT_INFO_CSV)
    ensure_optional_templates(df)

    sub=pd.read_csv(SUBMISSION_TEMPLATE)
    month_cols=month_cols_from_submission(sub); n_months=len(month_cols)
    print(f"[INFO] products={len(df)}, months={n_months}, personas={len(personas)}, USE_LLM={USE_LLM}")

    prod_struct=build_product_struct(df)
    tam_overrides=load_tam_overrides(TAM_OVERRIDES_CSV, df)

    public_idx=list(range(0,6))  # 24.07~12 (공개 구간 가정)
    learned=fit_tam_overrides_by_smape(df,prod_struct,personas,BASE_TAM,BASE_OUTSIDE,
                                       month_cols,public_idx,init_W=tam_overrides)

    pred=simulate_year(df,prod_struct,personas,tam_overrides=learned or tam_overrides,
                       n_months=n_months,base_tam=BASE_TAM,base_out=BASE_OUTSIDE)
    pred_cal=hierarchical_calibration(pred,df,EARLY_ACTUALS_CSV,
                                      A0_CAT,B0_CAT,A0_PRD,B0_PRD,HARD_OVERRIDE_K)
    pred_blend=blend_predictions(pred_cal,n_months,blend=BLEND_WEIGHT)
    pred_smooth=smooth_all(pred_blend)

    out=sub.copy()
    for nm in df["product_name"].astype(str).tolist():
        if nm not in out["product_name"].astype(str).values:
            print(f"[WARN] 템플릿에 없는 제품: {nm}"); continue
        out.loc[out["product_name"].astype(str)==nm, month_cols]=np.round(pred_smooth[nm]).astype(int)
    out.to_csv(OUTPUT_CSV,index=False,encoding="utf-8-sig")
    print(f"[OK] saved {OUTPUT_CSV} shape={out.shape}")

    if learned:
        cats=sorted({k[0] for k in learned.keys()})
        to=pd.DataFrame({"category_level_1":cats})
        for m in range(1,13): to[f"month_{m}"]=[learned.get((c,m),1.0) for c in cats]
        to.to_csv("tam_overrides_learned.csv",index=False,encoding="utf-8-sig")
        print("[OK] saved tam_overrides_learned.csv")

if __name__=="__main__":
    main()
