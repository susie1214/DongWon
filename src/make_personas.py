import os, json, math, time, re
from string import Template
from dotenv import load_dotenv

# ---------------- Config ----------------
TOTAL_PERSONAS     = 200       # 총 인원
BATCH_SIZE         = 10        # 배치당 인원(작을수록 안전)
TEMPERATURE        = 0.2
MAX_RETRIES        = 3
MODEL_MAX_TOKENS   = 7000      # gpt-4o-mini 출력 한도(16k) 이내 여유
MODEL_NAME_DEFAULT = "gpt-4o-mini"

# ------------- OpenAI Client -------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("FT_MODEL_ID") or MODEL_NAME_DEFAULT
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY가 .env에 없습니다.")

# 구버전/신버전 호환 클라이언트 (chat.completions 사용)
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# -------- Prompt (Template: $n만 치환) --------
PROMPT_TPL = Template(r"""
아래 조건을 만족하는 JSON 객체만 생성하세요. 설명/코드펜스 금지.

반환 형식(정확히 이 구조):
{"data":[ ... ]}   // data는 $n개 원소(배열)

각 원소 = 한국 소비자 페르소나 1명. 스키마:
- name: string
- gender: "male"|"female"
- age: int
- region_kr: string
- channel_code: "ON"|"OM"|"OF"
- socioeconomic_tier: "low"|"middle"|"high"
- family_stage: "single"|"couple_no_kids"|"new_parent"|"young_kids"|"teens"|"multi_gen"
- region_birth_rate_index: number 0~1
- (옵션) household_size: integer 1~7
- (옵션) working_mom: boolean
- (옵션) car_ownership: boolean
- price_sensitivity, promotion_sensitivity, brand_loyalty,
  innovation_seeking, review_dependence, sustainability_preference, risk_aversion: number 0~1 (소수 2자리)
- baseline_purchase_frequency: number 0.70~1.00 (소수 2자리)
- taste_preference: "latte"|"vanilla"|"spicy"|"neutral"
- low_sugar_preference: number 0~1 (소수 2자리)
- month_weights: number[12] (합=1; 여름(6~9) 유제품↑, 1·2·9·10·12월 조미/통조림↑ 약하게 반영; 소수 3자리)
- weight: number 0.50~2.00 (소수 2자리)

분포 가이드(권장): channel_code 비율 ON 40~60%, OM 20~40%, OF 10~30%.
문자열은 큰따옴표(")만 사용, 마지막 원소 뒤 쉼표 금지.
""")

# ------------- JSON 파서(견고) -------------
def _clean_json_text(txt: str) -> str:
    # 스마트따옴표 제거, 트레일링 콤마 제거
    t = txt.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", '"').replace("\u2019", '"')
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return t

def parse_json_object(txt: str) -> dict:
    t = txt.strip()
    # 최상위 { ... } 범위만 추출
    s = t.find("{")
    if s < 0:
        raise ValueError("JSON 객체 시작 '{' 를 찾지 못했습니다.")
    depth = 0; end = -1; in_str = False; esc = False
    for i, ch in enumerate(t[s:], s):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"':  in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i; break
    if end < 0:
        raise ValueError("닫는 '}' 를 찾지 못했습니다.")
    raw = _clean_json_text(t[s:end+1])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # json5가 있다면 좀 더 관대하게
        try:
            import json5
            return json5.loads(raw)
        except Exception as e:
            with open("personas_raw_failed.txt","w",encoding="utf-8") as f:
                f.write(txt)
            raise e

# ------------- LLM 호출 -------------
def call_llm_for_batch(n: int, batch_idx: int, try_idx: int) -> list:
    prompt = PROMPT_TPL.substitute(n=n)
    # 일부 구버전 SDK는 response_format 인자를 지원하지 않음 → try/except로 분기
    kwargs = dict(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=TEMPERATURE,
        max_tokens=MODEL_MAX_TOKENS,
    )
    try:
        resp = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},  # 신버전에서만 동작
        )
    except TypeError:
        resp = client.chat.completions.create(**kwargs)

    txt = resp.choices[0].message.content.strip()
    # 디버그 저장
    open(f"personas_raw_batch_{batch_idx}_try{try_idx}.txt","w",encoding="utf-8").write(txt)
    obj = parse_json_object(txt)
    data = obj["data"] if isinstance(obj, dict) else obj
    if not isinstance(data, list):
        raise ValueError("JSON의 'data'가 배열이 아닙니다.")
    return data

# ---------- 정규화/기본값 ----------
def normalize_persona(p: dict) -> dict:
    # month_weights
    mw = p.get("month_weights")
    if not isinstance(mw, list) or len(mw) != 12:
        mw = [1/12.0]*12
    else:
        s = sum(max(0.0, float(x)) for x in mw)
        mw = [(max(0.0, float(x))/s) if s>0 else 1/12.0 for x in mw]
    p["month_weights"] = mw
    # weight
    p["weight"] = float(p.get("weight", 1.0))
    # 새 필드 기본값 (누락 대비)
    p.setdefault("socioeconomic_tier", "middle")
    p.setdefault("family_stage", "single")
    p["region_birth_rate_index"] = float(p.get("region_birth_rate_index", 0.5))
    if "household_size" in p:
        try: p["household_size"] = int(p["household_size"])
        except: p["household_size"] = 2
    else:
        p["household_size"] = 2
    p["working_mom"] = bool(p.get("working_mom", False))
    p["car_ownership"] = bool(p.get("car_ownership", True))
    return p

# ---------------- Main ----------------
def main():
    all_personas = []
    loops = math.ceil(TOTAL_PERSONAS / BATCH_SIZE)
    for i in range(1, loops+1):
        need = min(BATCH_SIZE, TOTAL_PERSONAS - len(all_personas))
        if need <= 0: break
        print(f"[INFO] batch {i}/{loops}: request {need}")
        data = None
        for t in range(1, MAX_RETRIES+1):
            try:
                data = call_llm_for_batch(need, i, t)
                # 개수 틀리면 자르기
                if len(data) > need: data = data[:need]
                break
            except Exception as e:
                print(f"[WARN] batch {i} try {t} 실패: {e}")
                time.sleep(1.5)
        if not data:
            print(f"[ERROR] batch {i} 전원 실패 → 스킵")
            continue

        exist = {p.get("name") for p in all_personas}
        uniq = [normalize_persona(p) for p in data if p.get("name") not in exist]
        all_personas.extend(uniq)
        print(f"[INFO] accumulated: {len(all_personas)}")

    with open("personas.json","w",encoding="utf-8") as f:
        json.dump(all_personas, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved personas.json with {len(all_personas)} personas")

if __name__ == "__main__":
    main()
