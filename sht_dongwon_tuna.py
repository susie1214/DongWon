import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, List
from tqdm import tqdm

# --- 0. 4-bit 양자화를 위한 BitsAndBytes 설정 ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def load_model_and_tokenizer():
    """LLM 모델과 토크나이저를 로드하는 함수"""
    if not torch.cuda.is_available():
        raise SystemError("GPU를 사용할 수 없습니다. 4-bit 양자화는 GPU 환경이 필수적입니다.")
    print(f"✅ GPU({torch.cuda.get_device_name(0)})를 사용합니다.")
    
    model_name = "K-intelligence/Midm-2.0-Mini-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("\n🔄 4-bit 양자화된 모델을 로드합니다... (시간이 걸릴 수 있습니다)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    model.eval()
    print("✅ 모델 로드 완료!")
    return model, tokenizer

@dataclass
class ProductConfig:
    product_name: str
    features: Optional[str]
    category: Optional[str]
    price: Optional[float]
    base_sales_volume: int

# 카테고리별 컨텍스트 및 계절성 함수
def get_category_context(category: Optional[str]) -> str:
    if not category: return ""
    c = category.lower()
    ctx = []
    ctx.append("- 공통: 설(1-2월), 추석(9-10월), 연말(12월)은 선물세트 및 가정용 수요가 증가할 수 있습니다.")
    ctx.append("- 학사 일정: 3월(개학), 7-8월(여름방학), 1-2월(겨울방학) 등은 가정 내 식료품 소비 패턴에 영향을 줍니다.")
    if any(k in c for k in ["yogurt", "요거트", "건강"]):
        ctx.append("- 건강 트렌드: 새해 결심(1월), 여름 대비 다이어트(5-6월) 시즌에 수요가 증가하는 경향이 있습니다.")
    if any(k in c for k in ["beverage", "drink", "음료"]):
        ctx.append("- 날씨 민감도: 무더위가 기승을 부리는 7-8월에 수요가 급증하고, 추운 1, 12월에는 감소할 수 있습니다.")
    return "\n".join(ctx)

def get_category_seasonality(category: Optional[str]) -> Dict[int, float]:
    if not category: return {m: 1.0 for m in range(1, 13)}
    c = category.lower()
    default = {1:1.05, 2:1.08, 3:0.98, 4:0.97, 5:1.02, 6:0.98, 7:1.00, 8:1.02, 9:1.07, 10:1.03, 11:1.02, 12:1.10}
    if any(k in c for k in ["beverage", "drink", "음료"]):
        default.update({6: 1.15, 7: 1.30, 8: 1.25, 12: 0.85, 1: 0.85})
    return default

month_names = {1: "1월(겨울)", 2: "2월(겨울, 설날)", 3: "3월(봄)", 4: "4월(봄)", 5: "5월(봄, 가정의 달)", 6: "6월(여름)", 7: "7월(여름, 휴가철)", 8: "8월(여름, 휴가철)", 9: "9월(가을, 추석)", 10: "10월(가을)", 11: "11월(겨울)", 12: "12월(겨울, 연말)"}

# '구매 의향 점수' 예측 함수
def build_score_prediction_prompt(persona: Dict, product: ProductConfig, month: int) -> str:
    category_context = get_category_context(product.category)
    price_info = f"- 가격: {product.price:,.0f}원" if product.price is not None else "- 가격: 정보 없음"
    return f"""### 질문:
당신은 날카로운 통찰력을 지닌 마케팅 분석가입니다. 아래 페르소나가 주어진 제품을 특정 시점에 얼마나 구매하고 싶어할지 '구매 의향 점수'로 평가해주세요.
**1. 분석 대상 페르소나**
{json.dumps(persona, ensure_ascii=False, indent=2)}
**2. 구매 고려 제품**
- 제품명: {product.product_name}
- 제품 특징: {product.features or "정보 없음"}
- 제품 유형: {product.category or "정보 없음"}
{price_info}
**3. 시장 및 시점 정보**
- 구매 시점: **{month_names[month]}**
- 시장 맥락: {category_context}
**4. 예측 지시**
- 위 모든 정보를 종합하여, 페르소나의 **구매 의향을 0점(전혀 관심 없음)부터 100점(반드시 구매)까지의 점수**로 평가해주세요.
- 다른 설명 없이 오직 **숫자(정수)**만 답변해주세요. 예: `85`
### 답변:
"""
def predict_purchase_score(model, tokenizer, prompt: str) -> int:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, eos_token_id=tokenizer.eos_token_id, temperature=0.1)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split("### 답변:")[-1]
    try:
        numbers = re.findall(r'\d+', response_text)
        score = int(numbers[0]) if numbers else 50
        return max(0, min(100, score))
    except (ValueError, IndexError):
        return 50

def main():
    """메인 실행 함수"""
    # --- 1. 모델 및 데이터 로드 ---
    model, tokenizer = load_model_and_tokenizer()
    
    data_path = './'
    product_df = pd.read_csv(os.path.join(data_path, 'product_info.csv'))
    submission_df = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))

    # --- 2. 저장된 페르소나 로드 ---
    b2c_personas_path = 'b2c_personas.json'
    b2b_personas_path = 'b2b_personas.json'

    if not os.path.exists(b2c_personas_path) or not os.path.exists(b2b_personas_path):
        print("\n❌ 페르소나 파일(.json)을 찾을 수 없습니다.")
        print("먼저 `generate_personas.py`를 실행하여 페르소나 파일을 생성해주세요.")
        return

    print(f"\n✅ '{b2c_personas_path}'와 '{b2b_personas_path}' 파일에서 페르소나를 읽어옵니다.")
    with open(b2c_personas_path, 'r', encoding='utf-8') as f:
        fixed_b2c_personas = json.load(f)
    with open(b2b_personas_path, 'r', encoding='utf-8') as f:
        fixed_b2b_personas = json.load(f)

    # --- 3. 수요량 예측 ---
    print("\n--- 🚀 수요량 예측을 시작합니다 ---")
    all_predictions = []
    all_personas = fixed_b2c_personas + fixed_b2b_personas
    np.random.seed(42)

    for _, product_row in tqdm(product_df.iterrows(), total=len(product_df), desc="📦 전체 제품 예측 진행"):
        product_info = ProductConfig(
            product_name=product_row['product_name'],
            features=product_row.get('product_feature'),
            category=product_row.get('category_level_1'),
            price=float(product_row['price']) if 'price' in product_row and pd.notna(product_row['price']) else None,
            base_sales_volume=int(product_row['base_units']) if 'base_units' in product_row and pd.notna(product_row['base_units']) else 5000
        )
        monthly_total_volumes = []
        seasonality = get_category_seasonality(product_info.category)

        for m in range(1, 13):
            prompts = [build_score_prediction_prompt(p, product_info, m) for p in all_personas]
            all_scores = [predict_purchase_score(model, tokenizer, p) for p in prompts]
            
            average_score = np.mean(all_scores) if all_scores else 50
            base_quantity = product_info.base_sales_volume * (average_score / 50.0)
            seasonal_multiplier = seasonality.get(m, 1.0)
            noise = np.random.normal(1.0, 0.05)
            final_quantity = base_quantity * seasonal_multiplier * noise
            monthly_total_volumes.append(int(round(max(0, final_quantity))))

        all_predictions.append([product_info.product_name] + monthly_total_volumes)

    # --- 4. 제출 파일 생성 ---
    prediction_output = pd.DataFrame(all_predictions, columns=submission_df.columns)
    submission_path = 'final_submission.csv'
    prediction_output.to_csv(submission_path, index=False)
    print(f"\n\n🎉 예측 완료! 최종 결과가 '{submission_path}' 파일로 저장되었습니다.")
    print(prediction_output.head())

if __name__ == "__main__":
    main()
