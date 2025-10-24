🧠 2025 동원 × KAIST AI Competition
---
Unlocking Future Sales & Demographics

소비자 페르소나 기반 신제품 월별 수요 예측 프로젝트

---
📘 대회 개요
구분	내용
대회명	2025 동원 × KAIST AI Competition: Unlocking Future Sales & Demographics
주최/주관	동원그룹 · KAIST 김재철 AI대학원
후원	AWS, Microsoft, PwC
운영 플랫폼	DACON 공식 페이지
기간	2024.12 ~ 2025.03 (예선 및 본선 포함)

---
🎯 대회 배경

동원그룹은 내부 챗봇 ‘동원GPT’ 도입을 통해 AI 문해력(AI Literacy) 강화를 추진해 왔으며,
KAIST 김재철 AI대학원과의 협력으로 산학연계형 AI 인재 발굴 및 기술 내재화를 목표로 한다.

이번 대회는
“소비자 페르소나(persona)”를 생성하고 이를 기반으로 신제품의 월별 판매량을 예측하는 문제를 중심으로 한다.

---
🧩 문제 정의
▪ 주제

소비자 페르소나 기반 동원 신제품 월별 수요 예측 (2024.07 ~ 2025.06)

참가자는 LLM을 활용해 가상의 소비자 페르소나를 생성하고,
각 페르소나의 속성과 행동 패턴을 기반으로 신제품의 월별 판매량을 예측한다.

---
🧠 페르소나 설계 가이드

LLM 활용: ChatGPT, Claude, Gemini, LLaMA 등 자유 선택

속성 구성: 최소 10개 이상 (연령, 성별, 지역, 소득, 직업, 구매력, 라이프스타일, 관심사 등)

가중치 설정: 각 속성별 영향도(%)를 정의

행동 패턴: 월별 구매 확률 및 빈도

프롬프트 형태: Single-turn 구조 (한 번의 입력으로 일관된 페르소나 생성)

---
📈 예측 모델 설계
🔹 데이터 활용

시장 트렌드, 계절성, 프로모션, 광고 이벤트 효과 반영

유사 제품군(참치캔, 간편식, HMR 등)의 과거 판매데이터 수집

할인율, 유통 경로, 프로모션 요소를 변수화

🔹 예측 방식

학습 기반 모델: LightGBM, XGBoost, CatBoost, Prophet, LSTM, Transformer

시뮬레이션 기반 모델: LLM+RAG 기반 페르소나 응답 시뮬레이션

하이브리드: LLM 추정 구매확률을 구조적 모델(Input Feature)로 결합

🔬 참고 연구 — Using LLMs for Market Research (Harvard Business School, 2025)

Brand, J., Israeli, A., & Ngwe, D. (2025). Using LLMs for Market Research. Working Paper 23-062, Harvard Business School.

---
🔍 핵심 개념

LLM(GPT)을 가상 소비자(simulated consumer) 로 활용해
전통적 설문 없이 제품 속성별 지불의사금액(WTP) 추정

응답을 단일 값이 아닌 확률분포(Distributional Response) 로 간주

Fine-tuning 을 통해 인간 응답과의 일치도 향상

---
📊 주요 결과

GPT 기반 컨조인트 분석 결과가 실제 소비자 조사와 근사

새로운 제품 속성(예: 노트북 내장 프로젝터)에 대해서도 예측 가능

GPT-3.5 / GPT-4 모델은 사람과 유사한 패턴을 보임

세그먼트별(성별, 소득, 지역) 이질성 반영은 아직 제한적

---
💡 시사점

LLM 기반 시장조사는 인간을 대체하기보다
**“아이디어 탐색–가설 검증–시장 반응 예측”**의 보조 수단 으로 활용 가능

---
🧮 구현 예시 (LLM 시뮬레이션 코드 요약)
from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY")

prompt = """
You are a 35-year-old female office worker with mid-level income,
shopping for ready-to-eat tuna meals.
Considering price, nutrition, and taste,
how likely (0-100%) are you to purchase this product monthly?
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    temperature=1.0
)

print(response.choices[0].message.content)

---
📂 Repository Structure
DongWon/
 ├── data/               # 예측용 입력 데이터셋 (판매, 트렌드, 이벤트 등)
 ├── persona/            # LLM 기반 페르소나 생성 및 속성 가중치 코드
 ├── models/             # 수요 예측 및 시뮬레이션 모델
 ├── notebooks/          # EDA 및 실험용 주피터 노트북
 ├── outputs/            # 결과 CSV / 시각화 이미지
 └── README.md

---
🧩 기대 효과

AI 기반 소비자 모델링 : 페르소나를 통한 정성적·정량적 수요 예측

제품 출시 리스크 최소화 : 가상 시장 실험을 통한 사전 검증

AI Literacy 확산 : 기업 내 데이터·AI 활용 문화 확립

---
🏁 Reference & Citation

Brand, J., Israeli, A., & Ngwe, D. (2025). Using LLMs for Market Research. Harvard Business School Working Paper 23-062.
DACON: 2025 동원 × KAIST AI Competition
