# 2025 동원 × KAIST AI Competition


##  대회 개요
- **대회명**  
  2025 동원 × KAIST AI Competition: Unlocking Future Sales & Demographics  
- **주최/주관**  
  동원그룹, KAIST (김재철 AI대학원)  
- **후원**  
  AWS, Microsoft, PwC  
- **운영 플랫폼**  
  데이콘(DACON) (https://dacon.io/en/competitions/official/236546/overview/description)

---

##  대회 배경
동원그룹은 조직 내 AI 문해력(AI literacy) 강화를 위해 내부 챗봇 '동원GPT'를 도입하고 사내 AI 경진대회를 추진해 왔습니다.  
KAIST와 공동으로 AI 인재 양성을 위한 김재철 AI대학원을 설립하는 등 국가 AI 역량 강화에도 기여하고 있습니다. 이러한 기반 위에서, KAIST와 협력하여 **2025년 AI Competition**을 개최하며 인재 발굴과 혁신 기술 발시에 중점을 두고 있습니다.  

---

##  대회 주제
**소비자 페르소나 기반 동원 신제품 월별 수요 예측**  
참가자는 언어모델(LLM)을 활용해 가상의 소비자 페르소나를 생성하고, 이를 기반으로 동원 신제품 출시 후 **2024년 7월부터 2025년 6월까지의 월별 판매량**을 예측합니다.  

---

##  문제 상세 가이드
1. **페르소나 설계**
   - LLM을 활용해 가상의 소비자 페르소나 생성.
   - 최소 10가지 속성 (연령, 성별, 소득 구간 등)을 포함하고, 각 속성에 가중치를 부여해야 함.
   - 각 페르소나당 구매 확률(%) 및 월별 구매 빈도 패턴 제시.
   - 프롬프트는 **‘싱글 턴(single-turn)’** 구조여야 하며, 제품별 또는 전체 제품 단위로 적용 가능.

2. **참고 데이터 활용**
   - 시장 트렌드, 계절성, 프로모션 및 광고와 같은 이벤트 효과 반영.
   - 유사 제품군의 시장 데이터 수집 및 활용.
   - 유통 구조 상 제품 가격에 발생할 수 있는 할인·프로모션 요소 고려.

3. **예측 방식**
   - 학습 기반 모델 또는 시뮬레이션·계산 기반 방식 자유 선택 가능.

--- 
Using LLMs for Market Research

James Brand (Microsoft), Ayelet Israeli (Harvard Business School), Donald Ngwe (Microsoft)
Working Paper 23-062, Harvard Business School | October 2025

🔍 Overview

이 연구는 대형언어모델(LLM) — 특히 OpenAI GPT — 을 시장조사 (Market Research) 에 활용할 수 있는 가능성을 탐구한다.
전통적 설문이나 컨조인트 조사 없이 LLM을 가상 소비자로 활용해 소비자 선호 및 지불의사금액 (WTP)을 추정하는 방법을 제시한다.

🧠 Core Idea

GPT 응답을 “단일 정답”이 아닌 확률분포로 간주해 여러 응답을 샘플링 → 응답 분포 분석 으로 소비자 행동 시뮬레이션

실제 사람을 대체하지는 않지만, 빠르고 저비용으로 시장 반응을 예측 및 테스트 가능

기존 인간 조사 데이터를 파인튜닝에 활용하면 응답 정합성이 향상

⚙️ Research Design

모델 – OpenAI gpt-3.5-turbo-0125 기반, Python API 직접 호출

실험 설계 – 컨조인트 분석 (conjoint analysis) 방식으로 상품 속성별 WTP 추정

비교군 – Fong et al. (2023) 실제 소비자 설문 + 연구진이 직접 수집한 5개 제품군 (치약, 노트북, 태블릿 등)

샘플링 – 각 선택 세트별 수십 ~ 수백 회 프롬프트 반복, 온도 1.0 설정

📊 Key Findings

현실적 WTP 추정: GPT가 추정한 가격민감도 및 속성선호가 인간 조사와 근사함.

파인튜닝 효과: 기존 조사로 미세조정하면 새 제품 속성에 대한 예측 정확도 향상.

제한점: 새로운 제품 카테고리 또는 소비자 세그먼트 이질성 (성별·소득 차이 등) 모사는 부정확.

모델 비교: GPT-3.5 및 GPT-4o 는 사람과 유사, Claude · LLaMA 계열은 편차 큼.

비용 및 확장성: 1회 조사 수백 달러 수준이지만 오픈소스 LLM 활용 시 비용 대폭 절감 가능.

💡 Implications

마케팅 전략 시뮬레이션 : 제품 출시 전 가상 시장 테스트 및 가격 민감도 예측 도구로 활용.

인간 조사의 보완재 : 컨셉 검증 및 가설 선별 단계에서 빠른 의사결정 지원.

미래 전망 : 디지털 트윈 · LLM-기반 소비자 시뮬레이션 등 새로운 시장조사 패러다임 기대.

🧾 Citation

Brand, J., Israeli, A., & Ngwe, D. (2025). Using LLMs for Market Research. Working Paper 23-062, Harvard Business School.
---
