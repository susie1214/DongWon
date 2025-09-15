# -*- coding: utf-8 -*-
"""
agent_addons_llm_rag.py

- LLM single-turn prompt builder + OpenAI call stub
- Lightweight RAG (TF-IDF cosine) over local corpus to derive monthly external signals
  (online_share, overseas_market_index, pet_household_share)

USAGE:
  from agent_addons_llm_rag import build_single_turn_prompt_for_products, call_openai_single_turn
  from agent_addons_llm_rag import RagSignals

Requirements (optional):
  pip install scikit-learn openai pypdf
"""
import os, json, math, re
from pathlib import Path
from typing import List, Dict, Any

# --------- LLM (single-turn) ---------
def build_single_turn_prompt_for_products(products: List[Dict[str,Any]], months: List[str]) -> str:
    payload = {
        "objective": "신제품 출시 후 12개월(2024-07~2025-06) 월별 수요 예측을 위한 페르소나 생성",
        "requirements": [
            "각 제품별로 6~10개 페르소나",
            "각 페르소나는 최소 10개 이상의 속성(연령, 성별, 소득, 지역, 채널 등)과 속성별 가중치 포함",
            "각 페르소나는 구매확률(%)과 월별 구매빈도 패턴(12개 길이, 1.0=중립 스케일) 포함",
            "JSON 포맷으로만 응답(주석/설명 없음)"
        ],
        "products": products,
        "months": months
    }
    return json.dumps(payload, ensure_ascii=False)

def call_openai_single_turn(prompt: str) -> Dict:
    """
    Minimal example; user must set OPENAI_API_KEY.
    Returns JSON dict parsed from model output.
    """
    try:
        import openai
    except Exception as e:
        raise RuntimeError("Please 'pip install openai' first.") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable.")
    openai.api_key = api_key

    # For modern clients, switch to client = OpenAI() etc. depending on SDK version
    # Here we keep a generic example that works on older openai library versions.
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # choose any chat-capable model available to you
        messages=[{"role":"system","content":"You are a helpful data assistant."},
                  {"role":"user","content": prompt}],
        temperature=0.4,
        max_tokens=4000
    )
    text = resp["choices"][0]["message"]["content"]
    # Expect JSON text
    data = json.loads(text)
    return data


# --------- Lightweight RAG for signals ---------
class RagSignals:
    """
    Build a TF-IDF vector space index over text files in a folder (rag_corpus),
    then query for specific signals and map to monthly external features.

    - online_share               (0~1)
    - overseas_market_index      (0~100)
    - pet_household_share        (0~1)

    Corpus examples to include (TXT/PDF/CSV converted to TXT):
    - market reports for e-commerce growth
    - overseas expansion news/IR
    - pet ownership statistics

    NOTE: This is a light baseline; replace with embeddings/FAISS if desired.
    """
    def __init__(self, corpus_dir: str):
        self.corpus_dir = Path(corpus_dir)
        self.docs = []
        self.doc_names = []
        self._vectorizer = None
        self._X = None

    def _load_text(self, p: Path) -> str:
        if p.suffix.lower()==".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(p))
                return "\n".join([page.extract_text() or "" for page in reader.pages])
            except Exception:
                return ""
        else:
            try:
                return p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                try:
                    return p.read_text(encoding="cp949", errors="ignore")
                except Exception:
                    return ""

    def build(self):
        files = [p for p in self.corpus_dir.glob("**/*") if p.is_file() and p.suffix.lower() in [".txt",".pdf",".md"]]
        for p in files:
            txt = self._load_text(p)
            if txt.strip():
                self.docs.append(txt)
                self.doc_names.append(str(p))
        if not self.docs:
            return False
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
        self._X = self._vectorizer.fit_transform(self.docs)
        return True

    def _search(self, query: str, topk=10):
        if self._X is None:
            return []
        q = self._vectorizer.transform([query])
        import numpy as np
        scores = (self._X @ q.T).toarray().ravel()
        idx = np.argsort(-scores)[:topk]
        return [(self.doc_names[i], float(scores[i])) for i in idx if scores[i] > 0]

    def derive_signals(self) -> dict:
        """
        Very naive keyword-derived signals. Replace with your domain logic or LLM summarization.
        """
        signals = {}
        q_online  = "Korea e-commerce online retail share percentage growth 2024 2025"
        q_overseas= "Dongwon overseas market expansion export sales index global market"
        q_pet     = "Korea pet ownership households percentage 2024 2025 dogs cats"

        hits_online   = self._search(q_online, topk=10)
        hits_overseas = self._search(q_overseas, topk=10)
        hits_pet      = self._search(q_pet, topk=10)

        # Map arbitrary raw scores to normalized ranges
        def norm(score_sum, lo, hi, max_ref):
            return float(max(lo, min(hi, (score_sum / max_ref) * (hi - lo) + lo)))

        s_online   = norm(sum([s for _,s in hits_online]),   0.35, 0.65, 5.0)  # ~35~65%
        s_overseas = norm(sum([s for _,s in hits_overseas]), 10.0, 80.0, 5.0)  # 10~80 index
        s_pet      = norm(sum([s for _,s in hits_pet]),      0.20, 0.40, 5.0)  # ~20~40%

        signals["online_share"] = round(s_online, 3)
        signals["overseas_market_index"] = round(s_overseas, 1)
        signals["pet_household_share"] = round(s_pet, 3)
        return signals

    def export_external_monthly(self, months: list, out_csv: str):
        """
        Writes/updates external_monthly.csv:
        Fills static signals from RAG; user can edit further or make them time-varying.
        """
        import pandas as pd, os
        sig = self.derive_signals()
        df = pd.DataFrame([{
            "month": m,
            "tavg": "",
            "feels_like": "",
            "precip_mm": "",
            "search_index": "",
            "holiday_dummy": "",
            "online_share": sig["online_share"],
            "overseas_market_index": sig["overseas_market_index"],
            "pet_household_share": sig["pet_household_share"]
        } for m in months])
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        return out_csv
