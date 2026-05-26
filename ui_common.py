"""HybridPdM - Streamlit 프론트엔드 공용 헬퍼.

가독성 통일을 위한 두 가지 기능을 제공:
  1) render_intent(): 각 페이지 상단에 "이 화면이 보여주는 것" 1줄 안내 박스
  2) render_glossary_sidebar() / glossary_help(): 비전공자용 용어 도움말

설계 원칙:
  - 모든 페이지가 동일한 시각적 패턴 → 사용자가 "어디서 뭘 보는지" 즉시 인지
  - GLOSSARY 사전은 한 곳에서 관리 → 용어 정의의 단일 출처
"""
from __future__ import annotations

from typing import Iterable, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# 용어집 — Risk Score, 평가 지표, 모델 등 비전공자가 마주칠 약어들
# ---------------------------------------------------------------------------
GLOSSARY: dict[str, str] = {
    # Risk Score 관련
    "Risk Score":  "세 모델 출력을 융합한 단일 위험 점수 [0,1]. 1에 가까울수록 위험.",
    "Critical":    "Risk Score ≥ 0.80. 즉시 정비 필요.",
    "Warning":     "Risk Score 0.50~0.80. 주의 관찰 필요.",
    "Advisory":    "Risk Score 0.30~0.50. 경미한 이상 징후.",
    "Normal":      "Risk Score < 0.30. 정상 운전.",
    "Weighted Sum":"세 모델 점수의 가중 평균. 직관적이나 한 모델만 발화하면 희석됨.",
    "Noisy-OR":    "어느 한 모델이라도 위험 신호를 내면 점수가 1에 가까워짐. FN 최소화용.",
    "Fusion":      "세 모델 점수를 하나의 Risk Score로 합치는 방식 (Weighted/Noisy-OR/Max).",

    # 평가 지표
    "Accuracy":    "전체 중 맞춘 비율. 클래스가 불균형이면 함정 있음 (99% 정상이면 다 정상이라 찍어도 99%).",
    "Precision":   "모델이 '고장'이라 한 것 중 진짜 고장 비율. 거짓 경보가 적을수록 높음.",
    "Recall":      "진짜 고장 중 모델이 잡아낸 비율. PdM에서 가장 중요 — 놓치면 장비 멈춤.",
    "F1":          "Precision과 Recall의 조화평균. 둘 다 균형 있게 잘하나의 지표.",
    "F1 Score":    "Precision과 Recall의 조화평균. 둘 다 균형 있게 잘하나의 지표.",
    "AUC":         "0.5(동전던지기) ~ 1.0(완벽). 모델이 확률을 얼마나 잘 매기는가.",
    "RMSE":        "평균 제곱 오차의 제곱근 (cycle 단위). 큰 오차에 더 큰 페널티. 작을수록 좋음.",
    "MAE":         "평균 절대 오차 (cycle 단위). 평균적으로 몇 cycle 틀렸나. 작을수록 좋음.",
    "R²":          "0~1 (음수 가능). 1에 가까울수록 RUL 변동을 잘 설명. 0.7+ 양호.",

    # PdM/모델 약어
    "RUL":         "Remaining Useful Life. 잔존 수명 (예: 앞으로 몇 cycle 더 사용 가능).",
    "PdM":         "Predictive Maintenance. 예지보전 — 고장 발생 전에 미리 정비.",
    "FN":          "False Negative. 고장인데 정상이라 한 경우. PdM에서 가장 치명적.",
    "FP":          "False Positive. 정상인데 고장이라 한 경우 (거짓 경보).",
    "Anomaly Score":"Autoencoder의 재구성 오차 기반 이상 점수. 평소 분포에서 벗어날수록 큼.",
    "Threshold":   "이 값을 넘으면 '고장/이상'으로 판정하는 임계값. 낮추면 Recall↑ FP↑.",
    "Decision Threshold":"분류 확률을 0/1로 가를 임계값. 도메인 손실을 반영해 설정.",
    "XAI":         "eXplainable AI. 모델이 왜 그런 판단을 했는지 설명 (여기선 Integrated Gradients).",
    "IG":          "Integrated Gradients. 입력 피처가 예측에 기여한 정도를 정량화하는 XAI 기법.",
    "CNN":         "Convolutional Neural Network. 본 시스템에선 고장 분류용.",
    "AE":          "Autoencoder. 정상 데이터의 분포를 학습해 이상을 탐지.",
    "LSTM":        "Long Short-Term Memory. 시계열 학습용 RNN. 본 시스템에선 RUL 회귀.",
    "GBDT":        "Gradient Boosting Decision Tree. tabular 데이터에 강한 머신러닝 모델.",
    "STFT":        "Short-Time Fourier Transform. 시간-주파수 평면으로 신호를 분해.",
}


# ---------------------------------------------------------------------------
# 화면 상단 인텐트 박스
# ---------------------------------------------------------------------------
def render_intent(
    what: str,
    when: Optional[str] = None,
    next_steps: Optional[Iterable[str]] = None,
) -> None:
    """페이지 상단에 '이 화면이 보여주는 것' 안내 박스를 그린다.

    파라미터:
      what       : 이 화면이 하는 일 (1~2문장). 필수.
      when       : 언제 이 화면을 쓰면 좋은지 (선택, 1문장).
      next_steps : 사용자가 다음에 할 수 있는 액션 목록 (선택).
    """
    lines = [f"**이 화면이 보여주는 것** — {what}"]
    if when:
        lines.append(f"**언제 사용** — {when}")
    if next_steps:
        steps = "  •  ".join(next_steps)
        lines.append(f"**할 수 있는 것** — {steps}")
    st.info("  \n".join(lines), icon="ℹ️")


# ---------------------------------------------------------------------------
# 사이드바 용어집
# ---------------------------------------------------------------------------
def render_glossary_sidebar(terms: Optional[Iterable[str]] = None) -> None:
    """사이드바에 용어집 expander를 추가.

    terms를 지정하면 해당 용어만, None이면 전체 GLOSSARY 표시.
    각 페이지에서 자주 쓰이는 용어 위주로 좁히면 가독성↑.
    """
    selected = (
        {k: GLOSSARY[k] for k in terms if k in GLOSSARY}
        if terms is not None
        else GLOSSARY
    )
    with st.sidebar:
        with st.expander("용어집", expanded=False):
            for term, desc in selected.items():
                st.markdown(f"**{term}** — {desc}")


def glossary_help(term: str) -> str:
    """위젯의 help= 파라미터에 넘길 짧은 설명 텍스트를 반환.

    예: st.metric("F1", val, help=glossary_help("F1"))
    """
    return GLOSSARY.get(term, "")
