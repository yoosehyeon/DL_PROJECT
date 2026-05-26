"""HybridPdM - 설정 (Settings).

Risk Score 가중치 · 융합 방식 · 등급 임계값 · 시스템 상태 · 모델 파일 목록을
3개 탭으로 분리: [위험도 설정] [시스템 상태] [모델 파일 목록].
"""
import streamlit as st

st.set_page_config(
    page_title="설정 — HybridPdM",
    page_icon="⚙️",
    layout="wide",
)

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import ui_common as ui

# Session State 초기화
if "risk_weights" not in st.session_state:
    st.session_state.risk_weights = dict(config.RISK_WEIGHTS)
if "fusion_method" not in st.session_state:
    st.session_state.fusion_method = "max"

st.title("⚙️ 설정")
st.caption("Risk Score 계산 방식과 시스템 환경 정보를 관리합니다.")

ui.render_intent(
    what="**Risk Score(위험도)** 가 어떻게 계산되는지 조절하고, 시스템 환경과 저장된 모델 파일을 확인합니다.",
    when="현장 상황에 맞게 위험 판정 기준을 조정하고 싶을 때, "
         "시스템에 어떤 모델이 학습되어 있는지 점검할 때.",
    next_steps=[
        "[위험도 설정] — 세 모델의 비중과 융합 방식 조정",
        "[시스템 상태] — Python·PyTorch·GPU 환경 점검",
        "[모델 파일 목록] — 학습된 모델 파일과 용량 확인",
    ],
)

tab_risk, tab_status, tab_ckpt = st.tabs([
    "위험도 설정",
    "시스템 상태",
    "모델 파일 목록",
])

# =====================================================================
# Tab 1: 위험도 (Risk Score) 설정
# =====================================================================
with tab_risk:
    # ── 가중치 ─────────────────────────────────────────────────────
    st.subheader("세 모델의 비중 (가중치)")
    st.caption(
        "Risk Score를 계산할 때 어느 모델을 얼마나 비중 있게 볼지 정합니다. "
        "예: 고장 분류 비중을 키우면 CNN 신호에 더 민감해집니다."
    )

    w = st.session_state.risk_weights
    c1, c2, c3 = st.columns(3)
    with c1:
        new_f = st.slider("고장 분류 비중 (CNN)", 0.0, 1.0, w["failure"], 0.05,
                          help="CNN 모델의 고장 확률을 얼마나 반영할지 (0~1).")
    with c2:
        new_a = st.slider("이상 감지 비중 (AE)", 0.0, 1.0, w["anomaly"], 0.05,
                          help="AE 모델의 이상 점수를 얼마나 반영할지 (0~1).")
    with c3:
        new_r = st.slider("수명 예측 비중 (LSTM)", 0.0, 1.0, w["rul"], 0.05,
                          help="LSTM의 잔존 수명을 얼마나 반영할지 (0~1).")

    total = new_f + new_a + new_r
    if total > 0:
        st.info(f"세 비중의 합: **{total:.2f}** — 내부적으로 자동 정규화되어 적용됩니다.")
    else:
        st.warning("비중의 합이 0입니다. 최소 하나는 0보다 크게 설정해야 합니다.")

    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("비중 적용", type="primary"):
            st.session_state.risk_weights = {
                "failure": new_f, "anomaly": new_a, "rul": new_r,
            }
            st.success("비중이 적용되었습니다.")
    with btn2:
        if st.button("기본값으로 되돌리기"):
            st.session_state.risk_weights = dict(config.RISK_WEIGHTS)
            st.success(f"기본값으로 복원: {config.RISK_WEIGHTS}")
            st.rerun()

    st.divider()

    # ── 융합 방식 ──────────────────────────────────────────────────
    st.subheader("세 모델 점수를 합치는 방식")
    st.caption("세 모델의 결과를 어떤 방식으로 하나의 점수로 합칠지 선택합니다.")

    FUSION_DESC = {
        "max":      "두 방식 중 큰 값 (가장 보수적 — 안전 우선, 권장)",
        "weighted": "가중 평균 (직관적, 한 모델만 발화하면 약하게 반영)",
        "noisy_or": "셋 중 하나라도 위험하면 강하게 반영 (놓치지 않기 우선)",
    }

    fusion = st.radio(
        "융합 방식",
        ["max", "weighted", "noisy_or"],
        index=["max", "weighted", "noisy_or"].index(st.session_state.fusion_method),
        format_func=lambda k: f"**{k}** — {FUSION_DESC[k]}",
        horizontal=True,
    )
    st.session_state.fusion_method = fusion

    st.divider()

    # ── 등급 임계값 ───────────────────────────────────────────────
    st.subheader("위험 등급 기준 표")
    st.caption("Risk Score가 어느 값 이상이면 어떤 등급인지 보여줍니다 (수정은 config.py에서).")

    levels_df = pd.DataFrame(config.RISK_LEVELS, columns=["등급", "임계값"])
    levels_df["의미"] = [
        "≥ 0.80 — 즉시 정비 필요",
        "≥ 0.50 — 주의 관찰 필요",
        "≥ 0.30 — 경미한 이상 징후",
        "≥ 0.00 — 정상 운전",
    ]
    st.dataframe(levels_df, use_container_width=True, hide_index=True)

# =====================================================================
# Tab 2: 시스템 상태
# =====================================================================
with tab_status:
    st.subheader("시스템 환경 점검")
    st.caption("HybridPdM 실행에 필요한 주요 구성 요소가 정상 설치/연결되어 있는지 확인합니다.")

    status_items: list[tuple[str, str, str]] = []

    status_items.append(("Python", sys.version.split()[0], "✅"))
    try:
        import torch
        status_items.append(("PyTorch (딥러닝 엔진)", torch.__version__, "✅"))
        status_items.append((
            "계산 장치 (Device)",
            config.get_device(),
            "✅ GPU 사용 가능" if config.get_device() == "cuda" else "CPU 사용",
        ))
    except ImportError:
        status_items.append(("PyTorch (딥러닝 엔진)", "미설치", "❌"))

    try:
        import streamlit as _st
        status_items.append(("Streamlit (대시보드)", _st.__version__, "✅"))
    except Exception:
        pass

    try:
        import pyngrok
        status_items.append(("pyngrok (외부 공개용)", pyngrok.__version__, "✅"))
    except ImportError:
        status_items.append(("pyngrok (외부 공개용)", "미설치", "❌"))

    try:
        import captum
        status_items.append(("Captum (영향 센서 분석)", captum.__version__, "✅"))
    except ImportError:
        status_items.append(("Captum (영향 센서 분석)", "미설치", "⚠️ XAI 사용 불가"))

    ngrok_path = shutil.which("ngrok")
    status_items.append((
        "ngrok CLI",
        ngrok_path if ngrok_path else "미발견",
        "✅" if ngrok_path else "⚠️ (필요할 때만 설치)",
    ))

    status_df = pd.DataFrame(status_items, columns=["항목", "버전/경로", "상태"])
    st.dataframe(status_df, use_container_width=True, hide_index=True)

# =====================================================================
# Tab 3: 학습된 모델 파일 목록
# =====================================================================
with tab_ckpt:
    st.subheader("학습된 모델 파일")
    st.caption(
        "지금까지 학습되어 `artifacts/checkpoints/` 폴더에 저장된 모델 파일 목록입니다. "
        "최근 학습된 것부터 표시됩니다."
    )

    ckpts = sorted(
        config.CHECKPOINT_DIR.glob("*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if ckpts:
        ckpt_rows = []
        for p in ckpts[:30]:
            size = p.stat().st_size
            size_str = (
                f"{size / 1024:.1f} KB" if size < 1024 * 1024
                else f"{size / (1024 * 1024):.1f} MB"
            )
            ckpt_rows.append({"파일명": p.name, "크기": size_str, "형식": p.suffix})
        st.dataframe(pd.DataFrame(ckpt_rows), use_container_width=True, hide_index=True)
        st.caption(f"총 {len(ckpts)}개 파일 (최근 30개까지 표시).")
    else:
        st.info("아직 학습된 모델 파일이 없습니다. 터미널에서 `python main.py` 를 실행하세요.")

# =====================================================================
# Sidebar 용어집
# =====================================================================
ui.render_glossary_sidebar([
    "Risk Score", "Critical", "Warning", "Advisory", "Normal",
    "Weighted Sum", "Noisy-OR", "Fusion",
    "CNN", "AE", "LSTM", "XAI",
])
