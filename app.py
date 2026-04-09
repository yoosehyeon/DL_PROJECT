"""HybridPdM - Streamlit 메인 대시보드.

전체 모델 현황 · Risk Score 시뮬레이터 · 파이프라인 결과 요약을 제공한다.
실행: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="HybridPdM",
    page_icon="\U0001f3ed",
    layout="wide",
    initial_sidebar_state="expanded",
)

import json
import sys
from pathlib import Path

import numpy as np

# ── 프로젝트 루트를 sys.path에 추가 ──────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import risk_score as rs

# ── Session State 초기화 ─────────────────────────────────────────
if "risk_weights" not in st.session_state:
    st.session_state.risk_weights = dict(config.RISK_WEIGHTS)
if "fusion_method" not in st.session_state:
    st.session_state.fusion_method = "max"

# =====================================================================
# Header
# =====================================================================
st.title("\U0001f3ed HybridPdM Dashboard")
st.caption("Hybrid Deep Learning 기반 예지보전(PdM) 시스템")

# =====================================================================
# 1. 모델 현황  (Bird's Eye View)
# =====================================================================
st.header("모델 현황", divider="gray")

MODEL_LIST: list[tuple[str, str, str]] = [
    ("CNN (AI4I)", "ai4i_cnn", ".pt"),
    ("GBDT (AI4I)", "ai4i_gbdt", ".pkl"),
    ("CNN (CWRU)", "cwru_cnn", ".pt"),
    ("AE (Hydraulic)", "hydraulic_ae", ".pt"),
    ("LSTM (C-MAPSS)", "cmapss_lstm", ".pt"),
    ("LSTM (N-CMAPSS)", "ncmapss_lstm", ".pt"),
]

cols = st.columns(len(MODEL_LIST))
ready_count = 0
for col, (label, key, ext) in zip(cols, MODEL_LIST):
    found = list(config.CHECKPOINT_DIR.glob(f"{key}*{ext}"))
    if found:
        col.metric(label, "Ready", delta="\u2705", delta_color="normal")
        ready_count += 1
    else:
        col.metric(label, "N/A", delta="\u274c", delta_color="normal")

st.info(
    f"총 {len(MODEL_LIST)}개 모델 중 **{ready_count}개** 체크포인트 확인됨  \u2014  "
    "체크포인트가 없으면 `python main.py`로 학습을 먼저 실행하세요."
)

# =====================================================================
# 2. Risk Score 시뮬레이터
# =====================================================================
st.header("Risk Score 시뮬레이터", divider="gray")
st.caption(
    "3개 모델의 출력값을 직접 조절하여 Weighted Sum / Noisy-OR / 최종 Risk Score를 "
    "실시간으로 확인합니다."
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**\U0001f534 CNN \u2014 고장 확률**")
    fp = st.slider("P(failure)", 0.0, 1.0, 0.30, 0.01, key="sim_fp")
with c2:
    st.markdown("**\U0001f7e1 AE \u2014 이상 점수**")
    anom = st.slider("Anomaly Score", 0.0, 1.0, 0.20, 0.01, key="sim_an")
with c3:
    st.markdown("**\U0001f535 LSTM \u2014 잔존수명(정규화)**")
    rul = st.slider(
        "RUL (normalized)", 0.0, 1.0, 0.70, 0.01,
        key="sim_rl",
        help="1.0 = 수명 충분, 0.0 = 수명 거의 소진",
    )

w = st.session_state.risk_weights
fm = st.session_state.fusion_method

ws_val = float(rs.weighted_sum(fp, anom, rul, w))
nor_val = float(rs.noisy_or(fp, anom, rul, w))
risk_val = float(rs.compute_risk(fp, anom, rul, fm))
level = rs.to_risk_level(risk_val)

LEVEL_ICON = {"Critical": "\U0001f534", "Warning": "\U0001f7e1",
              "Advisory": "\U0001f7e0", "Normal": "\U0001f7e2"}

r1, r2, r3, r4 = st.columns([1, 1, 1, 1.5])
r1.metric("Weighted Sum", f"{ws_val:.3f}")
r2.metric("Noisy-OR", f"{nor_val:.3f}")
r3.metric(f"최종 ({fm})", f"{risk_val:.3f}")
with r4:
    icon = LEVEL_ICON.get(level, "\u26aa")
    st.markdown(f"### {icon} {level}")
    st.progress(min(risk_val, 1.0))

with st.expander("\U0001f4d0 산식 상세"):
    st.markdown(f"""
**가중치:** failure={w['failure']:.2f} · anomaly={w['anomaly']:.2f} · rul={w['rul']:.2f}
\u2003|\u2003**Fusion:** `{fm}`

| 방식 | 산식 |
|------|------|
| Weighted Sum | R = w\_f \u00b7 P(fail) + w\_a \u00b7 Anomaly + w\_r \u00b7 (1 \u2212 RUL) |
| Noisy-OR | R = 1 \u2212 (1\u2212f)^w\_f \u00b7 (1\u2212a)^w\_a \u00b7 (1\u2212rul\_risk)^w\_r |
| Max | max(Weighted Sum, Noisy-OR) |

> \U0001f4a1 가중치와 Fusion 방식은 **Settings** 페이지에서 변경할 수 있습니다.
""")

# =====================================================================
# 3. 파이프라인 실행 결과
# =====================================================================
st.header("파이프라인 실행 결과", divider="gray")

reports = sorted(
    config.REPORT_DIR.glob("pipeline_report*.json"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if reports:
    sel_report = st.selectbox(
        "리포트 선택", reports, format_func=lambda p: p.name,
    )
    try:
        with open(sel_report, "r", encoding="utf-8") as f:
            rdata = json.load(f)
        for entry in rdata:
            name = entry.get("name", "?")
            status = entry.get("status", "?")
            if status == "ok":
                ev = entry.get("eval", {})
                summary = {
                    k: f"{v:.4f}"
                    for k, v in ev.items()
                    if isinstance(v, (int, float))
                    and k in (
                        "accuracy", "f1", "test_f1", "rmse", "mae",
                        "r2", "precision", "recall",
                    )
                }
                st.success(f"**{name}** \u2014 {summary}")
            elif status == "skipped":
                st.warning(f"**{name}** \u2014 건너뜀: {entry.get('reason', '')}")
            else:
                st.error(f"**{name}** \u2014 {status}: {entry.get('error', '')}")
    except Exception as e:
        st.error(f"리포트 로드 실패: {e}")
else:
    st.info("실행 결과 없음. `python main.py`를 먼저 실행하세요.")

# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.header("시스템 정보")
    st.markdown(f"- **Device:** `{config.get_device()}`")
    st.markdown(f"- **체크포인트:** {ready_count}/{len(MODEL_LIST)}")
    st.markdown(f"- **Fusion:** `{fm}`")
    try:
        import pyngrok
        st.markdown(f"- **pyngrok:** `{pyngrok.__version__}` \u2705")
    except ImportError:
        st.markdown("- **pyngrok:** \u274c")
    st.divider()
    st.caption("HybridPdM v2.1 \u2014 Streamlit POC")
