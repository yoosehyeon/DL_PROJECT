"""HybridPdM - Streamlit 메인 대시보드.

장비 위험도 한눈 요약 · Risk Score 시뮬레이터 · 최신 파이프라인 결과를 탭으로 분리해
"지금 위험한가요?" 라는 질문에 한 화면에서 답한다.

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
import ui_common as ui

# ── Session State 초기화 ─────────────────────────────────────────
if "risk_weights" not in st.session_state:
    st.session_state.risk_weights = dict(config.RISK_WEIGHTS)
if "fusion_method" not in st.session_state:
    st.session_state.fusion_method = "max"

# =====================================================================
# Header + 인텐트 박스
# =====================================================================
st.title("\U0001f3ed HybridPdM Dashboard")
st.caption("Hybrid Deep Learning 기반 예지보전(PdM) 시스템")

ui.render_intent(
    what="세 가지 진단 모델(고장 분류·이상 탐지·잔존 수명)의 결과를 합쳐 만든 **장비 위험도(Risk Score)** 를 한 화면에 보여줍니다.",
    when="장비 상태를 빠르게 점검할 때, 위험도가 어떻게 계산되는지 직접 확인하고 싶을 때, 가장 최근 분석 결과를 보고 싶을 때.",
    next_steps=[
        "[요약] — 어떤 장비/모델이 분석 준비되었는지 확인",
        "[Risk Score 시뮬레이터] — 세 모델의 값을 직접 바꿔보며 위험도 변화 확인",
        "[상세 리포트] — 가장 최근 분석 결과 점수 확인",
    ],
)

# =====================================================================
# 모델 카탈로그 정의 (탭 간 공유)
# =====================================================================
MODEL_LIST: list[tuple[str, str, str]] = [
    ("CNN (AI4I)", "ai4i_cnn", ".pt"),
    ("GBDT (AI4I)", "ai4i_gbdt", ".pkl"),
    ("CNN (CWRU)", "cwru_cnn", ".pt"),
    ("AE (Hydraulic)", "hydraulic_ae", ".pt"),
    ("LSTM (C-MAPSS)", "cmapss_lstm", ".pt"),
    ("LSTM (N-CMAPSS)", "ncmapss_lstm", ".pt"),
]
ready_flags = {key: bool(list(config.CHECKPOINT_DIR.glob(f"{key}*{ext}")))
               for _, key, ext in MODEL_LIST}
ready_count = sum(ready_flags.values())

# =====================================================================
# Tabs — 정보 위계를 세 단계로 정리
# =====================================================================
tab_summary, tab_sim, tab_report = st.tabs([
    "요약",
    "Risk Score 시뮬레이터",
    "상세 리포트",
])

# ---------------------------------------------------------------------
# Tab 1: 요약 (모델 체크포인트 현황)
# ---------------------------------------------------------------------
with tab_summary:
    st.subheader("분석 모델 준비 현황")
    st.caption("각 장비/데이터에 대해 학습된 진단 모델이 준비되어 있는지 확인합니다.")

    cols = st.columns(len(MODEL_LIST))
    for col, (label, key, ext) in zip(cols, MODEL_LIST):
        if ready_flags[key]:
            col.metric(label, "준비됨", delta="✅", delta_color="normal",
                       help="이 모델은 학습이 완료되어 바로 분석에 사용할 수 있습니다.")
        else:
            col.metric(label, "미준비", delta="❌", delta_color="normal",
                       help=f"아직 학습되지 않았습니다. 터미널에서 `python main.py --datasets {key}` 를 실행하세요.")

    if ready_count == len(MODEL_LIST):
        st.success(
            f"전체 {len(MODEL_LIST)}개 모델이 모두 준비되었습니다. 다른 탭에서 분석을 시작하세요."
        )
    elif ready_count == 0:
        st.warning(
            "분석 가능한 모델이 없습니다. 터미널에서 `python main.py` 를 먼저 실행해 모델을 학습하세요."
        )
    else:
        st.info(
            f"총 {len(MODEL_LIST)}개 중 **{ready_count}개** 모델이 준비되었습니다. "
            f"준비되지 않은 모델은 `python main.py --datasets <이름>` 으로 따로 학습할 수 있습니다."
        )

# ---------------------------------------------------------------------
# Tab 2: Risk Score 시뮬레이터
# ---------------------------------------------------------------------
with tab_sim:
    st.subheader("Risk Score 시뮬레이터")
    st.caption(
        "세 모델의 값(고장 확률·이상 점수·남은 수명)을 직접 조절해보면, "
        "최종 위험도 점수가 어떻게 변하는지 실시간으로 확인할 수 있습니다."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**고장 확률 (CNN 모델)**")
        fp = st.slider(
            "고장 확률", 0.0, 1.0, 0.30, 0.01, key="sim_fp",
            help="0 = 완전 정상, 1 = 확실히 고장. CNN 모델이 판단한 고장 가능성.",
        )
    with c2:
        st.markdown("**이상 점수 (AE 모델)**")
        anom = st.slider(
            "이상 점수", 0.0, 1.0, 0.20, 0.01, key="sim_an",
            help="0 = 평소와 똑같음, 1 = 평소 패턴과 크게 다름. 이상 신호가 감지된 정도.",
        )
    with c3:
        st.markdown("**남은 수명 (LSTM 모델)**")
        rul = st.slider(
            "남은 수명", 0.0, 1.0, 0.70, 0.01,
            key="sim_rl",
            help="1.0 = 수명 충분히 남음, 0.0 = 수명 거의 소진. 0에 가까울수록 정비 시급.",
        )

    w = st.session_state.risk_weights
    fm = st.session_state.fusion_method

    ws_val = float(rs.weighted_sum(fp, anom, rul, w))
    nor_val = float(rs.noisy_or(fp, anom, rul, w))
    risk_val = float(rs.compute_risk(fp, anom, rul, fm))
    level = rs.to_risk_level(risk_val)

    LEVEL_ICON = {"Critical": "\U0001f534", "Warning": "\U0001f7e1",
                  "Advisory": "\U0001f7e0", "Normal": "\U0001f7e2"}

    st.divider()
    r1, r2, r3, r4 = st.columns([1, 1, 1, 1.5])
    r1.metric("가중 합 (Weighted Sum)", f"{ws_val:.3f}",
              help="세 모델 점수를 가중 평균으로 합친 값. 가장 직관적이나 한 모델만 위험 신호를 내면 희석됨.")
    r2.metric("논리 OR (Noisy-OR)", f"{nor_val:.3f}",
              help="셋 중 하나라도 위험하면 점수가 1에 가까워지는 방식. 고장을 놓치지 않는 데 강함.")
    r3.metric(f"최종 위험도 ({fm})", f"{risk_val:.3f}",
              help="현재 선택된 융합 방식으로 계산된 최종 Risk Score. 0~1.")
    with r4:
        icon = LEVEL_ICON.get(level, "⚪")
        st.markdown(f"### {icon} {level}")
        st.progress(min(risk_val, 1.0))
        st.caption(ui.glossary_help(level) if level in ui.GLOSSARY else "")

    with st.expander("산식 상세 — Risk Score 계산식"):
        st.markdown(f"""
**가중치:** failure={w['failure']:.2f} · anomaly={w['anomaly']:.2f} · rul={w['rul']:.2f}
 | **Fusion:** `{fm}`

| 방식 | 산식 |
|------|------|
| Weighted Sum | R = w\_f · P(fail) + w\_a · Anomaly + w\_r · (1 − RUL) |
| Noisy-OR | R = 1 − (1−f)^w\_f · (1−a)^w\_a · (1−rul\_risk)^w\_r |
| Max | max(Weighted Sum, Noisy-OR) |

> 가중치와 Fusion 방식은 **Settings** 페이지에서 변경할 수 있습니다.
""")

# ---------------------------------------------------------------------
# Tab 3: 파이프라인 실행 결과
# ---------------------------------------------------------------------
with tab_report:
    st.subheader("최근 분석 결과")
    st.caption("터미널에서 `python main.py` 를 실행할 때마다 자동으로 저장되는 분석 결과를 데이터셋별로 보여줍니다.")

    reports = sorted(
        config.REPORT_DIR.glob("pipeline_report*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if reports:
        sel_report = st.selectbox(
            "분석 결과 선택 (최신순)", reports, format_func=lambda p: p.name,
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
                    st.success(f"**{name}** — 분석 성공 — {summary}")
                elif status == "skipped":
                    st.warning(f"**{name}** — 분석 건너뜀: {entry.get('reason', '')}")
                else:
                    st.error(f"**{name}** — 분석 실패 ({status}): {entry.get('error', '')}")
        except Exception as e:
            st.error(f"리포트 로드 실패: {e}")
    else:
        st.info(
            "리포트가 존재하지 않습니다. `python main.py` 실행 후 "
            "`artifacts/reports/pipeline_report_*.json` 이 생성되면 자동 조회됩니다."
        )

# =====================================================================
# Sidebar - 시스템 정보 + 용어집
# =====================================================================
with st.sidebar:
    st.header("시스템 정보")
    st.markdown(f"- **Device:** `{config.get_device()}`")
    st.markdown(f"- **체크포인트:** {ready_count}/{len(MODEL_LIST)}")
    st.markdown(f"- **Fusion:** `{st.session_state.fusion_method}`")
    try:
        import pyngrok
        st.markdown(f"- **pyngrok:** `{pyngrok.__version__}` ✅")
    except ImportError:
        st.markdown("- **pyngrok:** ❌")
    st.divider()
    st.caption("HybridPdM v2.1 — Streamlit POC")

# 대시보드에서 자주 쓰이는 용어만 좁혀서 표시 (정보 과부하 방지)
ui.render_glossary_sidebar([
    "Risk Score", "Critical", "Warning", "Advisory", "Normal",
    "Weighted Sum", "Noisy-OR", "Fusion",
    "CNN", "AE", "LSTM", "RUL", "Anomaly Score", "PdM",
])
