"""HybridPdM - Settings.

Risk Score 가중치 · Fusion 방식 · 시스템 상태 · ngrok 배포 가이드를 제공한다.
"""
import streamlit as st

st.set_page_config(
    page_title="Settings \u2014 HybridPdM",
    page_icon="\u2699\ufe0f",
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

# Session State 초기화
if "risk_weights" not in st.session_state:
    st.session_state.risk_weights = dict(config.RISK_WEIGHTS)
if "fusion_method" not in st.session_state:
    st.session_state.fusion_method = "max"

st.title("\u2699\ufe0f Settings")
st.caption("Risk Score 파라미터 및 시스템 설정")

# =====================================================================
# 1. Risk Score 가중치
# =====================================================================
st.header("Risk Score 가중치", divider="gray")
st.caption("Weighted Sum / Noisy-OR에 사용되는 모델별 가중치입니다.")

w = st.session_state.risk_weights
c1, c2, c3 = st.columns(3)
with c1:
    new_f = st.slider("w_failure (CNN)", 0.0, 1.0, w["failure"], 0.05)
with c2:
    new_a = st.slider("w_anomaly (AE)", 0.0, 1.0, w["anomaly"], 0.05)
with c3:
    new_r = st.slider("w_rul (LSTM)", 0.0, 1.0, w["rul"], 0.05)

total = new_f + new_a + new_r
if total > 0:
    st.info(f"가중치 합: **{total:.2f}** (내부적으로 정규화되어 적용됩니다)")
else:
    st.warning("가중치 합이 0입니다. 최소 하나의 가중치를 0보다 크게 설정하세요.")

btn1, btn2 = st.columns(2)
with btn1:
    if st.button("가중치 적용", type="primary"):
        st.session_state.risk_weights = {
            "failure": new_f, "anomaly": new_a, "rul": new_r,
        }
        st.success("\u2705 가중치가 업데이트되었습니다!")
with btn2:
    if st.button("기본값으로 초기화"):
        st.session_state.risk_weights = dict(config.RISK_WEIGHTS)
        st.success(f"기본값으로 복원: {config.RISK_WEIGHTS}")
        st.rerun()

# =====================================================================
# 2. Fusion 방식
# =====================================================================
st.header("Fusion 방식", divider="gray")

FUSION_DESC = {
    "max":      "Weighted Sum과 Noisy-OR 중 큰 값 (보수적, PdM 기본 권장)",
    "weighted": "가중합만 사용 (해석 용이)",
    "noisy_or": "Noisy-OR만 사용 (FN 최소화 우선)",
}

fusion = st.radio(
    "Risk Score 융합 전략",
    ["max", "weighted", "noisy_or"],
    index=["max", "weighted", "noisy_or"].index(st.session_state.fusion_method),
    format_func=lambda k: f"**{k}** \u2014 {FUSION_DESC[k]}",
    horizontal=True,
)
st.session_state.fusion_method = fusion

# =====================================================================
# 3. Risk 등급 임계값
# =====================================================================
st.header("Risk 등급 임계값", divider="gray")

levels_df = pd.DataFrame(config.RISK_LEVELS, columns=["등급", "임계값"])
levels_df["설명"] = [
    "\u2265 0.80 \u2014 즉시 정비 필요",
    "\u2265 0.50 \u2014 주의 관찰 필요",
    "\u2265 0.30 \u2014 경미한 이상 징후",
    "\u2265 0.00 \u2014 정상 운전",
]
st.dataframe(levels_df, use_container_width=True, hide_index=True)

# =====================================================================
# 4. 시스템 상태
# =====================================================================
st.header("시스템 상태", divider="gray")

status_items: list[tuple[str, str, str]] = []

# Python / Torch
status_items.append(("Python", sys.version.split()[0], "\u2705"))
try:
    import torch
    status_items.append(("PyTorch", torch.__version__, "\u2705"))
    status_items.append((
        "Device",
        config.get_device(),
        "\u2705 GPU" if config.get_device() == "cuda" else "\u26a1 CPU",
    ))
except ImportError:
    status_items.append(("PyTorch", "미설치", "\u274c"))

try:
    import streamlit as _st
    status_items.append(("Streamlit", _st.__version__, "\u2705"))
except Exception:
    pass

try:
    import pyngrok
    status_items.append(("pyngrok", pyngrok.__version__, "\u2705"))
except ImportError:
    status_items.append(("pyngrok", "미설치", "\u274c"))

try:
    import captum
    status_items.append(("Captum", captum.__version__, "\u2705"))
except ImportError:
    status_items.append(("Captum", "미설치", "\u26a0\ufe0f XAI 불가"))

ngrok_path = shutil.which("ngrok")
status_items.append((
    "ngrok CLI",
    ngrok_path if ngrok_path else "미발견",
    "\u2705" if ngrok_path else "\u26a0\ufe0f",
))

status_df = pd.DataFrame(status_items, columns=["항목", "값", "상태"])
st.dataframe(status_df, use_container_width=True, hide_index=True)

# =====================================================================
# 5. 체크포인트 목록
# =====================================================================
st.header("체크포인트 목록", divider="gray")

ckpts = sorted(
    config.CHECKPOINT_DIR.glob("*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if ckpts:
    ckpt_rows = []
    for p in ckpts[:30]:
        size = p.stat().st_size
        size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
        ckpt_rows.append({"파일": p.name, "크기": size_str, "확장자": p.suffix})
    st.dataframe(pd.DataFrame(ckpt_rows), use_container_width=True, hide_index=True)
else:
    st.info("체크포인트가 없습니다.")
