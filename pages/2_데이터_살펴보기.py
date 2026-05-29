"""HybridPdM - 데이터 살펴보기 (Data Lab).

내장 데이터셋 미리보기 + 사용자 CSV 업로드 분석. tab으로 두 모드를 분리.
"""
import streamlit as st

st.set_page_config(
    page_title="데이터 살펴보기 — HybridPdM",
    page_icon="\U0001f4c8",
    layout="wide",
)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui_common as ui

st.title("\U0001f4c8 데이터 살펴보기")
st.caption("내장 데이터셋의 구조와 통계를 확인하거나, 직접 가진 CSV 파일을 업로드해 분석합니다.")

ui.render_intent(
    what="모델 학습에 사용된 **6개 산업 데이터셋의 형태·통계·분포**를 살펴보고, "
         "필요하면 직접 가진 CSV 파일을 업로드해 같은 방식으로 분석할 수 있습니다.",
    when="데이터가 어떻게 생겼는지 처음 파악할 때, 새로 수집한 센서 데이터의 품질을 확인하고 싶을 때.",
    next_steps=[
        "[내장 데이터셋] — 6개 산업 데이터셋의 구조·통계·분포 확인",
        "[CSV 업로드] — 직접 가진 데이터로 기본 통계 분석",
    ],
)

tab1, tab2 = st.tabs(["내장 데이터셋", "CSV 업로드"])

# =====================================================================
# Tab 1: 내장 데이터셋 브라우저
# =====================================================================
with tab1:
    import config
    import data_pipeline as dp

    DS_OPTIONS: dict[str, str] = {
        "ai4i_cnn":          "AI4I — 11개 센서 (가공 장비, CNN 입력)",
        "ai4i_gbdt":         "AI4I — 11개 센서 (GBDT/CatBoost 평탄형)",
        "cwru_cnn":          "CWRU — 진동 신호 1024 샘플 (베어링, raw 1D)",
        "cwru_cnn_stft":     "CWRU — STFT 스펙트로그램 (베어링, 2D)",
        "hydraulic_ae":      "Hydraulic — 17개 센서 (cycle 평균)",
        "hydraulic_lstm_ae": "Hydraulic — 17개 센서 × 60 시점 (cycle 내 시계열)",
        "cmapss_lstm":       "C-MAPSS — 14개 센서 × 30 시점 (터빈 엔진)",
        "cmapss_lstm_w20":   "C-MAPSS — 14개 센서 × 20 시점",
        "cmapss_lstm_w50":   "C-MAPSS — 14개 센서 × 50 시점",
        "ncmapss_lstm":      "N-CMAPSS — 43개 센서 × 30 시점 (실제 비행 조건)",
    }

    st.markdown("### 분석할 데이터셋 선택")
    selected = st.selectbox(
        "데이터셋",
        list(DS_OPTIONS.keys()),
        format_func=lambda k: DS_OPTIONS[k],
    )

    if st.button("데이터 불러오기", key="load_existing", type="primary"):
        try:
            with st.spinner(f"`{selected}` 불러오는 중…"):
                data = dp.LOADERS[selected]()

            st.success("불러오기 완료")

            # 요약 메트릭
            c1, c2, c3 = st.columns(3)
            c1.metric("학습 데이터 수", f"{data['X_train'].shape[0]:,}",
                      help="모델을 학습시키는 데 사용된 데이터 건수")
            c2.metric("검증 데이터 수", f"{data['X_val'].shape[0]:,}",
                      help="학습 중 모델 성능 확인용 데이터 건수")
            c3.metric("테스트 데이터 수", f"{data['X_test'].shape[0]:,}",
                      help="최종 성능 평가용 데이터 건수")

            st.caption(f"데이터 한 건의 형태: `{data['X_train'].shape[1:]}`")

            # 메타 정보
            meta = data["meta"]
            with st.expander("데이터셋 상세 정보 (메타데이터)"):
                display_meta = {}
                for k, v in meta.items():
                    if isinstance(v, (list, np.ndarray)):
                        display_meta[k] = str(v)[:500]
                    else:
                        display_meta[k] = v
                st.json(display_meta)

            # 데이터 미리보기
            st.subheader("테스트 데이터 미리보기 (처음 30건)")
            X_preview = data["X_test"][:30].copy()
            if X_preview.ndim == 3:
                n_cols_show = min(X_preview.shape[1] * X_preview.shape[2], 50)
                X_preview = X_preview.reshape(X_preview.shape[0], -1)[:, :n_cols_show]
                if X_preview.shape[1] == n_cols_show and n_cols_show == 50:
                    st.caption("(3차원 데이터 → 2차원으로 펼침, 처음 50개 열만 표시)")

            feat_names = meta.get("feature_names")
            if feat_names and X_preview.shape[1] == len(feat_names):
                df_preview = pd.DataFrame(X_preview, columns=feat_names)
            else:
                df_preview = pd.DataFrame(X_preview)
            st.dataframe(df_preview, use_container_width=True)

            # 기본 통계
            with st.expander("기본 통계 — 평균/표준편차/최댓값 등 (테스트 데이터 기준)"):
                st.dataframe(df_preview.describe(), use_container_width=True)

            # 레이블 분포
            st.subheader("정답 라벨 분포 (테스트 데이터 기준)")
            st.caption("정상/이상이 얼마나 균형 있게 들어있는지 보여줍니다. 한쪽으로 심하게 치우치면 모델 학습이 어렵습니다.")
            y = np.asarray(data["y_test"])
            if np.issubdtype(y.dtype, np.floating) and np.allclose(y, y.astype(int)):
                y = y.astype(int)
            unique, counts = np.unique(y, return_counts=True)
            label_df = pd.DataFrame({"Label": unique.astype(str), "Count": counts})
            st.bar_chart(label_df, x="Label", y="Count")

        except FileNotFoundError as e:
            st.error(f"데이터 파일을 찾을 수 없습니다: {e}")
        except Exception as e:
            st.error(f"불러오기 실패: {e}")
            import traceback
            with st.expander("상세 오류 메시지"):
                st.code(traceback.format_exc())

# =====================================================================
# Tab 2: CSV 업로드
# =====================================================================
with tab2:
    st.markdown(
        "공장 설비에서 직접 수집한 센서 데이터(CSV)를 업로드하면 "
        "기본 통계와 데이터 형태를 자동으로 분석해줍니다."
    )

    uploaded = st.file_uploader(
        "CSV 파일 선택", type=["csv"],
        help="첫 줄에 컬럼 이름이 있고, 쉼표(,)로 구분된 일반적인 CSV 형식이면 모두 지원합니다.",
    )

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"불러오기 완료 — {df.shape[0]:,}행 × {df.shape[1]}열")

            c1, c2, c3 = st.columns(3)
            c1.metric("행 (데이터 건수)", f"{df.shape[0]:,}")
            c2.metric("열 (센서/항목 수)", df.shape[1])
            c3.metric("빈 칸 (결측치)", int(df.isnull().sum().sum()),
                      help="값이 비어 있는 셀의 총 개수. 많으면 사전 보정 필요.")

            st.subheader("데이터 미리보기 (처음 50행)")
            st.dataframe(df.head(50), use_container_width=True)

            st.subheader("기본 통계 — 평균/표준편차/최댓값")
            st.dataframe(df.describe(), use_container_width=True)

            st.subheader("각 열의 데이터 형식")
            dtype_df = (
                df.dtypes.astype(str)
                .to_frame("Type")
                .reset_index()
                .rename(columns={"index": "Column"})
            )
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)

            # 수치형 컬럼 히스토그램
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                st.subheader("센서 값 분포 보기")
                sel_col = st.selectbox("어느 센서 값을 볼까요?", num_cols)
                st.bar_chart(df[sel_col].dropna())

        except Exception as e:
            st.error(f"CSV 읽기 실패: {e}")

# =====================================================================
# Sidebar
# =====================================================================
ui.render_glossary_sidebar([
    "CNN", "AE", "LSTM", "GBDT",
    "RUL", "Anomaly Score", "PdM",
])
