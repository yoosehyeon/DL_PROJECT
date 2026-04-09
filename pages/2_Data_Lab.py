"""HybridPdM - Data Lab.

기존 데이터셋 미리보기 · CSV 업로드 · 기본 통계를 제공한다.
"""
import streamlit as st

st.set_page_config(
    page_title="Data Lab \u2014 HybridPdM",
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

st.title("\U0001f4c8 Data Lab")
st.caption("데이터셋 탐색 및 CSV 업로드 분석")

tab1, tab2 = st.tabs(["기존 데이터셋", "CSV 업로드"])

# =====================================================================
# Tab 1: 기존 데이터셋 브라우저
# =====================================================================
with tab1:
    import config
    import data_pipeline as dp

    DS_OPTIONS: dict[str, str] = {
        "ai4i_cnn":     "AI4I \u2014 CNN 입력 (B, 1, 11)",
        "ai4i_gbdt":    "AI4I \u2014 GBDT 입력 (B, 11)",
        "cwru_cnn":     "CWRU \u2014 CNN 입력 (B, 1, 1024)",
        "hydraulic_ae": "Hydraulic \u2014 AE 입력",
        "cmapss_lstm":  "C-MAPSS \u2014 LSTM 입력",
        "ncmapss_lstm": "N-CMAPSS \u2014 LSTM 입력",
    }

    selected = st.selectbox(
        "데이터셋",
        list(DS_OPTIONS.keys()),
        format_func=lambda k: DS_OPTIONS[k],
    )

    if st.button("데이터 로드", key="load_existing"):
        try:
            with st.spinner(f"`{selected}` 로딩 중\u2026"):
                data = dp.LOADERS[selected]()

            st.success("로드 완료!")

            # 요약 메트릭
            c1, c2, c3 = st.columns(3)
            c1.metric("Train", f"{data['X_train'].shape[0]:,} \u00d7 {data['X_train'].shape[1:]}")
            c2.metric("Val",   f"{data['X_val'].shape[0]:,}")
            c3.metric("Test",  f"{data['X_test'].shape[0]:,}")

            # 메타 정보
            meta = data["meta"]
            with st.expander("메타 정보"):
                display_meta = {}
                for k, v in meta.items():
                    if isinstance(v, (list, np.ndarray)):
                        display_meta[k] = str(v)[:500]
                    else:
                        display_meta[k] = v
                st.json(display_meta)

            # 데이터 미리보기
            st.subheader("Test 데이터 미리보기 (처음 30행)")
            X_preview = data["X_test"][:30].copy()
            if X_preview.ndim == 3:
                n_cols_show = min(X_preview.shape[1] * X_preview.shape[2], 50)
                X_preview = X_preview.reshape(X_preview.shape[0], -1)[:, :n_cols_show]
                if X_preview.shape[1] == n_cols_show and n_cols_show == 50:
                    st.caption("(3D 입력 \u2192 2D 전개, 처음 50개 컬럼만 표시)")

            feat_names = meta.get("feature_names")
            if feat_names and X_preview.shape[1] == len(feat_names):
                df_preview = pd.DataFrame(X_preview, columns=feat_names)
            else:
                df_preview = pd.DataFrame(X_preview)
            st.dataframe(df_preview, use_container_width=True)

            # 기본 통계
            with st.expander("기본 통계 (Test set)"):
                st.dataframe(df_preview.describe(), use_container_width=True)

            # 레이블 분포
            st.subheader("레이블 분포 (Test set)")
            y = np.asarray(data["y_test"])
            if np.issubdtype(y.dtype, np.floating) and np.allclose(y, y.astype(int)):
                y = y.astype(int)
            unique, counts = np.unique(y, return_counts=True)
            label_df = pd.DataFrame({"Label": unique.astype(str), "Count": counts})
            st.bar_chart(label_df, x="Label", y="Count")

        except FileNotFoundError as e:
            st.error(f"데이터셋을 찾을 수 없습니다: {e}")
        except Exception as e:
            st.error(f"로드 실패: {e}")
            import traceback
            with st.expander("상세 오류"):
                st.code(traceback.format_exc())

# =====================================================================
# Tab 2: CSV 업로드
# =====================================================================
with tab2:
    st.markdown(
        "AI4I, CWRU 등 PdM 데이터를 CSV로 업로드하면 "
        "기본 탐색적 분석(EDA)을 자동으로 수행합니다."
    )

    uploaded = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"로드 완료! ({df.shape[0]:,}행 \u00d7 {df.shape[1]}열)")

            c1, c2, c3 = st.columns(3)
            c1.metric("행", f"{df.shape[0]:,}")
            c2.metric("열", df.shape[1])
            c3.metric("결측치", int(df.isnull().sum().sum()))

            st.subheader("데이터 미리보기")
            st.dataframe(df.head(50), use_container_width=True)

            st.subheader("기본 통계")
            st.dataframe(df.describe(), use_container_width=True)

            st.subheader("데이터 타입")
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
                st.subheader("수치형 컬럼 분포")
                sel_col = st.selectbox("컬럼 선택", num_cols)
                st.bar_chart(df[sel_col].dropna())

        except Exception as e:
            st.error(f"CSV 파싱 실패: {e}")
