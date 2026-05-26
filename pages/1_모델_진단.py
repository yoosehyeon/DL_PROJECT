"""HybridPdM - 모델 진단 (Diagnostics).

선택한 장비/모델로 테스트 데이터를 다시 분석하여 성능 지표·예측 분포·
어느 센서가 결정에 영향을 줬는지(XAI)까지 한 화면에 보여준다.
정보 위계: [성능 지표] [예측 분포] [영향 센서(XAI)] 3개 탭으로 분리.
"""
import streamlit as st

st.set_page_config(
    page_title="모델 진단 — HybridPdM",
    page_icon="\U0001f50d",
    layout="wide",
)

import sys
import traceback
from pathlib import Path

import numpy as np

# ── 프로젝트 루트 ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import ui_common as ui

# matplotlib 한글 폰트 (Windows)
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 데이터셋 카탈로그 ────────────────────────────────────────────
DATASETS: dict[str, dict] = {
    "ai4i_cnn":     {"label": "AI4I — CNN (고장 분류)",        "task": "binary_cls",  "ext": ".pt"},
    "ai4i_gbdt":    {"label": "AI4I — GBDT (고장 분류)",       "task": "gbdt",        "ext": ".pkl"},
    "cwru_cnn":     {"label": "CWRU — CNN (베어링 결함 분류)", "task": "multi_cls",   "ext": ".pt"},
    "hydraulic_ae": {"label": "Hydraulic — AE (이상 감지)",     "task": "anomaly",     "ext": ".pt"},
    "cmapss_lstm":  {"label": "C-MAPSS — LSTM (남은 수명 예측)","task": "regression",  "ext": ".pt"},
    "ncmapss_lstm": {"label": "N-CMAPSS — LSTM (남은 수명 예측)","task": "regression", "ext": ".pt"},
}

EVAL_TASK_MAP = {
    "binary_cls": "binary_classification",
    "multi_cls":  "multiclass",
    "anomaly":    "anomaly_detection",
    "regression": "regression",
}


def _find_ckpt(name: str, ext: str) -> Path | None:
    files = sorted(
        config.CHECKPOINT_DIR.glob(f"{name}*{ext}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _batched_forward(model, X: np.ndarray, fn=None, batch_size: int = 256):
    import torch
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i : i + batch_size], dtype=torch.float32)
            out = fn(model, xb) if fn else model(xb)
            chunks.append(out.cpu().numpy())
    return np.concatenate(chunks, axis=0)


# =====================================================================
# 페이지 본문
# =====================================================================
st.title("\U0001f50d 모델 진단")
st.caption("선택한 장비/모델의 분석 결과와 결정에 영향을 준 센서를 확인합니다.")

ui.render_intent(
    what="선택한 장비에 대해 학습된 진단 모델로 **테스트 데이터를 다시 분석**하고, "
         "성능 점수·예측 결과 분포·어떤 센서가 결정에 영향을 줬는지를 한 화면에 보여줍니다.",
    when="대시보드의 위험 신호가 왜 그렇게 나왔는지 근거를 확인하고 싶을 때, "
         "특정 모델이 얼마나 잘 맞추는지 점검하고 싶을 때.",
    next_steps=[
        "분석할 장비/모델 선택 → [분석 실행] 클릭",
        "[성능 지표] — 모델이 얼마나 정확한지 숫자로 확인",
        "[예측 분포] — 그래프로 정상/이상이 잘 구분되는지 확인",
        "[영향 센서] — 어떤 센서가 판단에 가장 큰 영향을 줬는지 확인",
    ],
)

# =====================================================================
# 데이터셋 선택 + 체크포인트 확인
# =====================================================================
selected = st.selectbox(
    "분석할 장비/모델 선택",
    list(DATASETS.keys()),
    format_func=lambda k: DATASETS[k]["label"],
)
info = DATASETS[selected]

ckpt = _find_ckpt(selected, info["ext"])
if ckpt:
    st.success(f"준비된 모델 파일: `{ckpt.name}`")
else:
    st.error(
        f"`{selected}` 모델이 아직 학습되지 않았습니다. "
        f"터미널에서 `python main.py --datasets {selected}` 를 먼저 실행하세요."
    )
    st.stop()

if st.session_state.get("_diag_ds") != selected:
    st.session_state._diag_ds = selected
    st.session_state.pop("diag_results", None)

if st.button("\U0001f680 분석 실행", type="primary"):
    try:
        with st.status("분석 진행 중…", expanded=True) as status:
            import torch
            import data_pipeline as dp
            from main import PIPELINE

            st.write(f"데이터 불러오는 중: `{selected}`")
            data = dp.LOADERS[selected]()
            meta = data["meta"]

            st.write("모델 준비 및 학습 파일 불러오기")
            if selected == "ai4i_gbdt":
                import pickle
                with open(ckpt, "rb") as f:
                    model = pickle.load(f)
            else:
                _, _, build_fn = PIPELINE[selected]
                model = build_fn(data)
                state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()

            st.write("테스트 데이터 분석 중…")
            X_test = data["X_test"]
            y_test = np.asarray(data["y_test"])
            task = info["task"]
            results: dict = {"task": task, "y_test": y_test, "meta": meta}

            if task == "binary_cls":
                logits = _batched_forward(model, X_test).reshape(-1)
                results["probs"] = 1.0 / (1.0 + np.exp(-logits))
            elif task == "gbdt":
                results["probs"] = model.predict_proba(X_test)[:, 1]
            elif task == "multi_cls":
                logits = _batched_forward(model, X_test)
                probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
                results["probs"] = probs
                results["preds"] = probs.argmax(axis=1)
            elif task == "anomaly":
                results["anomaly_scores"] = _batched_forward(
                    model, X_test, fn=lambda m, x: m.anomaly_score(x),
                )
                results["preds"] = _batched_forward(
                    model, X_test, fn=lambda m, x: m.predict(x),
                ).astype(int)
            elif task == "regression":
                results["rul_preds"] = _batched_forward(model, X_test).reshape(-1)

            st.write("어떤 센서가 결정에 영향을 줬는지 계산 중…")
            results["xai"] = None
            results["xai_error"] = None
            if task != "gbdt":
                try:
                    import explain as ex
                    eval_task = EVAL_TASK_MAP[task]
                    explainer = ex.EXPLAINERS.get(eval_task)
                    if explainer:
                        n_sample = min(64, len(X_test))
                        feat_names = meta.get("feature_names")
                        results["xai"] = explainer(
                            model, X_test[:n_sample], feature_names=feat_names,
                        )
                except Exception as xe:
                    results["xai_error"] = str(xe)

            status.update(label="분석 완료", state="complete")

        st.session_state.diag_results = results

    except FileNotFoundError as e:
        st.error(f"데이터 파일을 찾을 수 없습니다: {e}")
    except Exception as e:
        st.error(f"분석 실패: {e}")
        with st.expander("상세 오류 메시지"):
            st.code(traceback.format_exc())

# =====================================================================
# 결과 표시 — 3개 탭으로 분리
# =====================================================================
if "diag_results" not in st.session_state:
    st.info("위 [분석 실행] 버튼을 누르면 분석 결과가 여기에 표시됩니다.")
    st.stop()

res = st.session_state.diag_results
task = res["task"]
y_test = res["y_test"]

tab_metrics, tab_dist, tab_xai = st.tabs([
    "성능 지표",
    "예측 분포",
    "영향 센서 (XAI)",
])

# ---------------------------------------------------------------------
# Tab A: 성능 지표
# ---------------------------------------------------------------------
with tab_metrics:
    st.subheader("모델이 얼마나 정확한가?")
    st.caption("테스트 데이터에서 모델이 얼마나 잘 맞췄는지를 숫자로 보여줍니다.")

    if task in ("binary_cls", "gbdt"):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        probs = res["probs"]
        threshold = st.slider(
            "판정 기준값 (Threshold)",
            0.0, 1.0,
            float(config.CNN_CFG["decision_threshold"]), 0.01,
            help="이 값 이상이면 '고장'으로 판정합니다. 낮출수록 의심 사례를 더 많이 잡지만 거짓 경보가 늘어납니다.",
            key="diag_thr",
        )
        y_pred = (probs > threshold).astype(int)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("전체 데이터", len(probs))
        m2.metric("고장 판정 수", int(y_pred.sum()),
                  help="모델이 위 기준값 이상으로 '고장'이라 판정한 데이터 수")
        m3.metric("Accuracy (정확도)", f"{accuracy_score(y_test, y_pred):.4f}",
                  help="전체 중 맞춘 비율. 단, 정상이 압도적으로 많으면 함정 — 다 정상이라 찍어도 99%가 나옴.")
        m4.metric("Precision (정밀도)", f"{precision_score(y_test, y_pred, zero_division=0):.4f}",
                  help="모델이 '고장'이라 한 것 중 진짜 고장이었던 비율. 높을수록 거짓 경보가 적음.")
        m5.metric("Recall (재현율)", f"{recall_score(y_test, y_pred, zero_division=0):.4f}",
                  help="진짜 고장 중 모델이 잡아낸 비율. 낮으면 고장을 놓치고 있다는 뜻 — 정비 중요한 지표.")
        st.metric("F1 점수", f"{f1_score(y_test, y_pred, zero_division=0):.4f}",
                  help="정밀도와 재현율의 균형 점수. 한쪽만 높아도 점수가 낮게 나옴. 1에 가까울수록 좋음.")

    elif task == "multi_cls":
        from sklearn.metrics import accuracy_score, f1_score
        preds = res["preds"]
        m1, m2, m3 = st.columns(3)
        m1.metric("전체 데이터", len(preds))
        m2.metric("Accuracy (정확도)", f"{accuracy_score(y_test, preds):.4f}",
                  help="전체 중 결함 유형을 정확히 맞춘 비율.")
        m3.metric("Macro F1", f"{f1_score(y_test, preds, average='macro', zero_division=0):.4f}",
                  help="결함 종류별로 점수를 따로 내고 평균낸 값. 특정 결함만 잘 맞추면 낮아짐.")

    elif task == "anomaly":
        from sklearn.metrics import f1_score, recall_score
        preds = res["preds"]
        n_anomaly = int(preds.sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("전체 데이터", len(preds))
        m2.metric("이상 감지 수", n_anomaly,
                  help="평상시 패턴과 다르다고 모델이 판단한 데이터 수")
        m3.metric("F1 점수", f"{f1_score(y_test, preds, zero_division=0):.4f}",
                  help="정밀도와 재현율의 균형 점수. 1에 가까울수록 좋음.")
        m4.metric("Recall (재현율)", f"{recall_score(y_test, preds, zero_division=0):.4f}",
                  help="진짜 이상 상황 중 모델이 잡아낸 비율. 놓치면 안 되는 상황에서 가장 중요.")

    elif task == "regression":
        rul_preds = res["rul_preds"]
        rmse = float(np.sqrt(np.mean((y_test - rul_preds) ** 2)))
        mae = float(np.mean(np.abs(y_test - rul_preds)))
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE (cycle)", f"{rmse:.2f}",
                  help="평균적으로 몇 cycle 틀렸나. 크게 빗나간 경우에 더 큰 페널티. 작을수록 좋음.")
        m2.metric("MAE (cycle)", f"{mae:.2f}",
                  help="평균적인 절대 오차. 직관적인 평균 오차 cycle 수. 작을수록 좋음.")
        m3.metric("전체 데이터", len(y_test))
        st.caption(
            "단위는 cycle(작업 주기). RMSE와 MAE 차이가 클수록 가끔 크게 빗나가는 경우가 있다는 뜻."
        )

# ---------------------------------------------------------------------
# Tab B: 예측 분포 시각화
# ---------------------------------------------------------------------
with tab_dist:
    st.subheader("정상 / 이상이 잘 구분되는가?")
    st.caption("실제 정답과 모델 예측을 그래프로 비교합니다. 두 분포가 잘 떨어질수록 모델이 잘 학습된 것.")

    if task in ("binary_cls", "gbdt"):
        probs = res["probs"]
        threshold = st.session_state.get(
            "diag_thr", float(config.CNN_CFG["decision_threshold"]),
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(probs[y_test == 0], bins=50, alpha=0.6, label="정상 (실제)", color="#4CAF50")
        ax.hist(probs[y_test == 1], bins=50, alpha=0.6, label="고장 (실제)", color="#FF5722")
        ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
                   label=f"판정 기준값 = {threshold:.2f}")
        ax.set_xlabel("고장 확률 (모델 예측)")
        ax.set_ylabel("건수")
        ax.set_title("예측 확률 분포")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "초록(정상)과 빨강(고장) 분포가 서로 떨어져 있을수록 모델이 둘을 잘 구분합니다. "
            "두 분포가 많이 겹치면 어떤 기준값을 써도 한계가 있다는 신호."
        )

    elif task == "multi_cls":
        preds = res["preds"]
        unique, counts = np.unique(preds, return_counts=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(unique.astype(str), counts, color="steelblue", edgecolor="black")
        ax.set_xlabel("결함 유형")
        ax.set_ylabel("건수")
        ax.set_title("결함 유형별 예측 분포")
        st.pyplot(fig)
        plt.close(fig)

    elif task == "anomaly":
        scores = res["anomaly_scores"]
        preds = res["preds"]
        n_anomaly = int(preds.sum())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(scores[y_test == 0], bins=50, alpha=0.6, label="정상 (실제)", color="#4CAF50")
        axes[0].hist(scores[y_test == 1], bins=50, alpha=0.6, label="이상 (실제)", color="#FF5722")
        axes[0].set_xlabel("이상 점수 [0, 1]")
        axes[0].set_ylabel("건수")
        axes[0].set_title("이상 점수 분포")
        axes[0].legend()
        sizes = [len(preds) - n_anomaly, n_anomaly]
        axes[1].pie(
            sizes, labels=["정상", "이상"], autopct="%1.1f%%",
            colors=["#4CAF50", "#FF5722"], startangle=90,
        )
        axes[1].set_title("이상 감지 결과 비율")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    elif task == "regression":
        rul_preds = res["rul_preds"]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_test, rul_preds, alpha=0.3, s=10, color="steelblue")
        lim = max(float(y_test.max()), float(rul_preds.max())) * 1.05
        ax.plot([0, lim], [0, lim], "r--", alpha=0.7, label="완벽 예측선 (정답)")
        ax.set_xlabel("실제 남은 수명 (cycle)")
        ax.set_ylabel("예측한 남은 수명 (cycle)")
        ax.set_title("예측 vs 실제 — 남은 수명")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "점이 빨간 대각선에 가까울수록 정확. "
            "대각선 위쪽이면 수명을 실제보다 길게 봤다(위험할 수 있음), "
            "아래쪽이면 짧게 봤다(불필요한 정비가 늘어남)."
        )

# ---------------------------------------------------------------------
# Tab C: 영향 센서 (XAI)
# ---------------------------------------------------------------------
with tab_xai:
    st.subheader("어떤 센서가 결정에 가장 영향을 줬나?")
    st.caption(
        "모델이 판단을 내릴 때 어느 센서/입력이 가장 큰 영향을 줬는지 보여줍니다. "
        "값이 클수록 그 센서가 결정에 중요했다는 뜻."
    )

    if task == "gbdt":
        st.info(
            "GBDT 모델은 이 방식의 분석을 지원하지 않습니다. "
            "GBDT 자체가 가진 특성 중요도(feature_importances_)를 모델 파일에서 직접 확인할 수 있습니다."
        )
    elif res.get("xai"):
        xai = res["xai"]
        top_k = xai["top_k"]
        names = [item["name"] for item in top_k]
        attr_scores = [item["score"] for item in top_k]

        fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.4)))
        ax.barh(names[::-1], attr_scores[::-1], color="teal", edgecolor="black")
        ax.set_xlabel("영향도 점수")
        ax.set_title(f"상위 {len(top_k)}개 영향 센서/입력")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("자세한 데이터 보기"):
            import pandas as pd
            st.dataframe(pd.DataFrame(top_k), use_container_width=True, hide_index=True)
    elif res.get("xai_error"):
        st.warning(f"영향 센서 분석 실패: {res['xai_error']}")
    else:
        st.info("분석 결과가 없습니다.")

# =====================================================================
# Sidebar 용어집
# =====================================================================
ui.render_glossary_sidebar([
    "Accuracy", "Precision", "Recall", "F1", "AUC",
    "RMSE", "MAE", "R²",
    "Threshold", "Decision Threshold",
    "Anomaly Score", "RUL", "FN", "FP",
    "XAI", "IG",
])
