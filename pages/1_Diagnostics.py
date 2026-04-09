"""HybridPdM - 개별 모델 진단.

데이터셋 선택 → 체크포인트 로드 → 추론 → XAI 피처 중요도까지 일괄 수행한다.
"""
import streamlit as st

st.set_page_config(
    page_title="Diagnostics \u2014 HybridPdM",
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

# matplotlib 한글 폰트 (Windows)
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 데이터셋 정보 ────────────────────────────────────────────────
DATASETS: dict[str, dict] = {
    "ai4i_cnn":     {"label": "AI4I \u2014 CNN 이진 분류",       "task": "binary_cls",  "ext": ".pt"},
    "ai4i_gbdt":    {"label": "AI4I \u2014 GBDT 이진 분류",      "task": "gbdt",        "ext": ".pkl"},
    "cwru_cnn":     {"label": "CWRU \u2014 CNN 다중 분류",       "task": "multi_cls",   "ext": ".pt"},
    "hydraulic_ae": {"label": "Hydraulic \u2014 AE 이상 탐지",   "task": "anomaly",     "ext": ".pt"},
    "cmapss_lstm":  {"label": "C-MAPSS \u2014 LSTM RUL 예측",   "task": "regression",  "ext": ".pt"},
    "ncmapss_lstm": {"label": "N-CMAPSS \u2014 LSTM RUL 예측",  "task": "regression",  "ext": ".pt"},
}

EVAL_TASK_MAP = {
    "binary_cls": "binary_classification",
    "multi_cls":  "multiclass",
    "anomaly":    "anomaly_detection",
    "regression": "regression",
}


def _find_ckpt(name: str, ext: str) -> Path | None:
    """가장 최근 체크포인트 경로를 반환한다."""
    files = sorted(
        config.CHECKPOINT_DIR.glob(f"{name}*{ext}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _batched_forward(model, X: np.ndarray, fn=None, batch_size: int = 256):
    """배치 단위 추론. fn이 주어지면 fn(model, xb)를 호출한다."""
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
st.title("\U0001f50d Diagnostics")
st.caption("데이터셋별 모델 추론 결과와 XAI 피처 중요도를 확인합니다.")

selected = st.selectbox(
    "데이터셋 선택",
    list(DATASETS.keys()),
    format_func=lambda k: DATASETS[k]["label"],
)
info = DATASETS[selected]

# 체크포인트 존재 확인
ckpt = _find_ckpt(selected, info["ext"])
if ckpt:
    st.success(f"체크포인트: `{ckpt.name}`")
else:
    st.error(
        f"`{selected}` 체크포인트가 없습니다. "
        f"`python main.py --datasets {selected}`를 실행하세요."
    )
    st.stop()

# 데이터셋이 바뀌면 이전 결과 제거
if st.session_state.get("_diag_ds") != selected:
    st.session_state._diag_ds = selected
    st.session_state.pop("diag_results", None)

# ── 분석 실행 버튼 ────────────────────────────────────────────────
if st.button("\U0001f680 분석 실행", type="primary"):
    try:
        with st.status("분석 진행 중\u2026", expanded=True) as status:
            import torch
            import data_pipeline as dp
            from main import PIPELINE

            # 1) 데이터 로드
            st.write(f"\u2699\ufe0f 데이터 로딩: `{selected}`")
            data = dp.LOADERS[selected]()
            meta = data["meta"]

            # 2) 모델 빌드 + 체크포인트 로드
            st.write("\U0001f4e6 모델 빌드 + 체크포인트 로딩")
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

            # 3) 추론
            st.write("\U0001f52e 추론 중\u2026")
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

            # 4) XAI (선택적 — captum 없으면 건너뜀)
            st.write("\U0001f9e0 XAI 분석 중\u2026")
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

            status.update(label="\u2705 분석 완료!", state="complete")

        st.session_state.diag_results = results

    except FileNotFoundError as e:
        st.error(f"데이터셋을 찾을 수 없습니다: {e}")
    except Exception as e:
        st.error(f"분석 실패: {e}")
        with st.expander("상세 오류"):
            st.code(traceback.format_exc())

# =====================================================================
# 결과 표시
# =====================================================================
if "diag_results" not in st.session_state:
    st.stop()

res = st.session_state.diag_results
task = res["task"]
y_test = res["y_test"]

st.divider()
st.subheader("추론 결과")

# ── 이진 분류 ─────────────────────────────────────────────────────
if task in ("binary_cls", "gbdt"):
    probs = res["probs"]
    threshold = st.slider(
        "Decision Threshold", 0.0, 1.0,
        float(config.CNN_CFG["decision_threshold"]), 0.01,
    )
    y_pred = (probs > threshold).astype(int)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("전체 샘플", len(probs))
    m2.metric("고장 예측", int(y_pred.sum()))
    m3.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    m4.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs[y_test == 0], bins=50, alpha=0.6, label="정상 (y=0)", color="#4CAF50")
    ax.hist(probs[y_test == 1], bins=50, alpha=0.6, label="고장 (y=1)", color="#FF5722")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("P(failure)")
    ax.set_ylabel("Count")
    ax.set_title("예측 확률 분포")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ── 다중 분류 ─────────────────────────────────────────────────────
elif task == "multi_cls":
    preds = res["preds"]
    from sklearn.metrics import accuracy_score, f1_score

    m1, m2, m3 = st.columns(3)
    m1.metric("전체 샘플", len(preds))
    m2.metric("Accuracy", f"{accuracy_score(y_test, preds):.4f}")
    m3.metric(
        "Macro F1",
        f"{f1_score(y_test, preds, average='macro', zero_division=0):.4f}",
    )

    unique, counts = np.unique(preds, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(unique.astype(str), counts, color="steelblue", edgecolor="black")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("예측 클래스 분포")
    st.pyplot(fig)
    plt.close(fig)

# ── 이상 탐지 ─────────────────────────────────────────────────────
elif task == "anomaly":
    scores = res["anomaly_scores"]
    preds = res["preds"]
    from sklearn.metrics import f1_score

    n_anomaly = int(preds.sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("전체 샘플", len(preds))
    m2.metric("이상 감지", n_anomaly)
    m3.metric("정상", len(preds) - n_anomaly)
    m4.metric("F1", f"{f1_score(y_test, preds, zero_division=0):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # 히스토그램
    axes[0].hist(scores[y_test == 0], bins=50, alpha=0.6, label="정상", color="#4CAF50")
    axes[0].hist(scores[y_test == 1], bins=50, alpha=0.6, label="이상", color="#FF5722")
    axes[0].set_xlabel("Anomaly Score [0, 1]")
    axes[0].set_ylabel("Count")
    axes[0].set_title("이상 점수 분포")
    axes[0].legend()
    # 파이 차트
    sizes = [len(preds) - n_anomaly, n_anomaly]
    axes[1].pie(
        sizes, labels=["정상", "이상"], autopct="%1.1f%%",
        colors=["#4CAF50", "#FF5722"], startangle=90,
    )
    axes[1].set_title("탐지 결과")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── RUL 회귀 ──────────────────────────────────────────────────────
elif task == "regression":
    rul_preds = res["rul_preds"]
    rmse = float(np.sqrt(np.mean((y_test - rul_preds) ** 2)))
    mae = float(np.mean(np.abs(y_test - rul_preds)))

    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", f"{rmse:.2f}")
    m2.metric("MAE", f"{mae:.2f}")
    m3.metric("전체 샘플", len(y_test))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, rul_preds, alpha=0.3, s=10, color="steelblue")
    lim = max(float(y_test.max()), float(rul_preds.max())) * 1.05
    ax.plot([0, lim], [0, lim], "r--", alpha=0.7, label="Perfect")
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("RUL 예측 vs 실제")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# =====================================================================
# XAI 섹션
# =====================================================================
st.divider()
st.subheader("\U0001f9e0 XAI \u2014 피처 중요도 (Integrated Gradients)")

if task == "gbdt":
    st.info("GBDT 모델은 Captum Integrated Gradients를 지원하지 않습니다.")
elif res.get("xai"):
    xai = res["xai"]
    top_k = xai["top_k"]
    names = [item["name"] for item in top_k]
    attr_scores = [item["score"] for item in top_k]

    fig, ax = plt.subplots(figsize=(10, max(3, len(names) * 0.4)))
    ax.barh(names[::-1], attr_scores[::-1], color="teal", edgecolor="black")
    ax.set_xlabel("Attribution Score (|IG|)")
    ax.set_title(f"Top-{len(top_k)} Feature Importance")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("상세 데이터"):
        import pandas as pd

        st.dataframe(pd.DataFrame(top_k), use_container_width=True, hide_index=True)
elif res.get("xai_error"):
    st.warning(f"XAI 분석 실패: {res['xai_error']}")
else:
    st.info("XAI 결과가 없습니다.")
