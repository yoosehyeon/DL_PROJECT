"""HybridPdM - 학습/평가 파이프라인 진입점.

사용 예:
    python main.py                       # 모든 데이터셋 학습+평가
    python main.py --datasets ai4i_cnn cmapss_lstm
    python main.py --skip-explain        # IG attribution 단계 건너뛰기
    python main.py --smoke               # epoch=1 짧은 동작 검증

각 데이터셋은 다음 파이프라인을 거친다:
  1) data_pipeline.LOADERS[name]() 로 데이터 로드 (없으면 skip)
  2) 적절한 model 빌드 (cnn_vibration / cnn_tabular / ae / lstm)
  3) train.TRAINERS[task]() 학습 + 체크포인트 저장
  4) evaluate.EVALUATORS[task]() 평가
  5) (옵션) explain.EXPLAINERS[task]() 해석

각 단계에서 발생한 오류는 캐치하여 dataset 별로 격리한다 → 한 데이터셋의
누락이 다른 데이터셋의 학습을 막지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
import data_pipeline as dp
import evaluate as ev
import models
import train as tr


# ---------------------------------------------------------------------------
# 데이터셋 → 모델/태스크 매핑
# ---------------------------------------------------------------------------
# 각 항목은 (loader_key, task, build_model_fn) 의 의미.
# build_model_fn은 data dict를 받아 모델을 즉시 빌드한다 (feature_dim 반영).

def _build_ai4i_cnn(data: Dict):
    feat_names = data["meta"]["feature_names"]
    reorder = models.build_reorder_index(feat_names, models.AI4I_REORDER_NAMES)
    return models.build_model(
        "cnn_tabular",
        in_channels=1,
        seq_len=data["meta"]["feature_dim"],
        n_classes=1,
        dropout=config.CNN_CFG["dropout"],
        reorder_index=reorder,
        expected_feature_names=feat_names,
    )


def _build_ai4i_ae(data: Dict):
    return models.build_model(
        "ae",
        input_dim=data["meta"]["feature_dim"],
        latent_dim=config.AE_CFG["latent_dim"],
    )


def _build_ai4i_gbdt(data: Dict):
    """AI4I용 HistGradientBoostingClassifier 빌더.

    tabular 데이터에 최적화된 tree-based 모델. CPU에서 빠른 학습 + 클래스 불균형에 강함.
    하이퍼파라미터는 AI4I 규모(10K, 11피처, 3% 양성)에 맞춘 보수적 기본값.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=config.SEED,
    )


def _build_ai4i_catboost(data: Dict):
    """AI4I용 CatBoostClassifier 빌더 (Phase 1-1).

    sklearn HistGradient 대비 차별점:
      - Ordered Boosting → train data 누수에 더 강함 (overfit 방어)
      - Symmetric Tree → 추론 속도 빠름, regularization 자동
      - auto_class_weights='Balanced' → 불균형 자동 보정 (n_neg/n_pos)
    예상 효과: HistGradient F1 0.865 → CatBoost F1 0.87~0.89
    """
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        # 클래스 불균형 자동 보정 — sklearn 'balanced' 와 동일 원리
        auto_class_weights="Balanced",
        # 자동 overfit 방지
        early_stopping_rounds=30,
        eval_metric="F1",
        random_seed=config.SEED,
        verbose=False,
        allow_writing_files=False,   # tmp 파일 자동 생성 방지
    )


def _build_cwru_cnn(data: Dict):
    return models.build_model(
        "cnn_vibration",
        in_channels=1,
        n_classes=data["meta"]["n_classes"],
        dropout=config.CNN_CFG["dropout"],
    )


def _build_cwru_cnn_stft(data: Dict):
    """CWRU STFT 스펙트로그램 (B, 1, F, T) 입력용 2D-CNN 빌더."""
    return models.build_model(
        "cnn_vibration_stft",
        in_channels=1,
        n_classes=data["meta"]["n_classes"],
        dropout=config.CNN_CFG["dropout"],
    )


def _build_hydraulic_ae(data: Dict):
    return models.build_model(
        "ae",
        input_dim=data["meta"]["feature_dim"],
        latent_dim=config.AE_CFG["latent_dim"],
    )


def _build_hydraulic_lstm_ae(data: Dict):
    """Hydraulic용 LSTM-AE 빌더 (Phase 1-2).

    Dense AE는 cycle당 평균 1개로 압축 → 시간 정보 손실.
    LSTM-AE는 cycle 내 60 timestep × 17 센서 패턴을 학습 → 시간적 이상 탐지.
    예상 효과: F1 0.896 → 0.94 (5%p 향상 예상)
    """
    return models.build_model(
        "lstm_ae",
        n_features=data["meta"]["n_features"],
        seq_len=data["meta"]["seq_len"],
        hidden_dim=64,
        latent_dim=16,
        num_layers=1,
        noise_std=0.02,
    )


def _build_lstm(data: Dict):
    return models.build_model(
        "lstm",
        input_dim=data["meta"]["feature_dim"],
        hidden=config.LSTM_CFG["hidden"],
        num_layers=config.LSTM_CFG["num_layers"],
        dropout=config.LSTM_CFG["dropout"],
        input_format="BFL",
    )


def _build_cmapss_lstm_w20(data: Dict):
    """C-MAPSS Multi-Window Ensemble용 — window=20 LSTM (Phase 1-3)."""
    return models.build_model(
        "lstm",
        input_dim=data["meta"]["feature_dim"],
        hidden=config.LSTM_CFG["hidden"],
        num_layers=config.LSTM_CFG["num_layers"],
        dropout=config.LSTM_CFG["dropout"],
        input_format="BFL",
    )


def _build_cmapss_lstm_w50(data: Dict):
    """C-MAPSS Multi-Window Ensemble용 — window=50 LSTM (Phase 1-3)."""
    return models.build_model(
        "lstm",
        input_dim=data["meta"]["feature_dim"],
        hidden=config.LSTM_CFG["hidden"],
        num_layers=config.LSTM_CFG["num_layers"],
        dropout=config.LSTM_CFG["dropout"],
        input_format="BFL",
    )


def _build_ncmapss_lstm(data: Dict):
    """N-CMAPSS 전용 LSTM 빌더. 43피처에 맞춘 더 큰 hidden을 사용한다."""
    cfg = config.NCMAPSS_LSTM_CFG
    return models.build_model(
        "lstm",
        input_dim=data["meta"]["feature_dim"],
        hidden=cfg["hidden"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        input_format="BFL",
    )


# 등록부: name → (task_for_train, task_for_eval, builder)
# ── AI4I 권장 운영 구성 (검증 결과 기반): ────────────────────────────
#   1순위 단독: ai4i_catboost   (F1 0.893, 가장 강함)
#   2순위 단독: ai4i_gbdt       (F1 0.865)
#   2-way 스태킹: ai4i_gbdt + ai4i_catboost (F1 0.885 — 단독보다 약간 낮음)
#   ai4i_cnn 은 다양성 확보용이지만 스태킹에서 가중치 0으로 수렴 → 보조 신호로만 활용
PIPELINE = {
    "ai4i_cnn":      ("classification",    "binary_classification", _build_ai4i_cnn),
    "ai4i_gbdt":     ("gbdt_binary",       "gbdt_binary",           _build_ai4i_gbdt),
    "ai4i_catboost": ("gbdt_binary",       "gbdt_binary",           _build_ai4i_catboost),  # 권장 단독 모델
    "cwru_cnn":      ("classification",    "multiclass",            _build_cwru_cnn),
    "cwru_cnn_stft": ("classification",    "multiclass",            _build_cwru_cnn_stft),
    "hydraulic_ae":  ("anomaly_detection", "anomaly_detection",     _build_hydraulic_ae),
    "hydraulic_lstm_ae": ("anomaly_detection", "anomaly_detection", _build_hydraulic_lstm_ae),
    "cmapss_lstm":   ("regression",        "regression",            _build_lstm),
    "cmapss_lstm_w20": ("regression",      "regression",            _build_cmapss_lstm_w20),
    "cmapss_lstm_w50": ("regression",      "regression",            _build_cmapss_lstm_w50),
    "ncmapss_lstm":  ("regression",        "regression",            _build_ncmapss_lstm),
}


# ---------------------------------------------------------------------------
# 단일 데이터셋 파이프라인
# ---------------------------------------------------------------------------

def run_one(name: str, smoke: bool, skip_explain: bool) -> Tuple[Dict, Optional[Any], Optional[Dict]]:
    """name 데이터셋 1개에 대해 load → train → eval (→ explain).

    Returns:
      (result_dict, trained_model_or_None, data_dict_or_None)
      모델/데이터는 후처리(앙상블 등)를 위해 main에서 재사용한다.
    """
    if name not in PIPELINE:
        return {"name": name, "status": "unknown", "error": "no pipeline entry"}, None, None
    train_task, eval_task, build_fn = PIPELINE[name]

    # 1) 데이터 로드
    try:
        data = dp.LOADERS[name]()
    except FileNotFoundError as e:
        return {"name": name, "status": "skipped", "reason": str(e)}, None, None
    except Exception as e:
        return {
            "name": name, "status": "load_failed",
            "error": str(e), "trace": traceback.format_exc(),
        }, None, None

    # 2) 모델 빌드
    try:
        model = build_fn(data)
        # AI4I tabular CNN 등 nn.Module의 컬럼 일치 검증 (sklearn 모델은 hasattr=False)
        if hasattr(model, "verify_data_compatibility"):
            model.verify_data_compatibility(data["meta"]["feature_names"])
    except Exception as e:
        return {
            "name": name, "status": "build_failed",
            "error": str(e), "trace": traceback.format_exc(),
        }, None, None

    # 3) 학습 (smoke 모드는 epoch=1로 패치)
    try:
        train_fn = tr.TRAINERS[train_task]
        # N-CMAPSS만 전용 config 사용 (다른 모델은 기존 기본값 그대로)
        is_ncmapss = (name == "ncmapss_lstm")
        if smoke:
            cfg_map = {
                "classification":    {**config.CNN_CFG,  "epochs": 1},
                "anomaly_detection": {**config.AE_CFG,   "epochs": 1},
                "regression":        {**config.LSTM_CFG, "epochs": 1},
                "gbdt_binary":       {},
            }
            if is_ncmapss:
                cfg_map["regression"] = {**config.NCMAPSS_LSTM_CFG, "epochs": 1}
            train_metrics = train_fn(name, data, model, cfg=cfg_map[train_task])
        else:
            if is_ncmapss:
                train_metrics = train_fn(name, data, model, cfg=config.NCMAPSS_LSTM_CFG)
            else:
                train_metrics = train_fn(name, data, model)
    except Exception as e:
        return {
            "name": name, "status": "train_failed",
            "error": str(e), "trace": traceback.format_exc(),
        }, None, None

    # 4) 평가
    try:
        eval_fn = ev.EVALUATORS[eval_task]
        if eval_task == "binary_classification":
            eval_metrics = eval_fn(
                name, data, model,
                decision_threshold=config.CNN_CFG["decision_threshold"],
            )
        elif eval_task == "anomaly_detection":
            eval_metrics = eval_fn(name, data, model, use_mahalanobis=False)
        else:
            eval_metrics = eval_fn(name, data, model)
    except Exception as e:
        return {
            "name": name, "status": "eval_failed",
            "train": train_metrics,
            "error": str(e), "trace": traceback.format_exc(),
        }, None, None

    result = {
        "name": name,
        "status": "ok",
        "train": train_metrics,
        "eval":  eval_metrics,
    }

    # 5) 해석 (옵션). GBDT는 captum IG 대상이 아니므로 skip한다.
    if not skip_explain and eval_task != "gbdt_binary":
        try:
            import explain as ex
            ex_fn = ex.EXPLAINERS[eval_task]
            # 작은 sample로 IG (속도 우선)
            X_sample = data["X_test"][:64]
            feat_names = data["meta"].get("feature_names")
            if eval_task == "regression":
                result["explain"] = ex_fn(model, X_sample, feature_names=feat_names)
            elif eval_task == "anomaly_detection":
                result["explain"] = ex_fn(model, X_sample, feature_names=feat_names)
            else:
                result["explain"] = ex_fn(model, X_sample, feature_names=feat_names)
        except Exception as e:
            result["explain_error"] = str(e)

    return result, model, data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HybridPdM v2.1 학습/평가 파이프라인")
    parser.add_argument(
        "--datasets", nargs="+", default=list(PIPELINE.keys()),
        help=f"실행할 데이터셋 키 (기본: 전체). 가능: {list(PIPELINE.keys())}",
    )
    parser.add_argument("--smoke", action="store_true", help="epoch=1로 빠른 동작 검증")
    parser.add_argument("--skip-explain", action="store_true", help="Captum 해석 단계 건너뛰기")
    args = parser.parse_args()

    # 실행별 고유 ID — 체크포인트와 리포트 파일에 포함되어 덮어쓰기를 방지한다.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tr.set_run_id(run_id)

    print(f"[HybridPdM] run_id  = {run_id}")
    print(f"[HybridPdM] device  = {config.get_device()}")
    print(f"[HybridPdM] datasets = {args.datasets}")
    if args.smoke:
        print("[HybridPdM] *** SMOKE MODE: epochs=1 ***")

    results: List[Dict] = []
    # 후처리(스태킹 앙상블 등)를 위해 학습된 모델/데이터를 보관.
    # AI4I CNN과 GBDT 둘 다 성공하면 자동으로 스태킹 평가를 추가한다.
    trained: Dict[str, Tuple[Any, Dict]] = {}
    for name in args.datasets:
        print(f"\n=== {name} ===")
        res, model_obj, data_obj = run_one(name, smoke=args.smoke, skip_explain=args.skip_explain)
        status = res.get("status")
        if status == "ok":
            ev_m = res.get("eval", {})
            summary = {k: v for k, v in ev_m.items()
                       if k in ("accuracy", "f1", "rmse", "mae", "r2",
                                "test_f1", "best_threshold", "best_percentile")}
            print(f"  [OK]  {summary}")
            if model_obj is not None and data_obj is not None:
                trained[name] = (model_obj, data_obj)
        elif status == "skipped":
            print(f"  [--] skipped: {res.get('reason')}")
        else:
            print(f"  [FAIL] {status}: {res.get('error')}")
        results.append(res)

    # ── 후처리: AI4I 모델 자동 스태킹 (n-way) ─────────────────────────
    # AI4I 계열 모델(cnn, gbdt, catboost) 중 학습 성공한 것들을 자동으로 묶어
    # val grid 동시 탐색으로 최적 가중치·threshold 채택.
    # 2개 이상 성공 시에만 시도. 3개 다 성공하면 3-way 스태킹.
    # ── 후처리: C-MAPSS Multi-Window Ensemble (Phase 1-3) ──────────────
    # cmapss_lstm (w=30), cmapss_lstm_w20, cmapss_lstm_w50 중 2개 이상 학습되면 자동 평균
    cmapss_members = {
        name: trained[name]
        for name in ("cmapss_lstm", "cmapss_lstm_w20", "cmapss_lstm_w50")
        if name in trained
    }
    if len(cmapss_members) >= 2:
        member_str = " + ".join(cmapss_members.keys())
        print(f"\n=== cmapss_multiwindow_ensemble ({member_str}) ===")
        try:
            mw_metrics = ev.evaluate_cmapss_multiwindow_ensemble(cmapss_members)
            print(f"  [OK]  simple_avg:   "
                  f"RMSE={mw_metrics['simple_avg']['rmse']:.2f} "
                  f"MAE={mw_metrics['simple_avg']['mae']:.2f} "
                  f"R²={mw_metrics['simple_avg']['r2']:.3f}")
            print(f"        weighted_avg: "
                  f"RMSE={mw_metrics['weighted_avg']['rmse']:.2f} "
                  f"MAE={mw_metrics['weighted_avg']['mae']:.2f} "
                  f"R²={mw_metrics['weighted_avg']['r2']:.3f}")
            results.append({
                "name": "cmapss_multiwindow_ensemble",
                "status": "ok",
                "train": {"note": f"multi-window ensemble: {member_str}"},
                "eval":  mw_metrics,
            })
        except Exception as e:
            print(f"  [FAIL] cmapss_multiwindow_ensemble: {e}")
            results.append({
                "name": "cmapss_multiwindow_ensemble", "status": "ensemble_failed",
                "error": str(e), "trace": traceback.format_exc(),
            })

    ai4i_members = {
        name: trained[name]
        for name in ("ai4i_cnn", "ai4i_gbdt", "ai4i_catboost")
        if name in trained
    }
    if len(ai4i_members) >= 2:
        member_str = " + ".join(ai4i_members.keys())
        print(f"\n=== ai4i_stack ({member_str}) ===")
        try:
            stack_metrics = ev.evaluate_ai4i_stacking_nway(ai4i_members)
            w_str = " ".join(f"{k}={v:.2f}" for k, v in stack_metrics["weights"].items())
            print(f"  [OK]  {w_str}  "
                  f"thr={stack_metrics['decision_threshold']:.2f}  "
                  f"f1={stack_metrics['f1']:.4f}")
            results.append({
                "name": "ai4i_stack",
                "status": "ok",
                "train": {"note": f"post-hoc ensemble of {member_str}"},
                "eval":  stack_metrics,
            })
        except Exception as e:
            print(f"  [FAIL] ai4i_stack: {e}")
            results.append({
                "name": "ai4i_stack", "status": "stack_failed",
                "error": str(e), "trace": traceback.format_exc(),
            })

    # 전체 결과를 리포트로 저장 (run_id 포함 → 덮어쓰기 없이 누적)
    out_path = config.REPORT_DIR / f"pipeline_report_{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tr._json_safe(results), f, indent=2, ensure_ascii=False)
    print(f"\n[HybridPdM] report saved → {out_path}")


if __name__ == "__main__":
    main()
