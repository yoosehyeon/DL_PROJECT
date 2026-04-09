"""HybridPdM - 중앙 설정 모듈.

경로, 모델별 하이퍼파라미터, Risk Score 가중치, 재현성(seed) 헬퍼를 한 곳에서 관리한다.
이 모듈을 import하는 것만으로 numpy/torch/random seed가 고정된다.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# torch는 학습/추론 시 필수지만, config 자체는 torch 없이도 import 가능해야 함
try:
    import torch
except ImportError:
    torch = None  # type: ignore

# ---------------------------------------------------------------------------
# 1) 경로 설정
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "dataset"
ARTIFACT_ROOT = ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACT_ROOT / "checkpoints"
REPORT_DIR = ARTIFACT_ROOT / "reports"

for _d in (ARTIFACT_ROOT, CHECKPOINT_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 데이터셋별 경로. 키는 모델/스크립트 전반에서 사용되는 데이터셋 식별자.
DATASET_PATHS = {
    "ai4i":      DATA_ROOT / "ai4i2020.csv",
    "cwru":      DATA_ROOT / "10987113",
    "hydraulic": DATA_ROOT / "condition+monitoring+of+hydraulic+systems",
    "cmapss":    DATA_ROOT / "CMAPSSData",
    "ncmapss":   DATA_ROOT / "17. Turbofan Engine Degradation Simulation Data Set 2" / "data_set",
}

# ---------------------------------------------------------------------------
# 2) 재현성(Reproducibility)
# ---------------------------------------------------------------------------
SEED = 42

def set_seed(seed: int = SEED) -> None:
    """python/numpy/torch(CPU+CUDA)의 모든 난수 시드를 일괄 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker별 시드 고정 헬퍼.

    num_workers>0으로 DataLoader를 사용할 때 각 워커의 numpy/random 시드가
    무작위로 바뀌어 비결정적 동작이 되는 문제를 방지한다.
    DataLoader(worker_init_fn=config.worker_init_fn) 형태로 전달.
    """
    seed = SEED + worker_id
    np.random.seed(seed)
    random.seed(seed)


def get_device():
    """GPU가 있으면 'cuda', 아니면 'cpu'를 반환한다."""
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# 3) 모델별 하이퍼파라미터
# ---------------------------------------------------------------------------

# 3.1) 1D-CNN (고장 분류)
CNN_CFG = {
    "epochs": 50,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.2,
    "decision_threshold": 0.35,
    # Focal Loss: 극불균형 이진 분류에서 easy negative를 down-weight하여
    # hard positive에 학습 자원을 집중. alpha로 클래스 균형, gamma로 easy 샘플 억제.
    # Focal Loss와 BCE+pos_weight는 배타 사용 (use_focal_loss로 분기).
    "use_focal_loss": True,
    "focal_alpha": 0.85,
    "focal_gamma": 2.0,
}

# 3.2) Autoencoder (이상 탐지)
AE_CFG = {
    "epochs": 60,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "latent_dim": 16,
    "early_stop_patience": 10,
    "threshold_grid": list(range(85, 100)),  # val F1을 최대화하는 percentile 탐색 구간
}

# 3.3) LSTM (RUL 예측) — C-MAPSS 기본
LSTM_CFG = {
    "epochs": 80,
    "batch_size": 128,
    "lr": 1e-3,
    "dropout": 0.4,
    "weight_decay": 1e-4,
    "early_stop_patience": 10,
    "hidden": 128,
    "num_layers": 2,
    "window": 30,
    "rul_clip": 125,
    "grad_clip": 1.0,
    "use_lr_scheduler": True,
    "use_huber_loss": False,
}

# 3.4) LSTM (RUL 예측) — N-CMAPSS 전용
# 43피처(C-MAPSS 14피처의 3배)에 맞춰 hidden 확대, 에폭·데이터량 증가.
# Huber Loss로 이상치(큰 RUL 오차)에 대한 민감도를 줄인다.
NCMAPSS_LSTM_CFG = {
    "epochs": 120,
    "batch_size": 128,
    "lr": 1e-3,
    "dropout": 0.3,
    "weight_decay": 1e-4,
    "early_stop_patience": 15,
    "hidden": 256,
    "num_layers": 2,
    "window": 30,
    "rul_clip": 125,
    "grad_clip": 1.0,
    "use_lr_scheduler": True,
    "use_huber_loss": True,
    "huber_delta": 5.0,
    # data_pipeline 전용 파라미터
    "stride": 10,
    "max_units_train": 20,
    "max_units_test": 20,
    "max_windows_per_unit": 3000,
}

# ---------------------------------------------------------------------------
# 4) Risk Score
# ---------------------------------------------------------------------------
# 산식: R = w1*P(failure) + w2*Anomaly + w3*(1 - RUL_norm)
RISK_WEIGHTS = {"failure": 0.5, "anomaly": 0.3, "rul": 0.2}

# 등급 임계값 (내림차순). 점수마다 첫 번째 매칭 등급을 부여한다.
RISK_LEVELS = [
    ("Critical", 0.80),
    ("Warning",  0.50),
    ("Advisory", 0.30),
    ("Normal",   0.00),
]

# import 시점에 자동으로 시드를 고정한다.
set_seed()
