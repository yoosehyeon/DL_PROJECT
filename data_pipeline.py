"""HybridPdM - 통합 데이터 파이프라인.

5개 데이터셋(AI4I, CWRU, Hydraulic, C-MAPSS, N-CMAPSS)을 통합 처리한다.
모든 loader는 다음 규약을 따른다:
  - 입력: 없음 (경로는 config.DATASET_PATHS에서 자동 조회)
  - 출력: dict(X_train, y_train, X_val, y_val, X_test, y_test, meta)
  - 데이터셋이 없거나 파싱 실패 시 FileNotFoundError를 발생시켜 main에서 skip 가능

설계 원칙:
  1) 학습/검증/테스트 split은 시간/엔진 단위로 분리하여 데이터 누수 방지
  2) 정규화 통계는 train에서만 계산하여 val/test에 적용
  3) 모든 텐서는 PyTorch 호환 numpy float32로 반환
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config

# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def _split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: Optional[np.ndarray] = None,
):
    """X, y를 train/val/test 3개로 분할.

    sklearn의 train_test_split을 두 번 적용한다.
    분류 문제(stratify가 주어진 경우)는 클래스 비율을 유지한다.
    """
    # 1차: train+val vs test
    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=test_size, random_state=config.SEED, stratify=stratify
    )
    # 2차: train vs val (val_size를 잔여 데이터 비율로 환산)
    rel_val = val_size / (1.0 - test_size)
    strat_tv = y_tv if stratify is not None else None
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=rel_val, random_state=config.SEED, stratify=strat_tv
    )
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def _fit_apply_scaler_2d(
    X_train: np.ndarray, *others: np.ndarray
) -> Tuple[StandardScaler, List[np.ndarray]]:
    """2D (N, F) 입력에 대해 train 기반 StandardScaler를 fit·transform.

    AE/AI4I-CNN 등 채널이 1인 벡터형 입력에 사용한다.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    out = [scaler.transform(X_train).astype(np.float32)]
    for o in others:
        out.append(scaler.transform(o).astype(np.float32))
    return scaler, out


def _scale_seq(X_train: np.ndarray, *others: np.ndarray):
    """3D 시계열 (N, L, F) 입력을 채널 단위로 정규화.

    train 데이터에서 (N*L, F) 평균/표준편차를 계산해 모든 split에 적용한다.
    """
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

    def _apply(x: np.ndarray) -> np.ndarray:
        s = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        return s.astype(np.float32)

    return scaler, [_apply(X_train)] + [_apply(o) for o in others]


# ===========================================================================
# 1) AI4I 2020 - 1D-CNN 분류 입력 (B, 1, 11)
# ===========================================================================

# AI4I 11-피처의 정규 컬럼 순서. 이 모듈 외부(예: models.py)에서도 동일하게
# 참조할 수 있도록 상수로 노출한다. 학습/추론 사이의 컬럼 순서 일관성을 보장.
AI4I_FEATURE_NAMES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_L",
    "Type_M",
    "Type_H",
    "power",       # rpm × torque
    "temp_diff",   # process_temp - air_temp
    "strain",      # tool_wear × torque
]


def load_ai4i_cnn() -> Dict:
    """AI4I 2020 데이터셋을 1D-CNN 입력 형태로 로드.

    shape (B, 1, 11). 11개 피처 구성:
      - 5개 원본 수치 피처 (Air temp, Process temp, RPM, Torque, Tool wear)
      - 3개 Type one-hot (L, M, H 순서 고정)
      - 3개 파생 피처 (power=torque*rpm, temp_diff, strain=tool_wear*torque)
    레이블은 'Machine failure' 이진값.
    """
    path = config.DATASET_PATHS["ai4i"]
    if not path.exists():
        raise FileNotFoundError(f"AI4I dataset not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")  # BOM 처리

    # 원본 수치 피처 5개
    num_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    # Type one-hot 3개 (학습/추론 일관성을 위해 L/M/H 순서 고정)
    type_oh = pd.get_dummies(df["Type"], prefix="Type")
    for c in ["Type_L", "Type_M", "Type_H"]:
        if c not in type_oh.columns:
            type_oh[c] = 0
    type_oh = type_oh[["Type_L", "Type_M", "Type_H"]]

    # 파생 피처 3개 (도메인 지식: 기계적 부하 관련 변수)
    power = (df["Rotational speed [rpm]"] * df["Torque [Nm]"]).rename("power")
    temp_diff = (df["Process temperature [K]"] - df["Air temperature [K]"]).rename("temp_diff")
    strain = (df["Tool wear [min]"] * df["Torque [Nm]"]).rename("strain")

    feats = pd.concat(
        [df[num_cols], type_oh.astype(float), power, temp_diff, strain], axis=1
    )
    # 컬럼 개수 + 컬럼 순서 모두 검증.
    # 컬럼 순서가 바뀌면 모델의 reorder index가 무음으로 잘못 적용되므로 fail-fast.
    assert feats.shape[1] == 11, f"AI4I feature count mismatch: {feats.shape[1]}"
    actual_cols = list(feats.columns)
    assert actual_cols == AI4I_FEATURE_NAMES, (
        f"AI4I column order mismatch.\n"
        f"  expected: {AI4I_FEATURE_NAMES}\n"
        f"  got     : {actual_cols}"
    )

    X = feats.values.astype(np.float32)
    y = df["Machine failure"].values.astype(np.float32)

    X_tr, y_tr, X_va, y_va, X_te, y_te = _split_train_val_test(X, y, stratify=y)
    _, (X_tr, X_va, X_te) = _fit_apply_scaler_2d(X_tr, X_va, X_te)

    # 채널 차원 추가: (N, 11) → (N, 1, 11)
    X_tr = X_tr[:, None, :]
    X_va = X_va[:, None, :]
    X_te = X_te[:, None, :]

    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val":   X_va, "y_val":   y_va,
        "X_test":  X_te, "y_test":  y_te,
        "meta": {
            "name": "AI4I",
            "task": "binary_classification",
            "feature_dim": 11,
            "feature_names": list(AI4I_FEATURE_NAMES),
        },
    }


def load_ai4i_gbdt() -> Dict:
    """AI4I를 GradientBoosting 입력 (B, 11)로 로드.

    AE와 달리 정상-only 필터링 없이 전체 라벨을 사용한다 (supervised).
    형상은 (B, 11) flat — sklearn tabular API와 일치.
    """
    data = load_ai4i_cnn()
    flatten = lambda x: x.squeeze(1)  # (B, 1, 11) → (B, 11)
    return {
        "X_train": flatten(data["X_train"]),
        "y_train": data["y_train"],
        "X_val":   flatten(data["X_val"]),
        "y_val":   data["y_val"],
        "X_test":  flatten(data["X_test"]),
        "y_test":  data["y_test"],
        "meta": {
            "name": "AI4I-GBDT",
            "task": "gbdt_binary",
            "feature_dim": 11,
            "feature_names": list(data["meta"]["feature_names"]),
        },
    }


def load_ai4i_ae() -> Dict:
    """AI4I를 Autoencoder 입력 (B, 11)로 로드.

    이상 탐지 표준 관행에 따라 학습은 정상 샘플(label=0)로만 수행하고,
    검증/평가는 전체 데이터로 한다.
    """
    data = load_ai4i_cnn()
    flatten = lambda x: x.squeeze(1)  # (B, 1, 11) → (B, 11)
    X_tr_full = flatten(data["X_train"])
    y_tr = data["y_train"]
    normal_mask = y_tr == 0
    return {
        "X_train": X_tr_full[normal_mask],
        "y_train": np.zeros(int(normal_mask.sum()), dtype=np.float32),
        "X_val":   flatten(data["X_val"]),
        "y_val":   data["y_val"],
        "X_test":  flatten(data["X_test"]),
        "y_test":  data["y_test"],
        "meta": {"name": "AI4I-AE", "task": "anomaly_detection", "feature_dim": 11},
    }


# ===========================================================================
# 2) CWRU 베어링 진동 - 1D-CNN (B, 1, 1024)
# ===========================================================================

# CWRU 파일 번호 → 고장 유형(IR/B/OR) 매핑.
# 본 데이터 폴더에는 baseline normal(97-100)이 없어, 3-class 고장 유형 분류로 처리.
# 매핑은 CWRU 공식 데이터 카탈로그를 종합한 결과이다.
def _cwru_class_for(file_num: int) -> Optional[int]:
    """CWRU 파일 번호를 IR(0)/B(1)/OR(2) 3-class로 매핑.

    매핑에 없는 파일은 None을 반환하여 호출자가 skip하도록 한다.
    라벨이 명확한 파일만 사용하면 학습 신호가 깨끗해진다.
    """
    ir_ranges = [(105, 108), (169, 172), (209, 212), (278, 281)]
    b_ranges  = [(118, 121), (185, 188), (222, 225), (282, 285), (286, 289)]
    or_ranges = [
        (130, 133), (144, 147), (156, 159),
        (197, 200), (234, 237), (246, 249), (258, 261),
        (294, 297), (298, 301), (310, 313),
    ]
    def _in(num, rngs):
        return any(a <= num <= b for a, b in rngs)
    if _in(file_num, ir_ranges): return 0
    if _in(file_num, b_ranges):  return 1
    if _in(file_num, or_ranges): return 2
    return None  # 매핑 없음 → 호출자가 skip


def _read_cwru_signal(mat_path: Path) -> Optional[np.ndarray]:
    """CWRU .mat에서 단일 진동 신호를 1D로 추출.

    DE(Drive End) 가속도계가 베어링 결함 신호를 가장 명확하게 포착하므로
    DE 우선, 없으면 FE(Fan End)로 fallback 한다 (CWRU 학계 표준 관행).
    """
    try:
        import scipy.io as sio
    except ImportError as e:
        raise RuntimeError("scipy is required for CWRU loader") from e
    try:
        d = sio.loadmat(str(mat_path))
    except Exception:
        return None
    de_keys = [k for k in d.keys() if k.endswith("_DE_time")]
    fe_keys = [k for k in d.keys() if k.endswith("_FE_time")]
    chosen = de_keys[0] if de_keys else (fe_keys[0] if fe_keys else None)
    if chosen is None:
        return None
    return np.asarray(d[chosen]).reshape(-1).astype(np.float32)


def load_cwru_cnn(window: int = 1024, stride: int = 512, max_per_file: int = 200) -> Dict:
    """CWRU를 1D-CNN 입력 (B, 1, 1024)로 슬라이딩 윈도우 분할 로드.

    파일 단위 train/val/test split으로 데이터 누수를 방지한다.
    윈도우 단위 random split은 같은 .mat 파일에서 나온 인접 윈도우들이
    train/test 양쪽에 들어가 leakage를 일으킨다.
    파일을 먼저 stratified split한 뒤 각 split의 파일에서만 윈도우를 추출한다.
    """
    root = config.DATASET_PATHS["cwru"]
    if not root.exists():
        raise FileNotFoundError(f"CWRU dataset not found: {root}")

    # ── 1) 유효 파일 수집 (라벨 확정된 파일만) ─────────────────────────
    file_entries: List[Tuple[Path, int]] = []  # (path, class)
    for f in sorted(root.glob("*.mat")):
        m = re.match(r"(\d+)\.mat", f.name)
        if not m:
            continue
        num = int(m.group(1))
        cls = _cwru_class_for(num)
        if cls is None:
            continue
        file_entries.append((f, cls))

    if not file_entries:
        raise RuntimeError("CWRU: no labeled files found")

    file_classes = np.array([c for _, c in file_entries], dtype=np.int64)
    _, file_counts = np.unique(file_classes, return_counts=True)
    if file_counts.min() < 3:
        warnings.warn(
            f"CWRU: smallest class has only {file_counts.min()} files. "
            "Falling back to non-stratified file split.",
            UserWarning,
        )
        strat = None
    else:
        strat = file_classes

    # ── 2) 파일 인덱스 단위 train/val/test split (leakage 차단) ──────
    file_idx = np.arange(len(file_entries))
    idx_tr, _, idx_va, _, idx_te, _ = _split_train_val_test(
        file_idx, file_classes, stratify=strat,
    )

    # ── 3) 각 split의 파일에서만 윈도우 추출 ───────────────────────────
    def _extract_from_indices(indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for i in indices:
            f, cls = file_entries[int(i)]
            sig = _read_cwru_signal(f)
            if sig is None or sig.size < window:
                continue
            n_windows = (sig.size - window) // stride + 1
            n_windows = min(n_windows, max_per_file)
            for k in range(n_windows):
                start = k * stride
                Xs.append(sig[start:start + window])
                ys.append(cls)
        if not Xs:
            return (np.empty((0, window), dtype=np.float32),
                    np.empty((0,), dtype=np.int64))
        X = np.stack(Xs).astype(np.float32)[:, None, :]   # (N, 1, window)
        y = np.array(ys, dtype=np.int64)
        return X, y

    X_tr, y_tr = _extract_from_indices(idx_tr)
    X_va, y_va = _extract_from_indices(idx_va)
    X_te, y_te = _extract_from_indices(idx_te)

    if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
        raise RuntimeError(
            f"CWRU file-level split produced empty split: "
            f"train={len(X_tr)} val={len(X_va)} test={len(X_te)}"
        )

    # 진동 신호는 신호 단위로 정규화 (per-window z-score)가 표준이므로
    # 채널 통계 대신 각 윈도우 내에서 정규화한다 → 부하/속도 변화에 강건
    def _per_window_norm(arr: np.ndarray) -> np.ndarray:
        mu = arr.mean(axis=-1, keepdims=True)
        sd = arr.std(axis=-1, keepdims=True) + 1e-6
        return ((arr - mu) / sd).astype(np.float32)

    X_tr = _per_window_norm(X_tr)
    X_va = _per_window_norm(X_va)
    X_te = _per_window_norm(X_te)

    n_classes = int(len(np.unique(file_classes)))   # 실제 존재하는 클래스 수 (파일 부재 시 3 미만 가능)
    return {
        "X_train": X_tr, "y_train": y_tr.astype(np.int64),
        "X_val":   X_va, "y_val":   y_va.astype(np.int64),
        "X_test":  X_te, "y_test":  y_te.astype(np.int64),
        "meta": {"name": "CWRU", "task": "multiclass", "n_classes": n_classes, "feature_dim": 1024},
    }


# ===========================================================================
# 3) Hydraulic - Autoencoder (B, 17)
# ===========================================================================

# 17개 다변량 센서: PS1-6 + EPS1 + FS1-2 + TS1-4 + VS1 + CE + CP + SE
# 각 센서는 cycle별로 다른 샘플레이트를 가지므로 cycle당 평균값으로 집계한다.
HYDRAULIC_SENSORS = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",  # 압력 (100Hz)
    "EPS1",                                     # 전력 (100Hz)
    "FS1", "FS2",                               # 유량 (10Hz)
    "TS1", "TS2", "TS3", "TS4",                 # 온도 (1Hz)
    "VS1",                                      # 진동 (1Hz)
    "CE", "CP", "SE",                           # 효율/파워/스와시
]


def load_hydraulic_ae() -> Dict:
    """Hydraulic 시스템을 AE 입력 (B, 17)로 로드.

    각 센서 파일은 (cycles × samples_per_cycle) 행렬이다.
    cycle 단위 평균을 취해 17차원 벡터를 만든다.
    label은 profile.txt의 stable_flag 컬럼으로 정의:
      - stable=1 → 정상(0)
      - stable=0 → 이상(1)
    """
    root = config.DATASET_PATHS["hydraulic"]
    if not root.exists():
        raise FileNotFoundError(f"Hydraulic dataset not found: {root}")

    feats = []
    for s in HYDRAULIC_SENSORS:
        fp = root / f"{s}.txt"
        if not fp.exists():
            raise FileNotFoundError(f"Hydraulic sensor file missing: {fp}")
        arr = np.loadtxt(fp)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        feats.append(arr.mean(axis=1))   # cycle별 평균
    X = np.stack(feats, axis=1).astype(np.float32)   # (cycles, 17)

    profile = np.loadtxt(root / "profile.txt", dtype=int)
    # profile.txt는 cooler, valve, pump, hydraulic, stable 5개 컬럼이 표준.
    # 컬럼 수가 부족하면 stable_flag 인덱스가 잘못된 컬럼을 가리킬 수 있어 검증.
    if profile.ndim != 2 or profile.shape[1] < 5:
        raise RuntimeError(
            f"Hydraulic profile.txt unexpected shape {profile.shape}; "
            "expected at least 5 columns (stable_flag at index 4)."
        )
    stable_flag = profile[:, 4]
    y = (stable_flag == 0).astype(np.float32)  # 1=이상, 0=정상

    X_tr, y_tr, X_va, y_va, X_te, y_te = _split_train_val_test(X, y, stratify=y)

    # AE는 정상 샘플만으로 학습. 정규화 통계도 정상 train으로만 fit.
    normal_mask = y_tr == 0
    if not normal_mask.any():
        raise RuntimeError(
            "Hydraulic train split has no normal (stable) samples. "
            "Cannot train AE without normal data."
        )
    _, (X_tr_n, X_va, X_te) = _fit_apply_scaler_2d(X_tr[normal_mask], X_va, X_te)

    return {
        "X_train": X_tr_n,
        "y_train": np.zeros(len(X_tr_n), dtype=np.float32),
        "X_val":   X_va, "y_val": y_va,
        "X_test":  X_te, "y_test": y_te,
        "meta": {"name": "Hydraulic", "task": "anomaly_detection", "feature_dim": 17},
    }


# ===========================================================================
# 4) C-MAPSS - LSTM RUL (B, 14, 30) (PRD: B,C,L 채널 우선)
# ===========================================================================

# C-MAPSS 컬럼 정의
CMAPSS_COLS = ["unit", "cycle"] + [f"op{i}" for i in (1, 2, 3)] + [f"s{i}" for i in range(1, 22)]
# 14개 유효 센서 (분산이 거의 0인 센서 제외 - C-MAPSS 표준 전처리)
CMAPSS_USE_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11",
                      "s12", "s13", "s14", "s15", "s17", "s20", "s21"]


def _build_rul_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window: int,
    rul_clip: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """엔진별 시계열을 sliding window로 잘라 (N, window, F)와 RUL 레이블을 만든다.

    각 엔진에 대해 piecewise-linear RUL을 계산: max_cycle - current_cycle을
    rul_clip(125)으로 위쪽 clip. 이는 C-MAPSS 표준 RUL 가공법이다.
    """
    Xs, ys = [], []
    for _, g in df.groupby("unit"):
        g = g.sort_values("cycle").reset_index(drop=True)
        feats = g[feature_cols].values.astype(np.float32)
        max_cycle = g["cycle"].max()
        ruls = np.minimum(rul_clip, max_cycle - g["cycle"].values).astype(np.float32)
        if len(g) < window:
            continue
        for i in range(len(g) - window + 1):
            Xs.append(feats[i:i + window])
            ys.append(ruls[i + window - 1])
    if not Xs:
        # 모든 엔진이 window보다 짧은 경우. 호출자가 처리할 수 있도록 명시 오류.
        raise RuntimeError(
            "_build_rul_windows produced no windows; "
            f"all units have cycle count < window={window}"
        )
    return np.stack(Xs), np.array(ys, dtype=np.float32)


def load_cmapss_lstm(subset: str = "FD001") -> Dict:
    """C-MAPSS FD001을 LSTM 입력 (B, 14, 30)로 로드.

    엔진 단위로 train/val 분할(데이터 누수 방지). test set은 별도 파일과
    RUL_*.txt에서 로드하며, 각 엔진의 마지막 window만 평가에 사용한다.
    """
    root = config.DATASET_PATHS["cmapss"]
    train_fp = root / f"train_{subset}.txt"
    test_fp  = root / f"test_{subset}.txt"
    rul_fp   = root / f"RUL_{subset}.txt"
    if not train_fp.exists():
        raise FileNotFoundError(f"C-MAPSS file missing: {train_fp}")

    train_df = pd.read_csv(train_fp, sep=r"\s+", header=None, names=CMAPSS_COLS)
    test_df  = pd.read_csv(test_fp,  sep=r"\s+", header=None, names=CMAPSS_COLS)
    rul_test = np.loadtxt(rul_fp, dtype=np.float32)

    window = config.LSTM_CFG["window"]
    rul_clip = config.LSTM_CFG["rul_clip"]

    # 엔진 단위 train/val split (random 80/20) - 누수 방지
    units = train_df["unit"].unique()
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(units)
    n_val = max(1, int(len(units) * 0.2))
    val_units = set(units[:n_val])
    tr_units  = set(units[n_val:])

    tr_part = train_df[train_df["unit"].isin(tr_units)]
    va_part = train_df[train_df["unit"].isin(val_units)]

    feature_cols = CMAPSS_USE_SENSORS
    X_tr, y_tr = _build_rul_windows(tr_part, feature_cols, window, rul_clip)
    X_va, y_va = _build_rul_windows(va_part, feature_cols, window, rul_clip)

    # Test: 각 엔진의 마지막 window만 사용.
    # 길이 부족 시 0-패딩 대신 첫 프레임을 forward-fill 한다.
    # 0-패딩은 정규화 후 평균값과 다른 의미를 가져 분포를 오염시키므로 회피.
    X_te_list, y_te_list = [], []
    for i, unit in enumerate(sorted(test_df["unit"].unique())):
        g = test_df[test_df["unit"] == unit].sort_values("cycle")
        feats = g[feature_cols].values.astype(np.float32)
        if len(feats) < window:
            pad_n = window - len(feats)
            first = feats[:1]                       # (1, F)
            pad = np.repeat(first, pad_n, axis=0)   # 첫 프레임 반복
            feats = np.concatenate([pad, feats], axis=0)
        X_te_list.append(feats[-window:])
        y_te_list.append(min(rul_clip, rul_test[i]))
    X_te = np.stack(X_te_list)
    y_te = np.array(y_te_list, dtype=np.float32)

    # 정규화 (train의 (N*L, F) 통계로 fit)
    _, (X_tr, X_va, X_te) = _scale_seq(X_tr, X_va, X_te)

    # shape (B, F, L) = (B, 14, 30) — 시계열 모델 입력은 (B, L, F) 이지만
    # channel-first 형식으로 저장하고 모델에서 transpose 한다.
    def _to_bcl(x):
        return x.transpose(0, 2, 1).astype(np.float32)

    return {
        "X_train": _to_bcl(X_tr), "y_train": y_tr,
        "X_val":   _to_bcl(X_va), "y_val":   y_va,
        "X_test":  _to_bcl(X_te), "y_test":  y_te,
        "meta": {"name": f"C-MAPSS-{subset}", "task": "regression",
                 "feature_dim": len(feature_cols), "window": window,
                 "rul_clip": rul_clip},
    }


# ===========================================================================
# 5) N-CMAPSS - LSTM (B, 47, 30)
# ===========================================================================

def load_ncmapss_lstm(file_name: str = "N-CMAPSS_DS01-005.h5",
                      stride: Optional[int] = None,
                      max_units_train: Optional[int] = None,
                      max_units_test: Optional[int] = None,
                      max_windows_per_unit: Optional[int] = None) -> Dict:
    """N-CMAPSS h5 파일을 LSTM 입력 (B, 43, 30)로 로드.

    피처 구성 (총 43개):
      X_s(14) + X_v(14) + T(10) + W(4) + Fc(1) = 43

    누수 제거 사유:
      - A 컬럼의 unit ID, cycle index, hs(health state flag)는 레이블/식별자에
        해당하여 학습에 포함하면 RUL 예측 점수가 부풀려짐.
      - Fc(flight class)만 운전조건의 일부로 의미가 있어 유지.

    원본 데이터는 수백만 row로 매우 크므로 다음으로 다운샘플링한다:
      - max_units_*: 사용할 엔진(unit) 수 제한
      - stride: 시계열 step 간격 (10이면 10샘플마다 1개)
      - max_windows_per_unit: 엔진당 최대 window 수

    RUL 타깃 정규화:
      - y / rul_clip → [0, 1] 범위. 모델이 절대 수명이 아니라 상대 열화 비율을
        학습하여 dev/test 엔진 간 분포 불일치에 강인해진다.
      - meta["rul_norm"] = True로 표시. 평가 시 pred * rul_clip으로 복원.
    """
    # config.NCMAPSS_LSTM_CFG에서 기본값을 가져온다
    _ncfg = config.NCMAPSS_LSTM_CFG
    if stride is None:
        stride = _ncfg.get("stride", 10)
    if max_units_train is None:
        max_units_train = _ncfg.get("max_units_train", 20)
    if max_units_test is None:
        max_units_test = _ncfg.get("max_units_test", 20)
    if max_windows_per_unit is None:
        max_windows_per_unit = _ncfg.get("max_windows_per_unit", 3000)
    try:
        import h5py
    except ImportError as e:
        raise RuntimeError("h5py is required for N-CMAPSS loader") from e

    root = config.DATASET_PATHS["ncmapss"]
    fp = root / file_name
    if not fp.exists():
        raise FileNotFoundError(f"N-CMAPSS file missing: {fp}")

    window   = config.LSTM_CFG["window"]
    rul_clip = config.LSTM_CFG["rul_clip"]

    with h5py.File(str(fp), "r") as f:
        def _stack(prefix):
            X_s = np.asarray(f[f"X_s_{prefix}"])  # (N, 14) 측정 센서
            X_v = np.asarray(f[f"X_v_{prefix}"])  # (N, 14) 가상 센서
            T   = np.asarray(f[f"T_{prefix}"])    # (N, 10) 건강상태 변수
            W   = np.asarray(f[f"W_{prefix}"])    # (N, 4)  운전조건(고도·마하 등)
            A   = np.asarray(f[f"A_{prefix}"])    # (N, 4)  unit·cycle·Fc·hs
            Y   = np.asarray(f[f"Y_{prefix}"]).reshape(-1)  # RUL
            # 누수 회피: A에서 Fc(index 2)만 운전조건 피처로 채택.
            # unit ID/cycle/hs는 레이블 정보를 포함하므로 학습 입력에서 제외.
            Fc = A[:, 2:3].astype(np.float32)
            mat = np.concatenate(
                [X_s, X_v, T, W, Fc], axis=1
            ).astype(np.float32)
            assert mat.shape[1] == 43, f"N-CMAPSS feature count mismatch: {mat.shape[1]}"
            unit_id = A[:, 0].astype(np.int32)
            return mat, Y.astype(np.float32), unit_id

        dev_X, dev_Y, dev_U = _stack("dev")
        test_X, test_Y, test_U = _stack("test")

    def _windows_per_unit(X, Y, U, max_units, max_per_unit):
        """unit별로 stride 다운샘플링 후 슬라이딩 윈도우 추출."""
        units = np.unique(U)[:max_units]
        Xs, Ys = [], []
        for u in units:
            mask = U == u
            x_u = X[mask][::stride]
            y_u = Y[mask][::stride]
            if len(x_u) < window:
                continue
            n_win = min(len(x_u) - window + 1, max_per_unit)
            for i in range(n_win):
                Xs.append(x_u[i:i + window])
                Ys.append(min(rul_clip, y_u[i + window - 1]))
        if not Xs:
            raise RuntimeError("N-CMAPSS: no windows extracted")
        return np.stack(Xs), np.array(Ys, dtype=np.float32)

    # 엔진(unit) 단위 train/val split — window 단위 shuffle은 같은 엔진의
    # 인접 window가 train/val 양쪽에 들어가는 시간적 누수를 일으키므로
    # 엔진 단위로 분할하여 방지한다.
    dev_units = np.unique(dev_U)[:max_units_train]
    rng_u = np.random.default_rng(config.SEED)
    rng_u.shuffle(dev_units)
    n_val_u = max(1, int(len(dev_units) * 0.2))
    val_dev_u = set(dev_units[:n_val_u])
    tr_dev_u  = set(dev_units[n_val_u:])
    if len(tr_dev_u) == 0:
        raise RuntimeError(
            f"N-CMAPSS: max_units_train={max_units_train}이 너무 작아 "
            "엔진 단위 train/val split이 불가합니다. max_units_train >= 2 필요."
        )

    def _mask_units(X, Y, U, units):
        m = np.isin(U, list(units))
        return X[m], Y[m], U[m]

    tr_X_r, tr_Y_r, tr_U_r = _mask_units(dev_X, dev_Y, dev_U, tr_dev_u)
    va_X_r, va_Y_r, va_U_r = _mask_units(dev_X, dev_Y, dev_U, val_dev_u)

    X_tr, y_tr = _windows_per_unit(tr_X_r, tr_Y_r, tr_U_r, len(tr_dev_u), max_windows_per_unit)
    X_va, y_va = _windows_per_unit(va_X_r, va_Y_r, va_U_r, len(val_dev_u), max_windows_per_unit)
    X_te, y_te = _windows_per_unit(test_X, test_Y, test_U, max_units_test, max_windows_per_unit)

    # RUL 타깃 정규화: y / rul_clip → [0, 1]
    # 모델이 절대 수명이 아니라 상대 열화 비율을 학습하게 하여
    # dev/test 엔진 간 분포 불일치에 강인해진다.
    y_tr = y_tr / rul_clip
    y_va = y_va / rul_clip
    y_te = y_te / rul_clip

    _, (X_tr, X_va, X_te) = _scale_seq(X_tr, X_va, X_te)

    def _to_bcl(x):
        return x.transpose(0, 2, 1).astype(np.float32)

    return {
        "X_train": _to_bcl(X_tr), "y_train": y_tr,
        "X_val":   _to_bcl(X_va), "y_val":   y_va,
        "X_test":  _to_bcl(X_te), "y_test":  y_te,
        "meta": {"name": "N-CMAPSS", "task": "regression",
                 "feature_dim": 43, "window": window,
                 "rul_clip": rul_clip, "rul_norm": True},
    }


# ---------------------------------------------------------------------------
# 데이터셋 등록부 - main에서 일괄 호출용
# ---------------------------------------------------------------------------
LOADERS = {
    "ai4i_cnn":      load_ai4i_cnn,
    "ai4i_ae":       load_ai4i_ae,        # legacy 비교용 (PIPELINE에서는 제외)
    "ai4i_gbdt":     load_ai4i_gbdt,
    "cwru_cnn":      load_cwru_cnn,
    "hydraulic_ae":  load_hydraulic_ae,
    "cmapss_lstm":   load_cmapss_lstm,
    "ncmapss_lstm":  load_ncmapss_lstm,
}
