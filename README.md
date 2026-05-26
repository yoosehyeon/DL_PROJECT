# HybridPdM — Hybrid Deep Learning 기반 예지보전(PdM) 시스템

> **여러 산업용 센서 데이터셋에 대해 분류 · 이상탐지 · RUL 예측을 동시에 수행하고, 세 모델의 출력을 통합한 단일 Risk Score로 장비 상태를 진단하는 End-to-End 파이프라인.**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)
![Status](https://img.shields.io/badge/version-v2.1-informational.svg)

---

## 목차
- [개요](#개요)
- [핵심 설계 원칙](#핵심-설계-원칙)
- [시스템 아키텍처](#시스템-아키텍처)
- [지원 데이터셋 · 모델](#지원-데이터셋--모델)
- [Risk Score 융합 전략](#risk-score-융합-전략)
- [기술 스택](#기술-스택)
- [빠른 시작](#빠른-시작)
- [사용법](#사용법)
- [프로젝트 구조](#프로젝트-구조)
- [원격 배포 (ngrok)](#원격-배포-ngrok)

---

## 개요

**HybridPdM**은 단일 모델로는 다루기 어려운 PdM(Predictive Maintenance, 예지보전) 문제를 **세 가지 관점의 딥러닝/머신러닝 모델**로 분해하여 해결합니다.

| 관점 | 모델 | 목적 |
|------|------|------|
| **고장 분류** | 1D-CNN / HistGBDT | "지금 이 상태가 고장인가?" — `P(failure)` |
| **이상 탐지** | Autoencoder | "정상 분포에서 얼마나 벗어났는가?" — `Anomaly Score` |
| **잔존 수명** | LSTM | "앞으로 얼마나 더 쓸 수 있는가?" — `RUL (cycles)` |

세 모델의 출력은 **Weighted Sum / Noisy-OR / Max** 융합을 거쳐 단일 **Risk Score ∈ [0,1]** 로 통합되며, 이를 `Critical · Warning · Advisory · Normal` 4단계 등급으로 매핑합니다.

> 💡 PdM 도메인의 특성상 *False Negative(고장을 정상으로 오판)* 의 비용이 매우 크므로, 보수적 융합 방식인 **Noisy-OR** 을 기본 옵션으로 함께 제공합니다.

---

## 핵심 설계 원칙

- **재현성 우선** — `config.set_seed()` 가 import 시점에 자동 호출되어 python · numpy · torch(CPU/CUDA) 시드를 일괄 고정합니다.
- **데이터셋 격리** — 한 데이터셋의 로드/학습 실패가 다른 데이터셋 파이프라인을 막지 않도록 단계별 try-catch 격리.
- **체크포인트 누적** — `run_id = YYYYMMDD_HHMMSS` 가 모든 산출물에 포함되어 덮어쓰기 없이 누적됩니다.
- **컬럼 검증** — Tabular CNN은 학습 시 사용된 feature 순서를 체크포인트와 함께 저장 → 추론 시 불일치 즉시 감지.
- **해석 가능성** — Captum **Integrated Gradients** 로 피처 중요도 자동 산출 (`--skip-explain` 으로 비활성화 가능).

---

## 시스템 아키텍처

```
┌───────────────────────────────────────────────────────────────────┐
│                    HybridPdM Pipeline (main.py)                   │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐               │
│  │ data_pipe  │──▶│   models   │──▶│   train    │               │
│  │  LOADERS   │    │   build    │    │  TRAINERS  │               │
│  └────────────┘    └────────────┘    └─────┬──────┘               │
│                                            │                      │
│                                            ▼                      │
│                                     ┌─────────────┐               │
│                                     │   evaluate  │               │
│                                     │  EVALUATORS │               │
│                                     └──────┬──────┘               │
│                                            │                      │
│                                            ▼                      │
│                                     ┌────────────┐                │
│                                     │  explain   │ (Captum IG)    │
│                                     └─────┬──────┘                │
└───────────────────────────────────────────┼───────────────────────┘
                                            ▼
                            ┌────────────────────────────┐
                            │   artifacts/checkpoints/   │
                            │   artifacts/reports/*.json │
                            └─────────────┬──────────────┘
                                          │
                  ┌───────────────────────┴──────────────────────┐
                  ▼                                              ▼
          ┌──────────────┐                               ┌──────────────┐
          │  risk_score  │                               │  Streamlit   │
          │ (Weighted /  │ ────────────────────────────▶│  Dashboard   │
          │   Noisy-OR)  │                               │  (app.py)    │
          └──────────────┘                               └──────────────┘
```

---

## 지원 데이터셋 · 모델

| 키 (`--datasets`) | 데이터셋 | 모델 | 태스크 |
|---|---|---|---|
| `ai4i_cnn` | AI4I 2020 | 1D-CNN (tabular) | Binary Classification |
| `ai4i_gbdt` | AI4I 2020 | HistGradientBoosting | Binary Classification |
| `cwru_cnn` | CWRU Bearing | 1D-CNN (raw vibration) | Multi-class Classification |
| `cwru_cnn_stft` ✨ | CWRU Bearing | **2D-CNN (STFT 스펙트로그램)** | Multi-class Classification |
| `hydraulic_ae` | UCI Hydraulic | Autoencoder | Anomaly Detection |
| `cmapss_lstm` | NASA C-MAPSS | BiLSTM | RUL Regression |
| `ncmapss_lstm` | NASA N-CMAPSS | BiLSTM (hidden=256) | RUL Regression |

> AI4I CNN은 극불균형(약 3% 양성) 대응을 위해 **Focal Loss(α=0.85, γ=2.0)** 를 기본 사용합니다.
> N-CMAPSS LSTM은 43개 피처 · 대용량 데이터에 맞춰 **Huber Loss(δ=5.0)** 와 stride/window 슬라이딩을 사용합니다.
> CWRU STFT는 raw 1D 대비 +3~7% 정확도 향상이 일반적입니다 (베어링 결함의 BPFI/BPFO 사이드밴드를 시간-주파수 평면에서 직접 학습).

### 🆕 후처리 — AI4I CNN+GBDT 스태킹 앙상블

`ai4i_cnn`과 `ai4i_gbdt`가 같은 실행에서 모두 성공하면, `main.py` 가 자동으로 **스태킹 앙상블** 평가를 추가합니다 (`ai4i_stack` 엔트리).

- val set에서 `(w_cnn, threshold)` grid를 동시 탐색 → F1-best 채택
- 가중치 grid: `[0.0, 0.1, …, 1.0]` 11개
- 단일 CNN/GBDT 대비 F1 향상 여부가 리포트에 함께 기록됩니다.

```bash
python main.py --datasets ai4i_cnn ai4i_gbdt
# → ai4i_cnn, ai4i_gbdt, ai4i_stack 3개 결과가 리포트에 누적
```

---

## Risk Score 융합 전략

세 모델의 출력 `f = P(failure)`, `a = Anomaly`, `r = RUL_norm` 를 다음 세 가지 중 하나로 융합합니다.

| 방식 | 산식 | 특성 |
|------|------|------|
| **Weighted Sum** | `R = w_f·f + w_a·a + w_r·(1−r)` | 직관적·해석 용이. 한 모델만 발화하면 희석됨 |
| **Noisy-OR** | `R = 1 − Π (1−p_i)^{w_i}` | "어느 한 모델이라도 위험" → FN 최소화 |
| **Max** (기본) | `max(WeightedSum, NoisyOR)` | 보수적, PdM 권장 |

기본 가중치: `failure=0.5, anomaly=0.3, rul=0.2`
등급: `Critical ≥ 0.80 > Warning ≥ 0.50 > Advisory ≥ 0.30 > Normal`

### 🆕 가중치 자동 최적화

세 모델의 출력 + 정답 라벨이 있으면 가중치 grid search로 최적값을 찾을 수 있습니다.

```python
from risk_score import optimize_weights

result = optimize_weights(
    failure_prob=f,    # CNN sigmoid 확률 (N,)
    anomaly_score=a,   # AE anomaly score (N,)
    rul_norm=r,        # RUL / rul_clip (N,)
    y_true=y,          # 0/1 라벨 (N,)
    fusion="max",      # "weighted" | "noisy_or" | "max"
    metric="f1",       # "f1" (threshold 동시 탐색) | "pr_auc"
)
# → {"best_score": 0.87, "best_weights": {...}, "best_threshold": 0.45, ...}
```

- F1 모드: 각 가중치 후보마다 threshold도 grid 탐색 (총 ~1300회 평가)
- PR-AUC 모드: threshold-free 지표 → 더 안정적, 단 운영 시 threshold는 별도 결정

---

## 기술 스택

| 영역 | 라이브러리 |
|------|-----------|
| **딥러닝 코어** | PyTorch ≥ 2.0 |
| **수치/데이터** | NumPy, Pandas, SciPy (.mat), h5py (.h5) |
| **머신러닝 유틸** | scikit-learn (split / scaler / metrics / HistGBDT) |
| **모델 해석** | Captum (Integrated Gradients) |
| **시각화** | Matplotlib, tqdm |
| **대시보드** | Streamlit ≥ 1.30 |
| **원격 배포** | pyngrok ≥ 7.0 |

---

## 빠른 시작

### 1) 환경 준비

```bash
git clone <repo-url>
cd hybrid_pdm
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

### 2) 데이터셋 배치

`dataset/` 폴더 아래에 다음과 같이 배치합니다 (필요한 데이터셋만 받아도 무방 — 누락된 데이터셋은 파이프라인에서 자동 skip).

```
dataset/
├── ai4i2020.csv
├── 10987113/                                                # CWRU
├── condition+monitoring+of+hydraulic+systems/               # UCI Hydraulic
├── CMAPSSData/                                              # C-MAPSS
└── 17. Turbofan Engine Degradation Simulation Data Set 2/
        └── data_set/*.h5                                    # N-CMAPSS
```

### 3) 학습 + 평가 실행

```bash
python main.py                                     # 전체 데이터셋
python main.py --datasets ai4i_cnn cmapss_lstm    # 일부만
python main.py --smoke                            # epochs=1, 동작 검증
python main.py --skip-explain                     # Captum IG 단계 건너뛰기
```

실행 결과 리포트: [artifacts/reports/](artifacts/reports/) · 체크포인트: [artifacts/checkpoints/](artifacts/checkpoints/)

### 4) 대시보드 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속.

---

## 사용법

### CLI 옵션 (`main.py`)

| 옵션 | 설명 |
|------|------|
| `--datasets <key ...>` | 실행할 데이터셋 키 (기본: 전체) |
| `--smoke` | epochs=1 로 짧은 동작 검증 (CI/디버깅용) |
| `--skip-explain` | Captum Integrated Gradients 단계 비활성화 |

### Streamlit 페이지

- **메인 대시보드** ([app.py](app.py)) — 모델 체크포인트 현황, Risk Score 시뮬레이터, 파이프라인 리포트 뷰어
- **1️⃣ Diagnostics** ([pages/1_Diagnostics.py](pages/1_Diagnostics.py)) — 모델별 평가 지표 상세
- **2️⃣ Data Lab** ([pages/2_Data_Lab.py](pages/2_Data_Lab.py)) — 데이터셋 탐색 / 시각화
- **3️⃣ Settings** ([pages/3_Settings.py](pages/3_Settings.py)) — Risk 가중치 / Fusion 방식 변경

---

## 프로젝트 구조

```
hybrid_pdm/
├── app.py                  # Streamlit 메인 대시보드
├── main.py                 # 학습/평가 파이프라인 진입점
├── config.py               # 경로 · 하이퍼파라미터 · 시드
├── data_pipeline.py        # 데이터셋별 로더 (LOADERS)
├── models.py               # 1D-CNN / AE / BiLSTM 빌더
├── train.py                # 태스크별 트레이너 (TRAINERS)
├── evaluate.py             # 태스크별 평가기 (EVALUATORS)
├── explain.py              # Captum IG 기반 해석기 (EXPLAINERS)
├── risk_score.py           # Weighted Sum / Noisy-OR / 등급 매핑
├── run_ngrok.py            # ngrok 원격 배포 헬퍼
├── requirements.txt
├── pages/                  # Streamlit 멀티페이지
│   ├── 1_Diagnostics.py
│   ├── 2_Data_Lab.py
│   └── 3_Settings.py
├── performance analysis/                # 데이터셋별 전처리·학습·평가 노트북
│   ├── README.md
│   ├── ai4i_analysis.ipynb
│   ├── hydraulic_analysis.ipynb
│   ├── cwru_analysis.ipynb           # raw 1D vs STFT 2D 비교
│   ├── cmapss_analysis.ipynb
│   └── ncmapss_analysis.ipynb        # MSE vs Huber 손실 비교
├── dataset/                # (gitignore) 원본 데이터셋
└── artifacts/              # 학습 산출물
    ├── checkpoints/        # *.pt / *.pkl
    └── reports/            # pipeline_report_<run_id>.json
```

---

## 원격 배포 (ngrok)

Colab/원격 서버에서 대시보드를 외부에 공개해야 할 때:

```bash
# 토큰 발급: https://dashboard.ngrok.com/get-started/your-authtoken

# 방법 1: CLI 인자
python run_ngrok.py --token <YOUR_TOKEN>

# 방법 2: 환경변수
set NGROK_TOKEN=<YOUR_TOKEN>     # Windows
python run_ngrok.py

# 방법 3: .env 파일에 NGROK_TOKEN=... 한 줄 작성
python run_ngrok.py
```

- 포트는 `8501~8600` 범위에서 자동 탐색됩니다 (`--port` 로 지정 가능).
- 기본 리전은 `jp` (한국에 인접). `--region` 으로 변경 가능.

---

## 참고

- 본 프로젝트는 학술/POC 목적의 구현이며, 사용된 데이터셋의 라이선스는 각 제공처(UCI, NASA, CWRU)의 정책을 따릅니다.
- 버전: **HybridPdM v2.1** — Streamlit POC
