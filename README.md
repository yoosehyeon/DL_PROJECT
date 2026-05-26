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
- [수치 읽는 법](#수치-읽는-법)
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

> PdM 도메인의 특성상 *False Negative(고장을 정상으로 오판)* 의 비용이 매우 크므로, 보수적 융합 방식인 **Noisy-OR** 을 기본 옵션으로 함께 제공합니다.

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
| `cwru_cnn_stft`  | CWRU Bearing | **2D-CNN (STFT 스펙트로그램)** | Multi-class Classification |
| `hydraulic_ae` | UCI Hydraulic | Autoencoder | Anomaly Detection |
| `cmapss_lstm` | NASA C-MAPSS | BiLSTM | RUL Regression |
| `ncmapss_lstm` | NASA N-CMAPSS | BiLSTM (hidden=256) | RUL Regression |

> AI4I CNN은 극불균형(약 3% 양성) 대응을 위해 **Focal Loss(α=0.85, γ=2.0)** 를 기본 사용합니다.
> N-CMAPSS LSTM은 43개 피처 · 대용량 데이터에 맞춰 **Huber Loss(δ=5.0)** 와 stride/window 슬라이딩을 사용합니다.
> CWRU STFT는 raw 1D 대비 +3~7% 정확도 향상이 일반적입니다 (베어링 결함의 BPFI/BPFO 사이드밴드를 시간-주파수 평면에서 직접 학습).

### 후처리 — AI4I CNN+GBDT 스태킹 앙상블

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

### 가중치 자동 최적화

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

## 수치 읽는 법

리포트와 노트북에 등장하는 숫자들이 **실제로 무엇을 의미하는지** 일상 비유로 정리합니다.

### 1) 공통 개념: 두 가지 문제 유형

이 시스템에는 **성격이 다른 두 종류의 예측 문제**가 섞여 있습니다.

| 유형 | 비유 | 출력 | 대표 데이터셋 |
|------|------|------|--------------|
| **분류 (Classification)** | "이 환자, 감기인가요?" (예/아니오) | 0 또는 1 | AI4I, CWRU, Hydraulic |
| **회귀 (Regression)** | "이 자동차, 몇 km 더 탈 수 있어요?" (숫자) | 실수 (cycle 단위) | C-MAPSS, N-CMAPSS |

→ 유형에 따라 봐야 할 지표가 다릅니다.

---

### 2) 분류 문제 지표 (AI4I, CWRU, Hydraulic)

장비 1만 대 중 **고장 100대**가 섞여 있다고 가정.
모델이 "80대를 고장"이라고 예측했고, 그중 **60대만 진짜 고장**이었다고 합시다.

| 지표 | 산식·의미 | 위 예시 값 | 직관적 해석 |
|------|----------|-----------|------------|
| **Accuracy (정확도)** | 전체 중 맞춘 비율 | 약 99% | "100명 중 99명 맞췄어요" — 단, **불균형 시 함정 있음**. 모두 정상이라 찍어도 99%가 나옴 |
| **Precision (정밀도)** | 모델이 "고장"이라 한 것 중 진짜 비율 | 60/80 = **75%** | "모델이 고장이라 하면 4번 중 3번은 진짜다" — 거짓 경보(False Alarm) 적음 |
| **Recall (재현율)** | 진짜 고장 중 모델이 잡아낸 비율 | 60/100 = **60%** | "10대 고장 중 6대를 잡았다" — **놓치면 안 되는 PdM에서 가장 중요** |
| **F1-Score** | Precision과 Recall의 조화평균 | 약 **0.67** | 둘 다 균형 있게 잘하나? (한쪽만 높으면 점수 낮아짐) |
| **AUC** | 0.5(동전던지기) ~ 1.0(완벽) | — | "랭킹이 잘 매겨졌나"의 척도. **threshold와 무관** |

> **PdM에서 가장 중요한 건 Recall.**
> 고장을 놓치면(FN) 장비가 멈추는 손실이 거짓 경보보다 훨씬 큽니다.
> 그래서 본 시스템은 보수적인 Noisy-OR 융합을 기본으로 둡니다.

### 혼동 행렬 (Confusion Matrix) 읽기

|  | 예측: 정상 | 예측: 고장 |
|--|----------|----------|
| **실제: 정상** | TN (잘 맞춤) | **FP** (거짓 경보) — 멀쩡한데 점검 보내서 비용 낭비 |
| **실제: 고장** | **FN** (놓침) — 가장 위험! | TP (잘 잡음) |

- **FN(False Negative)이 늘면 위험합니다** — 고장이 났는데 모델이 정상이라 했다는 뜻.
- FP가 늘면 "양치기 소년" 문제가 생겨 운영자가 알람을 무시하기 시작합니다.

### 데이터셋별 "이 정도면 잘한 것" 기준

| 데이터셋 | 어려움 | 좋은 성능 기준 | 이유 |
|---------|-------|---------------|------|
| **AI4I** | 고장이 전체의 3%뿐 (심한 불균형) | F1 > 0.3, Recall > 0.5 | 학계 벤치마크에서도 0.4 넘기기 어려움 |
| **CWRU** | 3개 결함 유형 다중 분류 | Accuracy > 0.90, Macro-F1 > 0.85 | 신호가 깨끗해서 쉬운 편 |
| **Hydraulic** | 17개 센서 모두 봐야 함 | F1 > 0.85, AUC > 0.90 | 라벨이 비교적 균형 |

---

### 3) 회귀 문제 지표 (C-MAPSS, N-CMAPSS)

"엔진이 앞으로 **몇 cycle 더 쓸 수 있는가**" 를 맞추는 문제.
실제 정답이 50 cycle인데 모델이 45 cycle이라 했다면 **오차 5 cycle**.

| 지표 | 의미 | 단위 | 좋은 값 |
|------|------|------|---------|
| **MAE** (Mean Absolute Error) | 평균 절대 오차 | cycle | **0에 가까울수록 좋음** |
| **RMSE** (Root Mean Squared Error) | 평균 제곱 오차의 제곱근 | cycle | MAE와 같이 봄. **큰 오차에 더 큰 페널티** |
| **R²** (결정계수) | 모델이 분산을 얼마나 설명? | 0~1 (음수도 가능) | 1에 가까울수록 좋음. **0.7 이상이면 양호**, 0 이하면 평균만 찍은 것보다 못함 |

> RMSE > MAE 는 항상 성립합니다. **차이가 클수록 가끔 크게 빗나가는 경우가 있다**는 뜻.

### 예시 (C-MAPSS FD001 기준 일반적인 값)

```
RMSE = 18.5 cycle    → "평균 18~19 cycle 정도 오차로 RUL 예측"
MAE  = 14.2 cycle    → "절대값 평균 14 cycle 차이"
R²   = 0.72          → "RUL 변동의 72%를 설명함"
```

| 데이터셋 | 좋은 성능 기준 | 이유 |
|---------|---------------|------|
| **C-MAPSS** | RMSE < 20, MAE < 15 | 학계 SOTA가 RMSE 12~13 수준 |
| **N-CMAPSS** | RMSE < 25 | 더 어려운 데이터셋 (실제 비행 조건) |

### 구간별 정확도가 더 중요

전체 RMSE 하나만 보면 안 됩니다. **수명 임박 구간(RUL 작을 때)** 의 정확도가 PdM 핵심입니다.

```
Late  (RUL ≤ 40):  RMSE 12.3   ← 가장 중요!
Mid   (40~90):     RMSE 18.5
Early (>90):       RMSE 22.1   ← 어차피 멀쩡할 때라 덜 중요
```

수명이 임박했을 때 정확하면 → 정비 일정을 정확히 잡을 수 있어 비용 절감.

---

### 4) Risk Score 등급 읽기 (대시보드)

세 모델의 출력을 융합한 **단일 점수**입니다.

| 등급 | 점수 범위 | 비유 | 운영자 액션 |
|------|----------|------|------------|
| **Critical** | ≥ 0.80 | 응급실 직행 환자 | 즉시 정비, 가동 중단 검토 |
| **Warning** | 0.50 ~ 0.80 | 병원 예약하세요 | 다음 정비 일정에 우선 배치 |
| **Advisory** | 0.30 ~ 0.50 | 비타민 챙기세요 | 모니터링 강화 |
| **Normal** | < 0.30 | 건강함 | 정기 점검만 유지 |

### 어느 모델이 위험 신호를 보냈는지 확인하기

대시보드 "Risk Score 시뮬레이터" 에서 세 슬라이더(`P(failure)`, `Anomaly`, `RUL`)를 바꿔보면 각 모델의 기여도가 보입니다.

- **CNN만 0.9**, 나머지 0 → 평소와 다른 *조건*에서 작동 중 (직접적 고장 신호)
- **AE만 0.9**, 나머지 0 → 평소와 다른 *패턴*이 감지됨 (이상 작동)
- **LSTM RUL이 0.1** → 잔존 수명 거의 소진

세 신호가 함께 올라가면 → **고장이 정말 임박했다**는 강한 증거입니다.

---

### 5) 학습 그래프(노트북) 읽는 법

#### Loss 곡선

```
Train Loss ↘ Val Loss ↘     → 정상적으로 학습 중
Train Loss ↘ Val Loss ↗     → 과적합 시작! Early Stop 발동
두 선이 가까이 붙어서 평평   → 학습 완료, 일반화 잘됨
```

#### 재구성 오차 분포 (AE)

```
정상 샘플 ──┐
           └→ 작은 MSE에 몰림
이상 샘플 ──┐
           └→ 큰 MSE 쪽으로 꼬리가 길게 분리됨   ← 두 분포가 잘 분리될수록 좋음
```

두 분포가 완전히 겹쳐 보인다면 → 모델이 정상/이상 구분을 못 한다는 뜻.

#### Pred vs Actual 산점도 (RUL)

```
점들이 대각선(y=x)에 가까이 모일수록 좋음.
대각선 위쪽으로 치우침 → 모델이 RUL을 과대평가 (위험!)
대각선 아래로 치우침   → 과소평가 (불필요한 정비)
```

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
├── ui_common.py            # 프론트엔드 공용 헬퍼 (intent 박스, 용어집)
├── pages/                  # Streamlit 멀티페이지
│   ├── 1_모델_진단.py        # 모델 추론·예측 분포·영향 센서(XAI)
│   ├── 2_데이터_살펴보기.py  # 내장 데이터셋·CSV 업로드 탐색
│   └── 3_설정.py             # 위험도 설정·시스템 상태·모델 파일 목록
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
