# 성능분석 노트북 모음

HybridPdM의 6개 데이터셋·모델별로 **데이터 전처리부터 모델 성능 분석까지** 전 과정을 시각화한 Jupyter 노트북.

각 노트북은 다음 공통 구조를 따른다:

1. **데이터 로드 & 개요** — 샘플 수, 라벨 분포, 결함 유형
2. **데이터 전처리 시각화** — 클래스 분포 / 상관관계 / 센서별 분포
3. **학습/검증/테스트 분할** — leakage 방지 전략 명시
4. **모델 정의 & 학습** — HybridPdM 본체와 동일 구조 재현
5. **학습 곡선** — Train/Val loss, gap, 과적합 진단
6. **성능 평가** — Task별 표준 메트릭 + 혼동 행렬 / ROC / Pred-vs-Actual
7. **Task-specific 심화 분석** — 결함 유형별·구간별·임계값별
8. **종합 요약**

---

## 노트북 목록

| 노트북 | 데이터셋 | 모델 / 태스크 | 핵심 분석 |
|--------|---------|--------------|----------|
| [ai4i_analysis.ipynb](ai4i_analysis.ipynb) | AI4I 2020 | DenoisingAE / 이상 탐지 | 9가지 증강 × 임계값 실험, 하이퍼파라미터 4단계 튜닝 비교, 고장 유형별(TWF/HDF/PWF/OSF/RNF) 탐지율 |
| [hydraulic_analysis.ipynb](hydraulic_analysis.ipynb) | UCI Hydraulic | DenoisingAE / 이상 탐지 | 17개 센서 그룹별 분포, percentile grid 임계값 탐색, 센서별 재구성 오차 기여도 Top-5 |
| [cwru_analysis.ipynb](cwru_analysis.ipynb) | CWRU Bearing | WDCNN1D vs STFTCNN2D / 다중 분류 | raw 1D vs STFT 2D 성능 비교, FFT 스펙트럼, 파일 단위 split (leakage 방지) |
| [cmapss_analysis.ipynb](cmapss_analysis.ipynb) | NASA C-MAPSS FD001 | BiLSTM+Attention / RUL 회귀 | 14개 유효 센서 선별, piecewise-linear RUL(clip=125), Early/Mid/Late 구간별 정확도 |
| [ncmapss_analysis.ipynb](ncmapss_analysis.ipynb) | NASA N-CMAPSS DS01 | BiLSTM(hidden=256) + Huber Loss / RUL 회귀 | 43 피처 stride 다운샘플링, RUL 정규화 [0,1], **MSE vs Huber 손실 비교** |

---

## 실행 방법

```bash
# 프로젝트 루트에서
.venv\Scripts\activate
jupyter notebook 성능분석/
```

또는 VS Code에서 직접 `.ipynb` 파일 열기.

### 경로 규약
- 모든 노트북은 `성능분석/` 폴더 안에서 실행되는 것을 가정 → 데이터 경로는 `../dataset/...` 상대 경로.
- 한글 폰트는 `Malgun Gothic`(Windows). macOS/Linux는 `AppleGothic` 또는 `NanumGothic`으로 변경.

---

## HybridPdM 본체 코드와의 관계

각 노트북은 본체 모듈([models.py](../models.py), [data_pipeline.py](../data_pipeline.py), [train.py](../train.py))의 **핵심 로직을 단일 노트북에 재현**하여 단계별 시각화가 가능하도록 만든 것이다.

학습된 체크포인트를 본체와 공유하지는 않으며, 노트북 내부에서 독립적으로 학습 → 평가 → 시각화한다.

본체에서 전체 파이프라인을 한 번에 돌리려면:
```bash
python main.py --datasets cwru_cnn cwru_cnn_stft hydraulic_ae cmapss_lstm ncmapss_lstm
```
