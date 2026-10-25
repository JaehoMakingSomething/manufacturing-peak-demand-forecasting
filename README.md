# Manufacturing Peak Demand Forecasting
## 제조 생산데이터 기반 전력 피크 예측

제조 공장의 생산·전력·기상 데이터를 이용해 **시간별 15분 최대수요전력(kW)**을 예측하는 K제조 공모전 프로젝트입니다. 전날의 생산계획 검토와 당일의 피크 위험 재평가를 서로 다른 예측 시점으로 정의했습니다.

**핵심 질문:** 전체 예측 오차를 관리하면서, 고부하 시간의 오차와 과소예측을 줄일 수 있는가?

| 항목 | 내용 |
| --- | --- |
| 예측 대상 | 각 1시간 구간 안의 15분 수요전력 최대값, kW |
| A · 전날 예측 | 전날 23:00에 다음 날 00–23시의 24개 시간 구간 예측 |
| B · 당일 예측 | 매 정각에 해당 시각부터 1시간 구간 예측 |
| 비교 모델 | LightGBM, Random Forest |
| 고도화 후보 | 0.60/0.65 분위수 회귀, 피크 구간 표본 가중치 |
| 피처 선택 구간 | 2021-07-05–2021-08-29 |
| 최종 확인 구간 | 2021-08-30–2021-09-14 |
| 주요 지표 | MAE, 피크 MAE, 피크 과소예측, Peak Recall |

### 연구에서 다룬 문제

- **데이터 품질:** 하루 96개 전력 측정값이 완전히 같은 복사일을 찾아 학습에서 제외했습니다. 원본에 저장된 실행 로그에는 115일이 탐지되어 있습니다.
- **정보의 시점:** A와 B를 나눠 과거 전력 피처의 사용 가능 범위를 다르게 두고, 미래 전력을 변조하는 누수 검사 함수를 구현했습니다.
- **피크 중심 평가:** 실제 수요가 180kW 이상인 시간의 오차와 과소예측을 전체 MAE와 함께 평가합니다.
- **모델 비교:** 동일한 최종 피처 세트에서 LightGBM과 Random Forest를 비교하고, 분위수·가중치 고도화를 별도 실험으로 둡니다.
- **현장 연결:** 예측한 위험 시간대를 생산 일정 검토에 활용합니다. 작업 이동에 따른 실제 절감액이나 인과효과는 검증하지 않았습니다.

### 원본에 저장된 기존 결과

아래는 원본 노트북의 저장 출력에서 옮긴 **과거 결과**입니다. 이번 공개용 정리 과정에서 재학습하지 않았고, 현재 코드로 재현된 수치로 해석하지 않습니다. 각 모델의 평가 행은 383개, 실제 피크 행은 56개입니다.

| 시점 | 모델 | MAE (kW) | 피크 MAE (kW) | 피크 과소예측 평균 (kW) |
| --- | --- | ---: | ---: | ---: |
| A | LightGBM | 7.881 | 7.593 | 6.782 |
| A | Random Forest | 7.497 | 7.578 | 7.075 |
| B | LightGBM | 6.077 | 7.265 | 6.003 |
| B | Random Forest | 6.558 | 8.032 | 6.985 |

고도화 후보 `LGB_Q65`와 `LGB_PeakW_Q65`의 최종 성능은 원본에 저장되어 있지 않습니다. 위 표를 해당 후보의 성과로 사용하지 않습니다. 결과의 출처와 해석은 [결과 설명](docs/results.md)을 참고하세요.

### 살펴볼 순서

1. [분석 노트북](notebooks/01_peak_demand_forecasting.ipynb): 예측 시나리오, 피처, 실험 흐름.
2. [모델 코드](src/peak_forecasting.py): 전처리, 후보 피처, 주 단위 평가, 피크 지표.
3. [방법론과 한계](docs/methodology.md): 발행 시점, 검증 구간, 가정과 확인 과제.
4. [결과 설명](docs/results.md): 저장 결과의 의미와 추가 검증 범위.

### 실행

Python 3.10 이상에서 프로젝트 루트를 기준으로 실행합니다. 원래 학습 환경의 정확한 패키지 버전은 기록되지 않아 `requirements.txt`는 설치 목록이며 환경 잠금 파일이 아닙니다.

```bash
python -m pip install -r requirements.txt
```

전처리된 `okm_전처리완료_2021.xlsx`를 `data/`에 놓으세요. 데이터의 시트와 열 설명은 [입력 데이터 안내](data/README.md)에 있습니다. 원본 데이터, 학습 가중치, 개별 예측 CSV는 포함되어 있지 않습니다.

```bash
# 원본에 저장된 과거 성능표 확인: 데이터 없이 실행 가능
python scripts/run_experiment.py historical

# 피처 생성 및 필수 열 확인
python scripts/run_experiment.py inspect

# 선택 구간에서 기준 모델 비교
python scripts/run_experiment.py select

# 최종 14개 피처의 미래 전력 누수 검사
python scripts/run_experiment.py leak-check

# 선택된 고도화 후보끼리 피처 조합을 비교: 선택 구간만 사용
python scripts/run_experiment.py combos

# 모델과 피처를 동결한 뒤에만 최종 구간 평가
python scripts/run_experiment.py final --frozen
```

`final`은 A의 `LGB_Q65`, B의 `LGB_PeakW_Q65` 및 각 기준 모델을 평가합니다. 조합 실험의 권장 피처를 자동으로 적용하지 않습니다. 최종 구간을 다시 보며 피처를 선택하면 선택 편향이 생깁니다.

```bash
python -m unittest discover -s tests -v
jupyter lab notebooks/01_peak_demand_forecasting.ipynb
```

