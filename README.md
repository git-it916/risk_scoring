# CISS Risk Scoring (Korea)

이 프로젝트는 Bloomberg 시계열 데이터를 받아, 15개 스트레스 지표를 만든 뒤,  
5개 분야(Money/Bond/Equity/FX/Financial Intermediaries)를 합쳐 최종 CISS를 계산합니다.

핵심 질문 3개에 맞춰 문서를 구성했습니다.
1. 어떤 데이터를 쓰는가?
2. 그 데이터를 어떻게 가공하는가?
3. 5개 분야를 어떻게 하나의 점수로 합치는가?

## 1) 어떤 데이터를 쓰는가

원천 데이터는 `src/data_loader.py`에서 Bloomberg BDH로 수집합니다.

- 가격/금리 데이터: `PX_LAST` 12개
- 거래량 데이터: `PX_VOLUME` 1개
- 총 13개 raw 컬럼

| Raw 컬럼명 | Bloomberg 티커/필드 | 주 사용 지표 |
|---|---|---|
| `KWCDC_Curncy` | `KWCDC Curncy / PX_LAST` | MM1, MM2, MM3, BD2 |
| `GVSK3M_Index` | `GVSK3M Index / PX_LAST` | MM2 |
| `GVSK3YR_Index` | `GVSK3YR Index / PX_LAST` | BD1, BD2 |
| `GVSK10YR_Index` | `GVSK10YR Index / PX_LAST` | BD1 |
| `MOVE_Index` | `MOVE Index / PX_LAST` | BD3 |
| `KOSPI_Index` | `KOSPI Index / PX_LAST` | EQ1, EQ3, FI3 |
| `KOSPI_Index_VOLUME` | `KOSPI Index / PX_VOLUME` | EQ3 |
| `VKOSPI_Index` | `VKOSPI Index / PX_LAST` | EQ2 |
| `USDKRW_Curncy` | `USDKRW Curncy / PX_LAST` | FX1 |
| `USDKRWV1M_BGN_Curncy` | `USDKRWV1M BGN Curncy / PX_LAST` | FX2 |
| `KWSWNI1_Curncy` | `KWSWNI1 Curncy / PX_LAST` | FX3 |
| `CKREA1U5_CBGN_Curncy` | `CKREA1U5 CBGN Curncy / PX_LAST` | FI1 |
| `KOSPFIN_Index` | `KOSPFIN Index / PX_LAST` | FI2, FI3 |

추가 참고:
- `xbbg`를 import하지 못하면 `mock data`로 대체됩니다(테스트용).
- `load_raw_data()`는 일별과 주별(`W-FRI`) 데이터를 모두 준비합니다.

## 2) 데이터를 어떻게 가공하는가

가공 코드는 `src/transforms.py`에 있습니다.

### 2-1. 전처리

1. 날짜 정렬 및 병합: 로더에서 모든 티커를 날짜 인덱스로 정렬해 하나의 DataFrame으로 만듭니다.  
2. 결측 보정: `ffill()`로 직전값을 채웁니다.  
3. 지표 계산 후 `dropna()`: 수익률/롤링 윈도우 때문에 생기는 초반 결측 구간을 제거합니다.

### 2-2. 13개 raw -> 15개 지표

아래 15개가 실제 CISS 입력 지표입니다.

Money Market
- `MM1 = KWCDC_Curncy` (91일 CD 금리의 절대 수준)
- `MM2 = KWCDC_Curncy - GVSK3M_Index` (CD 금리와 국고채 3개월 금리의 차이)
- `MM3 = rolling_std(diff(KWCDC_Curncy), 20) * sqrt(252)` (CD 금리 변화의 20일 연율화 변동성)

Bond Market
- `BD1 = -(GVSK10YR_Index - GVSK3YR_Index)` (국고채 10년-3년 스프레드를 반전한 값)  
  `10Y-3Y` 스프레드가 줄거나 역전될수록 스트레스가 높다고 보고 부호를 반전합니다.
- `BD2 = KWCDC_Curncy - GVSK3YR_Index` (CD 금리와 국고채 3년 금리의 차이)
- `BD3 = MOVE_Index` (채권시장 변동성 지수 수준)

Equity Market
- `EQ1 = -pct_change(KOSPI_Index)` (코스피 수익률을 반전한 값, 하락일수록 스트레스)  
  주가 하락이 스트레스이므로 수익률 부호를 반전합니다.
- `EQ2 = VKOSPI_Index` (코스피 내재변동성 수준)
- `EQ3 = rolling_mean(abs(pct_change(KOSPI_Index)) / KOSPI_Index_VOLUME * 1e12, 20)` (아미후드 비유동성의 20일 평균)

FX Market
- `FX1 = pct_change(USDKRW_Curncy)` (원/달러 환율 수익률, 원화 약세 방향이 스트레스)
- `FX2 = USDKRWV1M_BGN_Curncy` (원/달러 1개월 내재변동성 수준)
- `FX3 = -KWSWNI1_Curncy` (1년 CRS 수준을 반전한 값, 낮아질수록 스트레스)

Financial Intermediaries
- `FI1 = CKREA1U5_CBGN_Curncy` (한국 5년 CDS 프리미엄 수준)
- `FI2 = rolling_std(pct_change(KOSPFIN_Index), 20) * sqrt(252) * 100` (금융업종 수익률의 20일 연율화 변동성)
- `FI3 = -(pct_change(KOSPFIN_Index) - pct_change(KOSPI_Index))` (금융업종 상대수익률 반전값)  
  금융업종 언더퍼폼을 스트레스로 보아 부호를 반전합니다.

### 2-3. ECDF 정규화

각 지표를 ECDF로 `[0, 1]` 범위로 변환합니다.

`ECDF_i(x) = (# of sample values <= x) / N`

- 값이 1에 가까울수록 "역사적으로 높은 스트레스"
- 값이 0에 가까울수록 "역사적으로 낮은 스트레스"

코드상 `fit_transform`은 현재 데이터 구간 전체를 샘플로 사용합니다.

## 3) 5개 분야를 어떻게 합치는가

통합 코드는 `src/ciss_calculator.py`에 있습니다.

### 3-1. 분야 구성

- 분야 수: 5개
- 각 분야 지표 수: 3개
- 분야 가중치: 모두 `0.20`

즉 지표 단위 가중치는 기본적으로 `0.20 / 3`이며, 총합이 1이 되도록 정규화합니다.

### 3-2. 상관효과 계산

상관행렬 `C_t`는 `src/dcc_garch.py`에서 계산합니다.

- `ewma`: 기본 대안
- `dcc`: `arch` 패키지 필요

상관효과:

`Correlation_Effect_t = sqrt(w' * C_t * w)`

### 3-3. 최종 CISS

ECDF 지표 벡터를 `s_t`라고 하면:

`CISS_t = (w' * s_t) * Correlation_Effect_t`

즉,
1. 지표들의 가중 평균 스트레스 `w' * s_t`를 만들고
2. 동시변동(상관) 정도 `sqrt(w' * C_t * w)`로 증폭/완화합니다.

### 3-4. 5개 분야 기여도

분야 `k`의 기여도는:

`Contribution_{k,t} = Correlation_Effect_t * sum_{i in k}(w_i * s_{i,t})`

현재 구현은 아래를 만족합니다.

`sum(5개 분야 기여도) == CISS`

## 4) 실제 코드 실행 흐름

`src/main.py`의 `CISSPipeline.run()` 기준:

1. `load_raw_data()` 호출  
2. `compute_indicators()` 호출  
3. `compute_dynamic_correlations()` 호출  
4. `compute_ciss_score()` 호출  
5. 결과 저장 (`src/output/*.csv`)

## 5) 실행 방법

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cd src
python main.py
```

## 6) 출력 파일

- `src/output/raw_indicators.csv`: 가공된 15개 raw 지표
- `src/output/ecdf_indicators.csv`: ECDF 정규화 지표
- `src/output/ciss_results.csv`: `CISS`, `Correlation_Effect`, 5개 분야 기여도
- `src/output/historical_ciss.csv`: CISS 히스토리
