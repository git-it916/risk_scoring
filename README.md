# CISS Risk Scoring (Korea)

한국 금융시장 기반 CISS(Composite Indicator of Systemic Stress) 리스크 스코어링 시스템

## Overview

ECB의 CISS 방법론을 한국 시장에 적용하여 시스템 리스크를 측정합니다.

- **데이터 소스**: Bloomberg Terminal
- **주기**: Weekly (W-FRI)
- **기간**: 2024-01-01 ~ 현재

---

## 지표 구성 (5개 섹터 × 3개 = 15개)

### 1. 단기금융시장 (Money Market)

| ID | 지표명 | 설명 | Bloomberg 티커 | 스트레스 방향 |
|----|--------|------|----------------|--------------|
| MM1 | CD금리 수준 | 91일 CD 금리 | `KWCDC Curncy` | 상승 ↑ |
| MM2 | CD-국고채 스프레드 | CD금리 - 국고채 3개월 금리차 | `KWCDC` - `GVSK3M` | 확대 ↑ |
| MM3 | CD금리 변동성 | CD금리 20일 실현변동성 | `KWCDC Curncy` | 상승 ↑ |

### 2. 채권시장 (Bond Market)

| ID | 지표명 | 설명 | Bloomberg 티커 | 스트레스 방향 |
|----|--------|------|----------------|--------------|
| BD1 | 장단기 금리차 | 국고채 10년 - 3년 스프레드 | `GVSK10YR` - `GVSK3YR` | 축소/역전 ↓ |
| BD2 | 신용 스프레드 | CD금리 - 국고채 3년 금리차 | `KWCDC` - `GVSK3YR` | 확대 ↑ |
| BD3 | 채권 변동성 | MOVE 지수 (글로벌 채권 변동성) | `MOVE Index` | 상승 ↑ |

### 3. 주식시장 (Equity Market)

| ID | 지표명 | 설명 | Bloomberg 티커 | 스트레스 방향 |
|----|--------|------|----------------|--------------|
| EQ1 | 코스피 수익률 | 코스피 주간 수익률 | `KOSPI Index` | 하락 ↓ |
| EQ2 | 코스피 변동성 | VKOSPI (코스피200 내재변동성) | `VKOSPI Index` | 상승 ↑ |
| EQ3 | 시장 비유동성 | Amihud 비유동성 비율 | `KOSPI Index` (가격, 거래량) | 상승 ↑ |

### 4. 외환시장 (FX Market)

| ID | 지표명 | 설명 | Bloomberg 티커 | 스트레스 방향 |
|----|--------|------|----------------|--------------|
| FX1 | 원/달러 수익률 | USD/KRW 주간 변화율 | `USDKRW Curncy` | 상승 (원화약세) ↑ |
| FX2 | 환율 변동성 | 원/달러 1개월 내재변동성 | `USDKRWV1M BGN Curncy` | 상승 ↑ |
| FX3 | CRS 베이시스 | 1년 통화스왑 금리 (달러 조달비용) | `KWSWNI1 Curncy` | 하락 ↓ |

### 5. 금융중개기관 (Financial Intermediaries)

| ID | 지표명 | 설명 | Bloomberg 티커 | 스트레스 방향 |
|----|--------|------|----------------|--------------|
| FI1 | 국가 CDS | 한국 5년 CDS 스프레드 | `CKREA1U5 CBGN Curncy` | 상승 ↑ |
| FI2 | 금융업종 변동성 | 금융업종 지수 20일 실현변동성 | `KOSPFIN Index` | 상승 ↑ |
| FI3 | 금융업종 상대성과 | 금융업종 수익률 - 코스피 수익률 | `KOSPFIN` - `KOSPI` | 언더퍼폼 ↓ |

---

## 계산 방식

### 1단계: 데이터 수집
```
Bloomberg BDH API → 일별 원본 데이터 → 주간(금요일 기준) 리샘플링
```

### 2단계: 지표 변환

각 원본 데이터를 스트레스 지표로 변환:

| 변환 유형 | 공식 | 설명 |
|----------|------|------|
| 수준 (level) | 원본값 그대로 | 값이 클수록 스트레스 |
| 스프레드 (spread) | A - B | 스프레드 확대 = 스트레스 |
| 수익률 (return) | (P_t / P_{t-1}) - 1 | 하락 = 스트레스 (반전 적용) |
| 실현변동성 | 20일 표준편차 × √252 | 변동성 증가 = 스트레스 |
| 비유동성 | \|수익률\| / 거래량 | 비유동성 증가 = 스트레스 |

### 3단계: ECDF 정규화

각 지표를 [0, 1] 범위로 변환:

```
ECDF(x) = (x보다 작거나 같은 관측치 수) / (전체 관측치 수)
```

- **0에 가까움** = 역사적으로 낮은 스트레스 수준
- **1에 가까움** = 역사적으로 높은 스트레스 수준

### 4단계: 동적 상관행렬 추정

EWMA(지수가중이동평균) 방식으로 시변 상관행렬 추정:

```
공분산_t = λ × 공분산_{t-1} + (1-λ) × (수익률_t × 수익률_t')
```
- λ = 0.94 (감쇠계수)

### 5단계: CISS 스코어 계산

**CISS 공식:**

```
CISS_t = (가중평균 스트레스) × (상관 증폭 효과)
       = (w' × s_t) × √(w' × C_t × w)
```

- `s_t`: ECDF 변환된 15개 지표 벡터
- `w`: 섹터 가중치 (각 20%)
- `C_t`: 시점 t의 상관행렬
- `√(w' × C_t × w)`: **상관 증폭 효과**

**상관 증폭 효과 해석:**
- 지표들 간 상관이 높을수록 → CISS 값 증가
- 위기 시 여러 시장이 동시에 스트레스를 받으면 시스템 리스크 증폭

---

## 프로젝트 구조

```
risk_scoring/
├── config/
│   └── indicator_spec.yaml    # 지표 설정
├── src/
│   ├── data_loader.py         # Bloomberg 데이터 수집
│   ├── transforms.py          # 지표 변환 & ECDF
│   ├── dcc_garch.py           # 동적 상관행렬
│   ├── ciss_calculator.py     # CISS 계산
│   └── main.py                # 메인 파이프라인
├── scripts/
│   └── data_validation.py     # 데이터 검증
├── output/                    # 결과 저장
└── README.md
```

---

## 사용법

### 설치

```bash
pip install xbbg pandas numpy scipy arch
```

### 실행

```python
from src.main import CISSPipeline

# 파이프라인 생성
pipeline = CISSPipeline(
    start_date='2024-01-01',
    dcc_method='ewma'  # 또는 'dcc'
)

# 실행
ciss_result = pipeline.run()

# 요약 출력
pipeline.print_summary()

# 결과 저장
pipeline.save_results('output')
```

### 커맨드라인

```bash
cd src
python main.py
```

---

## 출력 예시

```
==============================================================
 CISS SUMMARY
==============================================================

 Date: 2026-01-24
 CISS Score: 0.4523
 Correlation Effect: 0.8912

 Sector Contributions:
   Money_Market                  : 0.0821 ████
   Bond_Market                   : 0.0934 ████
   Equity_Market                 : 0.1102 █████
   FX_Market                     : 0.0876 ████
   Financial_Intermediaries      : 0.0790 ███

 Statistics (Full Period):
   Mean:   0.4215
   Std:    0.0892
   Min:    0.2134
   Max:    0.6521
   Latest: 0.4523

 Risk Level: MODERATE
==============================================================
```

---

## 리스크 레벨 해석

| CISS 범위 | 레벨 | 해석 | 권장 조치 |
|-----------|------|------|----------|
| 0.0 - 0.3 | 낮음 (LOW) | 정상적인 시장 상황 | 일반적인 포지션 유지 |
| 0.3 - 0.5 | 보통 (MODERATE) | 다소 상승된 스트레스 | 모니터링 강화 |
| 0.5 - 0.7 | 경계 (ELEVATED) | 주의 필요 | 리스크 축소 검토 |
| 0.7 - 1.0 | 위험 (HIGH) | 시스템 리스크 경고 | 방어적 포지션 전환 |

---

## 참고 문헌

- Holló, D., Kremer, M., & Lo Duca, M. (2012). "CISS - A Composite Indicator of Systemic Stress in the Financial System." ECB Working Paper No. 1426.
- Engle, R. (2002). "Dynamic Conditional Correlation: A Simple Class of Multivariate GARCH Models." Journal of Business & Economic Statistics.

---

## License

MIT License
