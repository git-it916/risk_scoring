# -*- coding: utf-8 -*-
"""
CISS Risk Scoring - CISS Calculator Module
============================================
Composite Indicator of Systemic Stress 계산
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class CISSCalculator:
    """
    CISS (Composite Indicator of Systemic Stress) 계산

    CISS 공식 (ECB 원본):
        CISS_t = (s_t' * w) * (w' * C_t * w)

    여기서:
        - s_t: ECDF 변환된 지표 벡터 (15 x 1)
        - w: 섹터 가중치 벡터
        - C_t: 시변 상관행렬 (DCC-GARCH)

    상관 증폭 효과:
        - 지표들 간 상관이 높을수록 CISS 값 증가
        - 시스템 리스크의 동시 발생(contagion) 반영
    """

    # 섹터별 지표 매핑
    SECTOR_INDICATORS = {
        'Money_Market': ['MM1', 'MM2', 'MM3'],
        'Bond_Market': ['BD1', 'BD2', 'BD3'],
        'Equity_Market': ['EQ1', 'EQ2', 'EQ3'],
        'FX_Market': ['FX1', 'FX2', 'FX3'],
        'Financial_Intermediaries': ['FI1', 'FI2', 'FI3'],
    }

    # 섹터 가중치 (균등)
    SECTOR_WEIGHTS = {
        'Money_Market': 0.20,
        'Bond_Market': 0.20,
        'Equity_Market': 0.20,
        'FX_Market': 0.20,
        'Financial_Intermediaries': 0.20,
    }

    def __init__(self, sector_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            sector_weights: 섹터별 가중치 (기본값: 균등 20%)
        """
        self.sector_weights = sector_weights or self.SECTOR_WEIGHTS
        self._validate_weights()

    def _validate_weights(self):
        """가중치 합계 검증"""
        total = sum(self.sector_weights.values())
        if not np.isclose(total, 1.0):
            print(f"[WARN] Sector weights sum to {total}, normalizing to 1.0")
            for k in self.sector_weights:
                self.sector_weights[k] /= total

    def compute_sector_indices(self, ecdf_data: pd.DataFrame) -> pd.DataFrame:
        """
        섹터별 스트레스 지수 계산 (단순 평균)

        Args:
            ecdf_data: ECDF 변환된 15개 지표

        Returns:
            5개 섹터 지수 DataFrame
        """
        sector_indices = pd.DataFrame(index=ecdf_data.index)

        for sector, indicators in self.SECTOR_INDICATORS.items():
            available = [ind for ind in indicators if ind in ecdf_data.columns]
            if available:
                sector_indices[sector] = ecdf_data[available].mean(axis=1)
            else:
                print(f"[WARN] No indicators for {sector}")
                sector_indices[sector] = 0.5

        return sector_indices

    def compute_ciss(self,
                     ecdf_data: pd.DataFrame,
                     correlations: np.ndarray) -> pd.DataFrame:
        """
        CISS 종합 지수 계산

        Args:
            ecdf_data: ECDF 변환된 15개 지표 (T x 15)
            correlations: 동적 상관행렬 (T x 15 x 15)

        Returns:
            CISS 결과 DataFrame (CISS, 섹터별 기여도)
        """
        T = len(ecdf_data)
        n = ecdf_data.shape[1]

        # 지표 순서 맞추기
        indicator_order = []
        for sector in self.SECTOR_INDICATORS:
            indicator_order.extend(self.SECTOR_INDICATORS[sector])

        # ecdf_data를 지표 순서로 정렬
        available_indicators = [ind for ind in indicator_order if ind in ecdf_data.columns]
        ecdf_ordered = ecdf_data[available_indicators]

        # 가중치 벡터 구성 (지표별)
        indicator_weights = []
        for sector in self.SECTOR_INDICATORS:
            sector_w = self.sector_weights[sector]
            n_indicators = len([ind for ind in self.SECTOR_INDICATORS[sector]
                               if ind in available_indicators])
            if n_indicators > 0:
                w_per_indicator = sector_w / n_indicators
                for ind in self.SECTOR_INDICATORS[sector]:
                    if ind in available_indicators:
                        indicator_weights.append(w_per_indicator)

        w = np.array(indicator_weights)
        w = w / w.sum()  # 정규화

        # CISS 계산
        ciss_values = np.zeros(T)
        correlation_effect = np.zeros(T)

        for t in range(T):
            s_t = ecdf_ordered.iloc[t].values

            # 상관행렬 (지표 수에 맞게 조정)
            if correlations.shape[1] == len(w):
                C_t = correlations[t]
            else:
                # 상관행렬 크기 불일치 시 단위행렬 사용
                C_t = np.eye(len(w))

            # CISS = (s' * w) * sqrt(w' * C * w)
            # 또는 포트폴리오 분산 형태: w' * diag(s) * C * diag(s) * w
            s_weighted = s_t * w
            portfolio_var = w @ C_t @ w
            correlation_effect[t] = np.sqrt(portfolio_var)

            # 최종 CISS (상관 증폭 반영)
            ciss_values[t] = np.sum(s_weighted) * correlation_effect[t]

        # 결과 DataFrame
        result = pd.DataFrame(index=ecdf_data.index)
        result['CISS'] = ciss_values
        result['Correlation_Effect'] = correlation_effect

        # 섹터별 기여도
        sector_indices = self.compute_sector_indices(ecdf_data)
        for sector in self.SECTOR_INDICATORS:
            result[f'{sector}_Contribution'] = (
                sector_indices[sector] * self.sector_weights[sector]
            )

        return result

    def compute_simple_ciss(self, ecdf_data: pd.DataFrame) -> pd.DataFrame:
        """
        간단한 CISS 계산 (상관행렬 없이)

        상관행렬 추정이 어려운 경우 사용
        단순 가중 평균 방식
        """
        sector_indices = self.compute_sector_indices(ecdf_data)

        result = pd.DataFrame(index=ecdf_data.index)

        # 가중 평균
        ciss_simple = pd.Series(0.0, index=ecdf_data.index)
        for sector, weight in self.sector_weights.items():
            ciss_simple += sector_indices[sector] * weight

        result['CISS_Simple'] = ciss_simple

        # 섹터별 기여도
        for sector in self.SECTOR_INDICATORS:
            result[f'{sector}_Contribution'] = (
                sector_indices[sector] * self.sector_weights[sector]
            )

        return result


def compute_ciss_score(ecdf_data: pd.DataFrame,
                       correlations: np.ndarray,
                       use_correlation: bool = True) -> pd.DataFrame:
    """
    CISS 스코어 계산 (편의 함수)

    Args:
        ecdf_data: ECDF 변환된 지표
        correlations: 동적 상관행렬
        use_correlation: 상관 증폭 효과 사용 여부

    Returns:
        CISS 결과 DataFrame
    """
    calculator = CISSCalculator()

    if use_correlation and correlations is not None:
        return calculator.compute_ciss(ecdf_data, correlations)
    else:
        return calculator.compute_simple_ciss(ecdf_data)


if __name__ == '__main__':
    from data_loader import load_raw_data
    from transforms import compute_indicators
    from dcc_garch import compute_dynamic_correlations

    # 데이터 로드
    daily, weekly = load_raw_data()
    indicators, ecdf_data = compute_indicators(weekly)
    correlations, _ = compute_dynamic_correlations(ecdf_data)

    # CISS 계산
    ciss_result = compute_ciss_score(ecdf_data, correlations)

    print("\n[CISS Results]")
    print(ciss_result.tail(10))

    print("\n[Statistics]")
    print(ciss_result['CISS'].describe())
