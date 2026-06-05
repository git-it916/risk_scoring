# -*- coding: utf-8 -*-
"""
CISS Risk Scoring - Main Pipeline
==================================
전체 파이프라인 실행: 데이터 수집 → 변환 → DCC-GARCH → CISS 계산
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# 모듈 임포트
from data_loader_v2 import load_raw_data_v2
from transforms import compute_indicators, IndicatorTransformer, ECDFTransformer
from dcc_garch import compute_dynamic_correlations
from ciss_calculator import compute_ciss_score, CISSCalculator

# 프로젝트 루트(이 파일의 상위 폴더) 기준 output 경로 — 실행 위치/PC에 무관하게 고정
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(REPO_ROOT, 'output')


class CISSPipeline:
    """CISS 계산 전체 파이프라인"""

    def __init__(self,
                 start_date: str = '2024-01-01',
                 end_date: str = None,
                 dcc_method: str = 'ewma',
                 frequency: str = 'daily'):
        """
        Args:
            start_date: 시작일
            end_date: 종료일 (None = 오늘)
            dcc_method: 'dcc' (full DCC-GARCH) 또는 'ewma'
            frequency: 'daily' (일간) 또는 'weekly' (주간, 금요일 기준)
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.dcc_method = dcc_method
        self.frequency = frequency

        # 결과 저장
        self.daily_data = None
        self.weekly_data = None
        self.raw_indicators = None
        self.ecdf_indicators = None
        self.correlations = None
        self.ciss_result = None

    def run(self, verbose: bool = True) -> pd.DataFrame:
        """
        전체 파이프라인 실행

        Returns:
            CISS 결과 DataFrame
        """
        if verbose:
            print("=" * 60)
            print(" CISS Risk Scoring Pipeline")
            print(f" Period: {self.start_date} ~ {self.end_date}")
            print("=" * 60)

        # Step 1: 데이터 수집
        if verbose:
            print("\n[Step 1/4] Loading data from ECOS/FRED...")
        self.daily_data, self.weekly_data = load_raw_data_v2(
            self.start_date, self.end_date
        )
        if verbose:
            print(f"  - Daily: {len(self.daily_data)} rows")
            print(f"  - Weekly: {len(self.weekly_data)} rows")

        # 주기에 따라 사용할 데이터 선택
        if self.frequency == 'daily':
            data_for_indicators = self.daily_data
            freq_label = 'days'
        else:
            data_for_indicators = self.weekly_data
            freq_label = 'weeks'

        # Step 2: 지표 변환
        if verbose:
            print(f"\n[Step 2/4] Computing indicators ({self.frequency})...")
        self.raw_indicators, self.ecdf_indicators = compute_indicators(
            data_for_indicators
        )
        if verbose:
            print(f"  - Indicators: {self.ecdf_indicators.shape[1]} columns")
            print(f"  - Observations: {len(self.ecdf_indicators)} {freq_label}")

        # Step 3: DCC-GARCH 상관행렬
        if verbose:
            print(f"\n[Step 3/4] Estimating dynamic correlations ({self.dcc_method})...")
        self.correlations, _ = compute_dynamic_correlations(
            self.ecdf_indicators, method=self.dcc_method
        )
        if verbose:
            print(f"  - Correlation matrices: {self.correlations.shape}")

        # Step 4: CISS 계산
        if verbose:
            print("\n[Step 4/4] Computing CISS score...")
        self.ciss_result = compute_ciss_score(
            self.ecdf_indicators, self.correlations
        )
        if verbose:
            print(f"  - CISS range: [{self.ciss_result['CISS'].min():.4f}, "
                  f"{self.ciss_result['CISS'].max():.4f}]")
            print(f"  - Latest CISS: {self.ciss_result['CISS'].iloc[-1]:.4f}")

        if verbose:
            print("\n" + "=" * 60)
            print(" Pipeline completed successfully!")
            print("=" * 60)

        return self.ciss_result

    def get_latest_score(self) -> dict:
        """최신 CISS 스코어 및 구성요소 반환"""
        if self.ciss_result is None:
            raise ValueError("Pipeline not run yet. Call run() first.")

        latest = self.ciss_result.iloc[-1]
        date = self.ciss_result.index[-1]

        return {
            'date': date,
            'ciss': latest['CISS'],
            'correlation_effect': latest['Correlation_Effect'],
            'sector_contributions': {
                'Money_Market': latest['Money_Market_Contribution'],
                'Bond_Market': latest['Bond_Market_Contribution'],
                'Equity_Market': latest['Equity_Market_Contribution'],
                'FX_Market': latest['FX_Market_Contribution'],
                'Financial_Intermediaries': latest['Financial_Intermediaries_Contribution'],
            }
        }

    def save_results(self, output_dir: str = OUTPUT_DIR):
        """결과 저장"""
        os.makedirs(output_dir, exist_ok=True)

        # CISS 결과
        ciss_path = os.path.join(output_dir, 'ciss_results.csv')
        self.ciss_result.to_csv(ciss_path, encoding='utf-8-sig')
        print(f"[INFO] CISS results saved to: {ciss_path}")

        # 원본 지표
        indicators_path = os.path.join(output_dir, 'raw_indicators.csv')
        self.raw_indicators.to_csv(indicators_path, encoding='utf-8-sig')
        print(f"[INFO] Raw indicators saved to: {indicators_path}")

        # ECDF 지표
        ecdf_path = os.path.join(output_dir, 'ecdf_indicators.csv')
        self.ecdf_indicators.to_csv(ecdf_path, encoding='utf-8-sig')
        print(f"[INFO] ECDF indicators saved to: {ecdf_path}")

        # Historical CISS 누적 저장
        self._save_historical_ciss(output_dir)

    def _save_historical_ciss(self, output_dir: str):
        """CISS 전체 기간 저장 (start_date ~ today)"""
        historical_path = os.path.join(output_dir, 'historical_ciss.csv')

        # 저장할 컬럼만 선택
        cols_to_save = ['CISS', 'Correlation_Effect',
                        'Money_Market_Contribution', 'Bond_Market_Contribution',
                        'Equity_Market_Contribution', 'FX_Market_Contribution',
                        'Financial_Intermediaries_Contribution']
        full_data = self.ciss_result[cols_to_save].copy()

        # 전체 기간 저장 (매번 새로 계산된 전체 데이터로 덮어씀)
        full_data.to_csv(historical_path, encoding='utf-8-sig')
        print(f"[INFO] Historical CISS saved to: {historical_path} ({len(full_data)} records)")

    def print_summary(self):
        """결과 요약 출력"""
        if self.ciss_result is None:
            print("Pipeline not run yet.")
            return

        print("\n" + "=" * 60)
        print(" CISS SUMMARY")
        print("=" * 60)

        latest = self.get_latest_score()
        print(f"\n Date: {latest['date'].strftime('%Y-%m-%d')}")
        print(f" CISS Score: {latest['ciss']:.4f}")
        print(f" Correlation Effect: {latest['correlation_effect']:.4f}")

        print("\n Sector Contributions:")
        for sector, contrib in latest['sector_contributions'].items():
            bar = '#' * int(contrib * 50)
            print(f"   {sector:30s}: {contrib:.4f} {bar}")

        # 통계
        print("\n Statistics (Full Period):")
        print(f"   Mean:   {self.ciss_result['CISS'].mean():.4f}")
        print(f"   Std:    {self.ciss_result['CISS'].std():.4f}")
        print(f"   Min:    {self.ciss_result['CISS'].min():.4f}")
        print(f"   Max:    {self.ciss_result['CISS'].max():.4f}")
        print(f"   Latest: {self.ciss_result['CISS'].iloc[-1]:.4f}")

        # 스트레스 레벨
        ciss_latest = latest['ciss']
        if ciss_latest > 0.7:
            level = "HIGH STRESS"
        elif ciss_latest > 0.5:
            level = "ELEVATED"
        elif ciss_latest > 0.3:
            level = "MODERATE"
        else:
            level = "LOW"

        print(f"\n Risk Level: {level}")
        print("=" * 60)


def main():
    """메인 실행"""
    # 파이프라인 생성 및 실행
    pipeline = CISSPipeline(
        start_date='2024-01-01',
        dcc_method='ewma',
        frequency='daily'  # 'daily' 또는 'weekly'
    )

    # 실행
    ciss_result = pipeline.run(verbose=True)

    # 요약 출력
    pipeline.print_summary()

    # 결과 저장
    pipeline.save_results(OUTPUT_DIR)

    return pipeline


if __name__ == '__main__':
    pipeline = main()
