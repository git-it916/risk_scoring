# -*- coding: utf-8 -*-
"""
CISS Risk Scoring - Main Pipeline (No-Bloomberg edition)
=========================================================
data_loader_v2 (ECOS + FRED + pykrx + yfinance + proxies)를 사용한 파이프라인.

기존 main.py와 동일하게 4단계 실행:
  1) 무료 API에서 raw data 수집
  2) 15개 CISS 지표 변환 + ECDF
  3) DCC-GARCH 동적 상관행렬
  4) CISS 스코어 계산

환경변수:
  ECOS_API_KEY  : 한국은행 ECOS API 키 (https://ecos.bok.or.kr/api/)
  FRED_API_KEY  : FRED API 키 (https://fred.stlouisfed.org/docs/api/api_key.html)
"""

import os
from datetime import datetime

import pandas as pd

from data_loader_v2 import load_raw_data_v2
from transforms import compute_indicators
from dcc_garch import compute_dynamic_correlations
from ciss_calculator import compute_ciss_score


class CISSPipelineV2:
    """무료 API 데이터로 동작하는 CISS 파이프라인."""

    def __init__(
        self,
        start_date: str = '2024-01-01',
        end_date: str = None,
        dcc_method: str = 'ewma',
        frequency: str = 'daily',
    ):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.dcc_method = dcc_method
        self.frequency = frequency

        self.daily_data = None
        self.weekly_data = None
        self.raw_indicators = None
        self.ecdf_indicators = None
        self.correlations = None
        self.ciss_result = None

    def run(self, verbose: bool = True) -> pd.DataFrame:
        if verbose:
            print("=" * 60)
            print(" CISS Risk Scoring Pipeline  (No-Bloomberg / Free APIs)")
            print(f" Period: {self.start_date} ~ {self.end_date}")
            print(f" ECOS key set : {bool(os.getenv('ECOS_API_KEY'))}")
            print(f" FRED key set : {bool(os.getenv('FRED_API_KEY'))}")
            print("=" * 60)

        # Step 1: 데이터 수집
        if verbose:
            print("\n[Step 1/4] Loading data from free APIs...")
        self.daily_data, self.weekly_data = load_raw_data_v2(
            self.start_date, self.end_date
        )
        if verbose:
            print(f"  - Daily : {len(self.daily_data)} rows")
            print(f"  - Weekly: {len(self.weekly_data)} rows")

        # 주기 선택
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
            print(f"  - Indicators  : {self.ecdf_indicators.shape[1]} columns")
            print(f"  - Observations: {len(self.ecdf_indicators)} {freq_label}")

        # Step 3: DCC-GARCH 상관
        if verbose:
            print(f"\n[Step 3/4] Estimating dynamic correlations ({self.dcc_method})...")
        self.correlations, _ = compute_dynamic_correlations(
            self.ecdf_indicators, method=self.dcc_method
        )
        if verbose:
            print(f"  - Correlation matrices: {self.correlations.shape}")

        # Step 4: CISS
        if verbose:
            print("\n[Step 4/4] Computing CISS score...")
        self.ciss_result = compute_ciss_score(
            self.ecdf_indicators, self.correlations
        )
        if verbose:
            print(
                f"  - CISS range : [{self.ciss_result['CISS'].min():.4f}, "
                f"{self.ciss_result['CISS'].max():.4f}]"
            )
            print(f"  - Latest CISS: {self.ciss_result['CISS'].iloc[-1]:.4f}")
            print("\n" + "=" * 60)
            print(" Pipeline completed successfully!")
            print("=" * 60)

        return self.ciss_result

    def get_latest_score(self) -> dict:
        if self.ciss_result is None:
            raise ValueError("Pipeline not run yet. Call run() first.")
        latest = self.ciss_result.iloc[-1]
        return {
            'date': self.ciss_result.index[-1],
            'ciss': latest['CISS'],
            'correlation_effect': latest['Correlation_Effect'],
            'sector_contributions': {
                'Money_Market':               latest['Money_Market_Contribution'],
                'Bond_Market':                latest['Bond_Market_Contribution'],
                'Equity_Market':              latest['Equity_Market_Contribution'],
                'FX_Market':                  latest['FX_Market_Contribution'],
                'Financial_Intermediaries':   latest['Financial_Intermediaries_Contribution'],
            },
        }

    def save_results(self, output_dir: str = 'output_v2'):
        os.makedirs(output_dir, exist_ok=True)

        ciss_path = os.path.join(output_dir, 'ciss_results.csv')
        self.ciss_result.to_csv(ciss_path, encoding='utf-8-sig')
        print(f"[INFO] CISS results saved to: {ciss_path}")

        indicators_path = os.path.join(output_dir, 'raw_indicators.csv')
        self.raw_indicators.to_csv(indicators_path, encoding='utf-8-sig')
        print(f"[INFO] Raw indicators saved to: {indicators_path}")

        ecdf_path = os.path.join(output_dir, 'ecdf_indicators.csv')
        self.ecdf_indicators.to_csv(ecdf_path, encoding='utf-8-sig')
        print(f"[INFO] ECDF indicators saved to: {ecdf_path}")

        raw_path = os.path.join(output_dir, 'raw_data.csv')
        self.daily_data.to_csv(raw_path, encoding='utf-8-sig')
        print(f"[INFO] Raw data saved to: {raw_path}")

        # 전체 기간 누적 저장 (main.py와 동일 포맷)
        historical_path = os.path.join(output_dir, 'historical_ciss.csv')
        cols_to_save = [
            'CISS', 'Correlation_Effect',
            'Money_Market_Contribution', 'Bond_Market_Contribution',
            'Equity_Market_Contribution', 'FX_Market_Contribution',
            'Financial_Intermediaries_Contribution',
        ]
        self.ciss_result[cols_to_save].to_csv(historical_path, encoding='utf-8-sig')
        print(f"[INFO] Historical CISS saved to: {historical_path}")

    def print_summary(self):
        if self.ciss_result is None:
            print("Pipeline not run yet.")
            return

        print("\n" + "=" * 60)
        print(" CISS SUMMARY (No-Bloomberg pipeline)")
        print("=" * 60)

        latest = self.get_latest_score()
        print(f"\n Date              : {latest['date'].strftime('%Y-%m-%d')}")
        print(f" CISS Score        : {latest['ciss']:.4f}")
        print(f" Correlation Effect: {latest['correlation_effect']:.4f}")

        print("\n Sector Contributions:")
        for sector, contrib in latest['sector_contributions'].items():
            bar = '#' * int(contrib * 50)
            print(f"   {sector:30s}: {contrib:.4f} {bar}")

        print("\n Statistics (Full Period):")
        print(f"   Mean   : {self.ciss_result['CISS'].mean():.4f}")
        print(f"   Std    : {self.ciss_result['CISS'].std():.4f}")
        print(f"   Min    : {self.ciss_result['CISS'].min():.4f}")
        print(f"   Max    : {self.ciss_result['CISS'].max():.4f}")
        print(f"   Latest : {self.ciss_result['CISS'].iloc[-1]:.4f}")

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
    pipeline = CISSPipelineV2(
        start_date='2024-01-01',
        dcc_method='ewma',
        frequency='daily',
    )
    pipeline.run(verbose=True)
    pipeline.print_summary()
    pipeline.save_results('output_v2')
    return pipeline


if __name__ == '__main__':
    main()
