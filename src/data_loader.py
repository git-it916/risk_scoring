# -*- coding: utf-8 -*-
"""
CISS Risk Scoring - Data Loader Module
=======================================
Bloomberg API를 통한 데이터 수집 및 전처리
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    from xbbg import blp
    HAS_XBBG = True
except ImportError:
    HAS_XBBG = False
    print("[WARN] xbbg not installed. Using mock data for testing.")


class BloombergDataLoader:
    """Bloomberg 데이터 수집 클래스"""

    # 티커 설정
    TICKERS = {
        'KWCDC Curncy': 'PX_LAST',       # CD금리
        'GVSK3M Index': 'PX_LAST',       # 국고채 3M
        'GVSK3YR Index': 'PX_LAST',      # 국고채 3Y
        'GVSK10YR Index': 'PX_LAST',     # 국고채 10Y
        'MOVE Index': 'PX_LAST',         # 채권변동성
        'KOSPI Index': 'PX_LAST',        # KOSPI
        'VKOSPI Index': 'PX_LAST',       # VKOSPI
        'USDKRW Curncy': 'PX_LAST',      # 원달러
        'USDKRWV1M BGN Curncy': 'PX_LAST',  # FX Vol
        'KWSWNI1 Curncy': 'PX_LAST',     # CRS
        'CKREA1U5 CBGN Curncy': 'PX_LAST',  # CDS
        'KOSPFIN Index': 'PX_LAST',      # 금융업종
    }

    VOLUME_TICKERS = {
        'KOSPI Index': 'PX_VOLUME',      # KOSPI 거래량
    }

    def __init__(self, start_date: str = '2024-01-01', end_date: str = None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')

    def fetch_all_data(self) -> pd.DataFrame:
        """모든 티커 데이터 수집"""
        if not HAS_XBBG:
            return self._generate_mock_data()

        all_data = {}

        # PX_LAST 데이터
        print("[INFO] Fetching PX_LAST data...")
        for ticker, field in self.TICKERS.items():
            print(f"  - {ticker}")
            try:
                df = blp.bdh(
                    tickers=ticker,
                    flds=field,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                if not df.empty:
                    col_name = ticker.replace(' ', '_').replace('/', '_')
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col_name]
                    else:
                        df.columns = [col_name]
                    all_data[col_name] = df[col_name]
            except Exception as e:
                print(f"    [ERROR] {ticker}: {e}")

        # Volume 데이터
        print("[INFO] Fetching Volume data...")
        for ticker, field in self.VOLUME_TICKERS.items():
            print(f"  - {ticker} ({field})")
            try:
                df = blp.bdh(
                    tickers=ticker,
                    flds=field,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                if not df.empty:
                    col_name = ticker.replace(' ', '_') + '_VOLUME'
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col_name]
                    else:
                        df.columns = [col_name]
                    all_data[col_name] = df[col_name]
            except Exception as e:
                print(f"    [ERROR] {ticker}: {e}")

        # DataFrame 병합
        result = pd.DataFrame(all_data)
        result.index = pd.to_datetime(result.index)
        result = result.sort_index()

        print(f"[INFO] Loaded {len(result)} rows, {len(result.columns)} columns")
        return result

    def _generate_mock_data(self) -> pd.DataFrame:
        """테스트용 Mock 데이터 생성"""
        print("[INFO] Generating mock data for testing...")
        dates = pd.date_range(self.start_date, self.end_date, freq='B')

        np.random.seed(42)
        n = len(dates)

        data = {
            'KWCDC_Curncy': 3.5 + np.cumsum(np.random.randn(n) * 0.01),
            'GVSK3M_Index': 3.4 + np.cumsum(np.random.randn(n) * 0.01),
            'GVSK3YR_Index': 3.3 + np.cumsum(np.random.randn(n) * 0.02),
            'GVSK10YR_Index': 3.5 + np.cumsum(np.random.randn(n) * 0.02),
            'MOVE_Index': 100 + np.cumsum(np.random.randn(n) * 2),
            'KOSPI_Index': 2600 + np.cumsum(np.random.randn(n) * 20),
            'VKOSPI_Index': 18 + np.abs(np.cumsum(np.random.randn(n) * 0.5)),
            'USDKRW_Curncy': 1300 + np.cumsum(np.random.randn(n) * 5),
            'USDKRWV1M_BGN_Curncy': 10 + np.abs(np.cumsum(np.random.randn(n) * 0.3)),
            'KWSWNI1_Curncy': 3.5 + np.cumsum(np.random.randn(n) * 0.02),
            'CKREA1U5_CBGN_Curncy': 30 + np.abs(np.cumsum(np.random.randn(n) * 1)),
            'KOSPFIN_Index': 380 + np.cumsum(np.random.randn(n) * 5),
            'KOSPI_Index_VOLUME': np.abs(400000000 + np.random.randn(n) * 50000000),
        }

        return pd.DataFrame(data, index=dates)

    def resample_weekly(self, df: pd.DataFrame, rule: str = 'W-FRI') -> pd.DataFrame:
        """주간 리샘플링 (금요일 기준)"""
        return df.resample(rule).last().dropna(how='all')


def load_raw_data(start_date: str = '2024-01-01', end_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    원본 데이터 로드 및 주간 리샘플링

    Returns:
        daily_data: 일별 원본 데이터
        weekly_data: 주간 리샘플링 데이터
    """
    loader = BloombergDataLoader(start_date, end_date)
    daily_data = loader.fetch_all_data()
    weekly_data = loader.resample_weekly(daily_data)

    return daily_data, weekly_data


if __name__ == '__main__':
    daily, weekly = load_raw_data()
    print("\n[Daily Data]")
    print(daily.tail())
    print("\n[Weekly Data]")
    print(weekly.tail())
