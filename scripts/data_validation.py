# -*- coding: utf-8 -*-
"""
CISS Risk Scoring - Bloomberg Data Validation Script
=====================================================
모든 지표의 Bloomberg 티커/필드 가용성을 검증하고 데이터 head를 출력합니다.

실행 방법:
    python scripts/data_validation.py

요구사항:
    - Bloomberg Terminal 실행 중
    - blpapi 또는 xbbg 설치
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Bloomberg API 임포트 시도
USE_XBBG = None
try:
    from xbbg import blp
    USE_XBBG = True
    print("[INFO] Using xbbg library")
except ImportError:
    try:
        import blpapi
        USE_XBBG = False
        print("[INFO] Using blpapi library")
    except ImportError:
        print("[ERROR] Bloomberg API library not found.")
        print("        Please install: pip install xbbg")


# =============================================================================
# TICKER DEFINITIONS
# =============================================================================
TICKERS_CONFIG = {
    # Primary Tickers (Verified)
    'primary': {
        'KWCDC Curncy': {
            'field': 'PX_LAST',
            'description': '91-day CD Rate',
            'used_by': ['MM1', 'MM2', 'MM3', 'FI3'],
            'expected_start': '1999-01-04'
        },
        'GVSK3M Index': {
            'field': 'PX_LAST',
            'description': 'KTB 3-Month Yield',
            'used_by': ['MM2'],
            'expected_start': '2001-01-02'
        },
        'GVSK3YR Index': {
            'field': 'PX_LAST',
            'description': 'KTB 3-Year Yield',
            'used_by': ['BD1', 'BD2'],
            'expected_start': '1999-01-04'
        },
        'GVSK10YR Index': {
            'field': 'PX_LAST',
            'description': 'KTB 10-Year Yield',
            'used_by': ['BD1'],
            'expected_start': '2000-10-02'
        },
        'MOVE Index': {
            'field': 'PX_LAST',
            'description': 'US Bond Volatility Index',
            'used_by': ['BD3'],
            'expected_start': '1988-01-04'
        },
        'KOSPI Index': {
            'field': 'PX_LAST',
            'description': 'KOSPI Index',
            'used_by': ['EQ1', 'EQ2_fallback', 'EQ3'],
            'expected_start': '1980-01-04'
        },
        'VKOSPI Index': {
            'field': 'PX_LAST',
            'description': 'KOSPI Implied Volatility',
            'used_by': ['EQ2'],
            'expected_start': '2009-04-13'
        },
        'USDKRW Curncy': {
            'field': 'PX_LAST',
            'description': 'USD/KRW Exchange Rate',
            'used_by': ['FX1', 'FX2_fallback'],
            'expected_start': '1981-01-02'
        },
        'KOCR Index': {
            'field': 'PX_LAST',
            'description': 'Call Rate (old)',
            'used_by': ['FI3'],
            'expected_start': '1999-01-04'
        },
        'KOOVNCAL Index': {
            'field': 'PX_LAST',
            'description': 'Korea Overnight Call Rate',
            'used_by': ['FI3_alt'],
            'expected_start': '2008-01-01'
        },
        'KOSPFIN Index': {
            'field': 'PX_LAST',
            'description': 'Financial Sector Index',
            'used_by': ['FI2'],
            'expected_start': '2001-01-02'
        },
    },

    # Tickers to Verify
    'to_verify': {
        # BD2 - Credit Spread (다양한 후보 티커)
        'KRCBAAA3 Index': {
            'field': 'PX_LAST',
            'description': 'Corp Bond AA- 3Y (estimated)',
            'used_by': ['BD2'],
            'alternatives': [
                'KBPAAM3Y Index',    # 회사채 AA- 3Y
                'KCBPAA3Y Index',    # 회사채 AA 3Y
                'KRCRAA3Y Index',    # KR Corp AA 3Y
                'KOFIAA3Y Index',    # KOFIA AA 3Y
            ]
        },
        # FX2 - FX Implied Vol
        'USDKRWV1M BGN Curncy': {
            'field': 'PX_LAST',
            'description': 'USD/KRW 1M Implied Vol',
            'used_by': ['FX2'],
            'alternatives': ['KRWUSDV1M Curncy', 'USDKRW1MV Curncy', 'KRW1MV Curncy']
        },
        # FX3 - CRS Basis
        'KWSWNI1 Curncy': {
            'field': 'PX_LAST',
            'description': '1Y KRW CRS',
            'used_by': ['FX3'],
            'alternatives': ['KWCRS1Y Index', 'KRSWC1 Curncy', 'KWSW1 Curncy']
        },
        # FI1 - Sovereign CDS
        'CKREA1U5 CBGN Curncy': {
            'field': 'PX_LAST',
            'description': 'Korea Sovereign CDS 5Y',
            'used_by': ['FI1'],
            'alternatives': ['KOREA CDS USD SR 5Y D14 Corp', 'CKREA5Y Curncy', 'CKOR1U5 Curncy']
        },
    },

    # Volume field for EQ3
    'volume': {
        'KOSPI Index': {
            'field': 'PX_VOLUME',
            'description': 'KOSPI Volume',
            'used_by': ['EQ3'],
            'expected_start': '1990-01-02'
        },
    }
}


def fetch_bdh_xbbg(ticker: str, field: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch BDH data using xbbg"""
    try:
        df = blp.bdh(
            tickers=ticker,
            flds=field,
            start_date=start_date,
            end_date=end_date
        )
        if df.empty:
            return pd.DataFrame()
        # Clean columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
        return df
    except Exception as e:
        print(f"    [ERROR] {ticker}: {str(e)}")
        return pd.DataFrame()


def validate_ticker(ticker: str, field: str, description: str,
                   start_date: str = '2024-01-01', num_rows: int = 5) -> Dict:
    """
    Validate single ticker and return data head
    """
    result = {
        'ticker': ticker,
        'field': field,
        'description': description,
        'status': 'UNKNOWN',
        'data_start': None,
        'data_end': None,
        'total_rows': 0,
        'missing_pct': None,
        'head': None,
        'tail': None,
        'error': None
    }

    if not USE_XBBG:
        result['status'] = 'SKIP'
        result['error'] = 'xbbg not available'
        return result

    end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        # Fetch BDH data
        df = fetch_bdh_xbbg(ticker, field, start_date, end_date)

        if df.empty:
            result['status'] = 'NO_DATA'
            result['error'] = 'Empty response from Bloomberg'
            return result

        # Data statistics
        result['status'] = 'OK'
        result['data_start'] = df.index.min().strftime('%Y-%m-%d')
        result['data_end'] = df.index.max().strftime('%Y-%m-%d')
        result['total_rows'] = len(df)

        # Missing ratio
        col = df.columns[0]
        missing = df[col].isna().sum()
        result['missing_pct'] = round(missing / len(df) * 100, 2)

        # Head/Tail
        result['head'] = df.head(num_rows).copy()
        result['tail'] = df.tail(num_rows).copy()

    except Exception as e:
        result['status'] = 'ERROR'
        result['error'] = str(e)

    return result


def print_validation_result(result: Dict):
    """Print validation result"""
    status_icon = {
        'OK': '[OK]',
        'NO_DATA': '[NO DATA]',
        'ERROR': '[ERROR]',
        'SKIP': '[SKIP]',
        'UNKNOWN': '[?]'
    }.get(result['status'], '[?]')

    print(f"\n{'='*70}")
    print(f"{status_icon} {result['ticker']} | {result['field']}")
    print(f"    {result['description']}")
    print(f"{'='*70}")

    if result['status'] == 'OK':
        print(f"    Data Range: {result['data_start']} ~ {result['data_end']}")
        print(f"    Total Rows: {result['total_rows']:,}")
        print(f"    Missing: {result['missing_pct']}%")

        print(f"\n    [HEAD - First {len(result['head'])} rows]")
        print(result['head'].to_string(index=True))

        print(f"\n    [TAIL - Last {len(result['tail'])} rows]")
        print(result['tail'].to_string(index=True))

    elif result['error']:
        print(f"    Error: {result['error']}")


def run_full_validation():
    """Run full ticker validation"""
    print("\n" + "="*70)
    print(" CISS RISK SCORING - BLOOMBERG DATA VALIDATION")
    print(" Execution Time:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("="*70)

    if USE_XBBG is None:
        print("\n[ERROR] Bloomberg API not available.")
        return

    all_results = []

    # 1. Primary Tickers
    print("\n\n" + "="*70)
    print(" SECTION 1: PRIMARY TICKERS (Verified)")
    print("="*70)

    for ticker, config in TICKERS_CONFIG['primary'].items():
        result = validate_ticker(
            ticker=ticker,
            field=config['field'],
            description=config['description']
        )
        result['used_by'] = config['used_by']
        result['category'] = 'primary'
        all_results.append(result)
        print_validation_result(result)

    # 2. Volume field
    print("\n\n" + "="*70)
    print(" SECTION 2: VOLUME FIELDS")
    print("="*70)

    for ticker, config in TICKERS_CONFIG['volume'].items():
        result = validate_ticker(
            ticker=ticker,
            field=config['field'],
            description=config['description']
        )
        result['used_by'] = config['used_by']
        result['category'] = 'volume'
        all_results.append(result)
        print_validation_result(result)

    # 3. Tickers to Verify
    print("\n\n" + "="*70)
    print(" SECTION 3: TICKERS TO VERIFY (Including alternatives)")
    print("="*70)

    for ticker, config in TICKERS_CONFIG['to_verify'].items():
        # Try primary ticker
        result = validate_ticker(
            ticker=ticker,
            field=config['field'],
            description=config['description']
        )
        result['used_by'] = config['used_by']
        result['category'] = 'to_verify'
        all_results.append(result)
        print_validation_result(result)

        # Try alternatives if primary failed
        if result['status'] != 'OK' and 'alternatives' in config:
            print(f"\n    Trying alternative tickers...")
            for alt_ticker in config['alternatives']:
                alt_result = validate_ticker(
                    ticker=alt_ticker,
                    field=config['field'],
                    description=f"[ALT] {config['description']}"
                )
                alt_result['used_by'] = config['used_by']
                alt_result['category'] = 'alternative'
                all_results.append(alt_result)

                if alt_result['status'] == 'OK':
                    print(f"    [FOUND] Alternative ticker: {alt_ticker}")
                    print_validation_result(alt_result)
                    break
                else:
                    print(f"    [X] {alt_ticker}: No data")

    # 4. Summary
    print("\n\n" + "="*70)
    print(" VALIDATION SUMMARY")
    print("="*70)

    summary_df = pd.DataFrame([
        {
            'Ticker': r['ticker'],
            'Field': r['field'],
            'Status': r['status'],
            'Start': r['data_start'] or '-',
            'Rows': r['total_rows'],
            'Missing%': r['missing_pct'] if r['missing_pct'] is not None else '-',
            'Used By': ', '.join(r.get('used_by', []))
        }
        for r in all_results
    ])

    print(summary_df.to_string(index=False))

    # Status counts
    print("\n\nStatus Summary:")
    status_counts = summary_df['Status'].value_counts()
    for status, count in status_counts.items():
        icon = {'OK': '[OK]', 'NO_DATA': '[X]', 'ERROR': '[!]'}.get(status, '[?]')
        print(f"   {icon} {status}: {count}")

    # Save results to CSV
    output_path = 'output/validation_results.csv'
    try:
        import os
        os.makedirs('output', exist_ok=True)
        summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n[INFO] Results saved to: {output_path}")
    except Exception as e:
        print(f"\n[WARN] Could not save results: {e}")

    return all_results


if __name__ == '__main__':
    results = run_full_validation()
