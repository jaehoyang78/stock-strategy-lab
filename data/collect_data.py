"""
로컬 PC에서 실행할 데이터 수집 스크립트
=====================================

이 파일은 이 컴퓨터(샌드박스)가 아니라 사용자님의 로컬 PC에서 실행하는 용도입니다.
(샌드박스는 보안상 외부 시세 서버 접속이 막혀 있어 여기서는 실행이 안 됩니다.)

실행 방법
---------
1) 터미널에서:
   pip install yfinance pandas

2) 이 파일을 저장한 폴더에서:
   python collect_data.py

3) 폴더에 KODEX200.csv, VOO.csv, QQQM.csv 세 개가 생성됩니다.
   그 파일들을 그대로 이 대화창에 업로드해주시면 이어서 백테스트를 진행합니다.

티커 안내
---------
- KODEX 200 (국내 KOSPI200 추종 ETF): 069500.KS
- VOO (Vanguard S&P 500 ETF): VOO
- QQQM (Invesco NASDAQ-100 ETF): QQQM
"""

import yfinance as yf
import pandas as pd

TICKERS = {
    "KODEX200": "069500.KS",
    "VOO": "VOO",
    "QQQM": "QQQM",
}

START_DATE = "2015-01-01"  # 필요하면 기간 조정 가능 (QQQM은 2020년 상장이라 그 이전은 데이터 없음)


def fetch_and_save(name: str, ticker: str, start: str = START_DATE):
    print(f"[{name}] {ticker} 데이터 다운로드 중...")
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)

    if df.empty:
        print(f"  -> 데이터가 비어 있습니다. 티커를 확인해주세요: {ticker}")
        return

    # yfinance 최신 버전은 컬럼이 멀티인덱스로 나올 수 있어 정리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df.columns = [c.lower() for c in df.columns]
    df = df.reset_index().rename(columns={"Date": "date"})
    df.columns = [c.lower() for c in df.columns]

    out_path = f"{name}.csv"
    df[["date", "open", "high", "low", "close", "volume"]].to_csv(out_path, index=False)
    print(f"  -> 저장 완료: {out_path} ({len(df)}행, {df['date'].min()} ~ {df['date'].max()})")


if __name__ == "__main__":
    for name, ticker in TICKERS.items():
        fetch_and_save(name, ticker)

    print("\n모든 다운로드 완료. 생성된 CSV 파일들을 Claude 대화창에 업로드해주세요.")