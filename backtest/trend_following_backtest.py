"""
단기 추세추종(터틀 스타일) 백테스트 엔진
=========================================

전략 규칙 (이전 대화에서 확정한 룰)
- 진입: 20일 신고가 종가 돌파 + MA50 > MA200 (정배열) 필터
- 청산(추세추종): 10일 신저가 하향 이탈
- 손절: 진입가 대비 -2*ATR
- 트레일링: 보유 중 최고가 대비 -2*ATR
- 포지션 사이징: 1거래당 리스크 = 계좌 자본의 1~2%
  수량 = (계좌 리스크 허용액) / (진입가 - 손절가)

사용 방법
---------
1) 실제 데이터로 쓸 때: OHLCV(날짜, 시가, 고가, 저가, 종가, 거래량) 컬럼을 가진
   CSV를 준비하고 run_backtest_from_csv() 호출.
   - 국내: 증권사 API(키움/이베스트 등) 또는 KRX 데이터 CSV 내보내기
   - 해외: 로컬 환경에서 `pip install yfinance` 후 yfinance로 받은 데이터를 CSV로 저장
2) 로직만 먼저 확인하고 싶을 때: generate_sample_data()로 합성 데이터를 만들어
   즉시 실행 가능 (아래 __main__ 참고)

이 파일 하나로 데이터만 바꿔 끼우면 그대로 재사용할 수 있게 설계했습니다.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ----------------------------------------------------------------------------
# 1. 지표 계산
# ----------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame,
                        entry_breakout: int = 20,
                        exit_breakout: int = 10,
                        ma_fast: int = 50,
                        ma_slow: int = 200,
                        atr_period: int = 20) -> pd.DataFrame:
    """OHLCV 데이터프레임에 전략에 필요한 지표를 추가한다.

    df는 컬럼 ['open','high','low','close','volume']을 가지고
    인덱스가 날짜(오름차순 정렬)여야 한다.
    """
    df = df.copy()

    # 돌파 기준선 (당일 제외, 직전 N일 기준으로 봐야 미래참조 오류가 없음)
    df["entry_high"] = df["close"].rolling(entry_breakout).max().shift(1)
    df["exit_low"] = df["close"].rolling(exit_breakout).min().shift(1)

    # 이동평균 정배열 필터
    df["ma_fast"] = df["close"].rolling(ma_fast).mean()
    df["ma_slow"] = df["close"].rolling(ma_slow).mean()
    df["trend_ok"] = df["ma_fast"] > df["ma_slow"]

    # ATR (True Range의 이동평균)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(atr_period).mean()

    return df


# ----------------------------------------------------------------------------
# 2. 백테스트 엔진
# ----------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    initial_capital: float = 10_000_000  # 초기 자본 (원 또는 달러 단위 자유)
    risk_per_trade: float = 0.01         # 1거래당 리스크 1%
    stop_atr_mult: float = 2.0           # 손절 = 진입가 - 2*ATR
    trail_atr_mult: float = 2.0          # 트레일링 = 최고가 - 2*ATR
    fee_rate: float = 0.0005             # 편도 수수료+슬리피지 가정치 (0.05%)


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    stop_price: float
    exit_date: pd.Timestamp = None
    exit_price: float = None
    exit_reason: str = None

    @property
    def pnl(self):
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def pnl_pct(self):
        if self.exit_price is None:
            return None
        return (self.exit_price / self.entry_price) - 1


def run_backtest(df: pd.DataFrame, config: BacktestConfig = BacktestConfig()):
    """단일 종목에 대해 규칙 기반 백테스트를 실행한다.

    반환: (거래내역 DataFrame, 자본곡선 Series, 요약 지표 dict)
    """
    cash = config.initial_capital
    equity_curve = []
    position: Trade | None = None
    trades: list[Trade] = []
    highest_since_entry = None

    for date, row in df.iterrows():
        price = row["close"]

        # 지표가 아직 다 안 쌓인 초기 구간은 스킵
        if pd.isna(row["entry_high"]) or pd.isna(row["atr"]) or pd.isna(row["ma_slow"]):
            equity_curve.append(cash if position is None else cash + position.shares * price)
            continue

        # ---- 보유 중이면 청산 조건 체크 ----
        if position is not None:
            highest_since_entry = max(highest_since_entry, row["high"])
            trailing_stop = highest_since_entry - config.trail_atr_mult * row["atr"]
            hard_stop = position.stop_price

            exit_reason = None
            exit_price = None

            if row["low"] <= hard_stop:
                exit_reason, exit_price = "손절(ATR 스탑)", hard_stop
            elif row["low"] <= trailing_stop and trailing_stop > hard_stop:
                exit_reason, exit_price = "트레일링 청산", trailing_stop
            elif price <= row["exit_low"]:
                exit_reason, exit_price = "추세이탈(10일 신저가)", price

            if exit_reason:
                proceeds = position.shares * exit_price * (1 - config.fee_rate)
                cash += proceeds
                position.exit_date = date
                position.exit_price = exit_price
                position.exit_reason = exit_reason
                trades.append(position)
                position = None
                highest_since_entry = None

        # ---- 미보유 상태면 진입 조건 체크 ----
        if position is None and row["trend_ok"] and price > row["entry_high"]:
            stop_price = price - config.stop_atr_mult * row["atr"]
            risk_amount = cash * config.risk_per_trade
            per_share_risk = price - stop_price

            if per_share_risk > 0:
                shares = risk_amount / per_share_risk
                cost = shares * price * (1 + config.fee_rate)
                if cost <= cash:  # 자본 초과 방지
                    cash -= cost
                    position = Trade(entry_date=date, entry_price=price,
                                      shares=shares, stop_price=stop_price)
                    highest_since_entry = row["high"]

        # ---- 자본곡선 기록 ----
        mkt_value = position.shares * price if position else 0
        equity_curve.append(cash + mkt_value)

    equity_curve = pd.Series(equity_curve, index=df.index, name="equity")

    trades_df = pd.DataFrame([{
        "진입일": t.entry_date, "진입가": t.entry_price, "수량": t.shares,
        "손절가": t.stop_price, "청산일": t.exit_date, "청산가": t.exit_price,
        "청산사유": t.exit_reason, "손익": t.pnl, "손익률(%)": None if t.pnl_pct is None else round(t.pnl_pct * 100, 2),
    } for t in trades])

    summary = summarize(equity_curve, trades_df, config)
    return trades_df, equity_curve, summary


def summarize(equity_curve: pd.Series, trades_df: pd.DataFrame, config: BacktestConfig) -> dict:
    total_return = equity_curve.iloc[-1] / config.initial_capital - 1
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_dd = drawdown.min()

    closed = trades_df.dropna(subset=["손익"]) if not trades_df.empty else trades_df
    n_trades = len(closed)
    win_trades = closed[closed["손익"] > 0] if n_trades else closed
    win_rate = len(win_trades) / n_trades if n_trades else float("nan")
    avg_win = win_trades["손익률(%)"].mean() if len(win_trades) else float("nan")
    lose_trades = closed[closed["손익"] <= 0] if n_trades else closed
    avg_loss = lose_trades["손익률(%)"].mean() if len(lose_trades) else float("nan")

    daily_ret = equity_curve.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else float("nan")

    return {
        "총수익률(%)": round(total_return * 100, 2),
        "최대낙폭 MDD(%)": round(max_dd * 100, 2),
        "총거래횟수": n_trades,
        "승률(%)": round(win_rate * 100, 2) if n_trades else None,
        "평균수익 거래(%)": round(avg_win, 2) if n_trades else None,
        "평균손실 거래(%)": round(avg_loss, 2) if n_trades else None,
        "샤프비율(연환산, 근사치)": round(sharpe, 2),
        "최종자본": round(equity_curve.iloc[-1]),
    }


# ----------------------------------------------------------------------------
# 3. 데이터 입출력 헬퍼
# ----------------------------------------------------------------------------

def run_backtest_from_csv(csv_path: str, date_col: str = "date",
                           config: BacktestConfig = BacktestConfig()):
    """CSV(open,high,low,close,volume 컬럼 필요)를 읽어 백테스트 실행.

    예) 국내 증권사 다운로드 CSV, 또는 로컬에서 yfinance로 받은 CSV.
    """
    raw = pd.read_csv(csv_path, parse_dates=[date_col])
    raw = raw.set_index(date_col).sort_index()
    raw.columns = [c.lower() for c in raw.columns]
    df = compute_indicators(raw)
    return run_backtest(df, config)


def generate_sample_data(n_days: int = 1000, seed: int = 42) -> pd.DataFrame:
    """로직 검증용 합성 OHLCV 데이터 생성 (실제 시세 아님, 데모 전용)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)

    # 추세 구간과 횡보 구간이 섞이도록 드리프트를 구간별로 다르게
    drift = np.concatenate([
        np.full(250, 0.0006),   # 상승 추세
        np.full(200, -0.0002),  # 약세/횡보
        np.full(300, 0.0009),   # 강한 상승 추세
        np.full(n_days - 750, 0.0001),
    ])[:n_days]
    noise = rng.normal(0, 0.012, n_days)
    daily_ret = drift + noise
    close = 50000 * np.exp(np.cumsum(daily_ret))

    high = close * (1 + np.abs(rng.normal(0, 0.006, n_days)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n_days)))
    open_ = low + (high - low) * rng.random(n_days)
    volume = rng.integers(100_000, 1_000_000, n_days)

    return pd.DataFrame({"open": open_, "high": high, "low": low,
                          "close": close, "volume": volume}, index=dates)


# ----------------------------------------------------------------------------
# 4. 파라미터 최적화 (그리드서치 + 워크포워드 검증)
# ----------------------------------------------------------------------------

def grid_search(raw_df: pd.DataFrame,
                 entry_breakouts=(10, 20, 40, 55),
                 exit_breakouts=(5, 10, 20),
                 stop_atr_mults=(1.5, 2.0, 2.5, 3.0),
                 ma_fast=50, ma_slow=200,
                 config: BacktestConfig = BacktestConfig()) -> pd.DataFrame:
    """진입/청산 기간, 손절 ATR 배수 조합을 모두 돌려 성과를 비교한다.

    raw_df: open/high/low/close/volume 원본 데이터 (지표 계산 전)
    반환: 조합별 성과 요약이 담긴 DataFrame (샤프비율 내림차순 정렬)
    """
    results = []
    for eb in entry_breakouts:
        for xb in exit_breakouts:
            if xb >= eb:
                continue  # 청산 기간이 진입 기간보다 길면 의미 없음
            for sm in stop_atr_mults:
                cfg = BacktestConfig(
                    initial_capital=config.initial_capital,
                    risk_per_trade=config.risk_per_trade,
                    stop_atr_mult=sm,
                    trail_atr_mult=config.trail_atr_mult,
                    fee_rate=config.fee_rate,
                )
                df = compute_indicators(raw_df, entry_breakout=eb, exit_breakout=xb,
                                         ma_fast=ma_fast, ma_slow=ma_slow)
                _, _, summary = run_backtest(df, cfg)
                results.append({
                    "진입기간": eb, "청산기간": xb, "손절ATR배수": sm,
                    **summary,
                })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("샤프비율(연환산, 근사치)", ascending=False).reset_index(drop=True)
    return result_df


def walk_forward_split(df: pd.DataFrame, n_splits: int = 4):
    """데이터를 n_splits개의 연속 구간으로 나눠 (train, test) 쌍을 만든다.

    각 구간은 절반은 파라미터를 고르는 데(train), 나머지 절반은
    그 파라미터를 그대로 적용해 검증하는 데(test) 쓰인다. 과최적화 방지용.
    """
    n = len(df)
    fold_size = n // n_splits
    splits = []
    for i in range(n_splits - 1):
        train_start = i * fold_size
        train_end = train_start + fold_size
        test_end = min(train_end + fold_size, n)
        splits.append((df.iloc[train_start:train_end], df.iloc[train_end:test_end]))
    return splits


def walk_forward_validate(raw_df: pd.DataFrame, top_n: int = 3, n_splits: int = 4,
                           **grid_kwargs) -> pd.DataFrame:
    """구간별로 grid_search 최적 조합을 찾고, 그 다음 구간에서 그대로 검증.

    train 구간에서 좋았던 조합이 test 구간(한 번도 보지 않은 미래 데이터)에서도
    성과가 유지되는지 확인 → 실전 신뢰도를 훨씬 더 잘 보여준다.
    """
    rows = []
    for i, (train, test) in enumerate(walk_forward_split(raw_df, n_splits)):
        train_results = grid_search(train, **grid_kwargs)
        if train_results.empty:
            continue
        best = train_results.iloc[0]

        test_ind = compute_indicators(test, entry_breakout=int(best["진입기간"]),
                                       exit_breakout=int(best["청산기간"]))
        cfg = BacktestConfig(stop_atr_mult=best["손절ATR배수"])
        _, _, test_summary = run_backtest(test_ind, cfg)

        rows.append({
            "구간": i + 1,
            "train_최적조합": f"진입{int(best['진입기간'])}/청산{int(best['청산기간'])}/ATR{best['손절ATR배수']}",
            "train_샤프": best["샤프비율(연환산, 근사치)"],
            "test_샤프": test_summary["샤프비율(연환산, 근사치)"],
            "test_수익률(%)": test_summary["총수익률(%)"],
            "test_MDD(%)": test_summary["최대낙폭 MDD(%)"],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 5. 데모 실행
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    sample = generate_sample_data()
    sample_with_ind = compute_indicators(sample)
    trades_df, equity_curve, summary = run_backtest(sample_with_ind)

    print("=== 백테스트 요약 (합성 데이터 데모) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== 최근 거래 5건 ===")
    print(trades_df.tail(5).to_string(index=False))

    equity_curve.to_csv("/home/claude/backtest/equity_curve_demo.csv")
    trades_df.to_csv("/home/claude/backtest/trades_demo.csv", index=False)

    print("\n=== 그리드서치 상위 5개 조합 (합성 데이터 데모) ===")
    grid_results = grid_search(sample)
    print(grid_results.head(5).to_string(index=False))

    print("\n=== 워크포워드 검증 (구간별 train 최적조합 -> test 성과) ===")
    wf_results = walk_forward_validate(sample, n_splits=4)
    print(wf_results.to_string(index=False))
