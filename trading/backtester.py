import pandas as pd
import numpy as np
from pathlib import Path

from config.settings import (
    INITIAL_CAPITAL, MAX_POSITION_RATIO, COMMISSION,
    SLIPPAGE, STOP_LOSS, TAKE_PROFIT, RESULTS_DIR,
)


class Position:
    def __init__(self, code: str, entry_price: float, shares: int, entry_date):
        self.code = code
        self.entry_price = entry_price
        self.shares = shares
        self.entry_date = entry_date

    def current_return(self, price: float) -> float:
        return (price - self.entry_price) / self.entry_price

    def market_value(self, price: float) -> float:
        return price * self.shares


class Backtester:
    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        max_position_ratio: float = MAX_POSITION_RATIO,
        commission: float = COMMISSION,
        slippage: float = SLIPPAGE,
        stop_loss: float = STOP_LOSS,
        take_profit: float = TAKE_PROFIT,
    ):
        self.initial_capital = initial_capital
        self.max_position_ratio = max_position_ratio
        self.commission = commission
        self.slippage = slippage
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.trade_history: list[dict] = []
        self.portfolio_values: list[dict] = []

    def _execution_price(self, price: float, side: str) -> float:
        slip = self.slippage if side == "buy" else -self.slippage
        return price * (1 + slip)

    def _buy(self, code: str, price: float, date, signal_confidence: float = 1.0):
        if code in self.positions:
            return

        exec_price = self._execution_price(price, "buy")
        budget = self.cash * self.max_position_ratio * signal_confidence
        budget = min(budget, self.cash)
        shares = int(budget // exec_price)
        if shares <= 0:
            return

        cost = exec_price * shares * (1 + self.commission)
        if cost > self.cash:
            return

        self.cash -= cost
        self.positions[code] = Position(code, exec_price, shares, date)
        self.trade_history.append({
            "date": date, "code": code, "action": "BUY",
            "price": exec_price, "shares": shares, "cost": cost,
        })

    def _sell(self, code: str, price: float, date, reason: str = "signal"):
        if code not in self.positions:
            return

        pos = self.positions.pop(code)
        exec_price = self._execution_price(price, "sell")
        proceeds = exec_price * pos.shares * (1 - self.commission)
        self.cash += proceeds

        ret = (exec_price - pos.entry_price) / pos.entry_price
        self.trade_history.append({
            "date": date, "code": code, "action": "SELL",
            "price": exec_price, "shares": pos.shares,
            "proceeds": proceeds, "return": ret, "reason": reason,
        })

    def _check_stop_take(self, prices: dict[str, float], date):
        for code in list(self.positions.keys()):
            price = prices.get(code)
            if price is None:
                continue
            pos = self.positions[code]
            ret = pos.current_return(price)
            if ret <= self.stop_loss:
                self._sell(code, price, date, reason="stop_loss")
            elif ret >= self.take_profit:
                self._sell(code, price, date, reason="take_profit")

    def _portfolio_value(self, prices: dict[str, float]) -> float:
        pos_val = sum(
            pos.market_value(prices.get(code, pos.entry_price))
            for code, pos in self.positions.items()
        )
        return self.cash + pos_val

    def run(
        self,
        signals_by_code: dict[str, pd.DataFrame],
        prices_by_code: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        signals_by_code: {code: DataFrame with columns ['signal', 'confidence']}
        prices_by_code:  {code: DataFrame with column 'Close'}

        Returns portfolio value series.
        """
        # 전체 날짜 목록
        all_dates = sorted(
            set().union(*[df.index for df in prices_by_code.values()])
        )

        for date in all_dates:
            # 현재 가격 수집
            current_prices = {
                code: float(df.loc[date, "Close"])
                for code, df in prices_by_code.items()
                if date in df.index
            }

            # 손절/익절 체크
            self._check_stop_take(current_prices, date)

            # 매매 신호 처리
            for code, sig_df in signals_by_code.items():
                if date not in sig_df.index:
                    continue
                if code not in current_prices:
                    continue

                row = sig_df.loc[date]
                signal = int(row["signal"])
                confidence = float(row.get("confidence", 1.0))
                price = current_prices[code]

                if signal == 1:
                    self._buy(code, price, date, confidence)
                elif signal == -1:
                    self._sell(code, price, date, reason="signal")

            # 포트폴리오 가치 기록
            total = self._portfolio_value(current_prices)
            self.portfolio_values.append({"date": date, "value": total})

        return pd.DataFrame(self.portfolio_values).set_index("date")

    def compute_metrics(
        self,
        portfolio: pd.DataFrame,
        benchmark: pd.Series | None = None,
    ) -> dict:
        values = portfolio["value"]
        total_return = (values.iloc[-1] / values.iloc[0]) - 1

        n_years = (values.index[-1] - values.index[0]).days / 365.25
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        daily_ret = values.pct_change().dropna()
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0

        rolling_max = values.cummax()
        drawdown = (values - rolling_max) / rolling_max
        mdd = float(drawdown.min())

        trades = pd.DataFrame(self.trade_history)
        if not trades.empty and "return" in trades.columns:
            sell_trades = trades[trades["action"] == "SELL"]
            wins = sell_trades[sell_trades["return"] > 0]
            win_rate = len(wins) / len(sell_trades) if len(sell_trades) > 0 else 0
            avg_win = wins["return"].mean() if len(wins) > 0 else 0
            losses = sell_trades[sell_trades["return"] <= 0]
            avg_loss = losses["return"].mean() if len(losses) > 0 else 0
        else:
            win_rate = avg_win = avg_loss = 0

        metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe,
            "mdd": mdd,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "n_trades": len([t for t in self.trade_history if t["action"] == "SELL"]),
        }

        if benchmark is not None:
            bm_return = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1
            metrics["benchmark_return"] = bm_return
            metrics["alpha"] = total_return - bm_return

        return metrics

    def save_results(self, portfolio: pd.DataFrame, metrics: dict) -> None:
        portfolio.to_csv(RESULTS_DIR / "portfolio_values.csv")
        pd.DataFrame(self.trade_history).to_csv(RESULTS_DIR / "trade_history.csv", index=False)

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)

        print("\n=== 백테스트 결과 ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print(f"\n결과 저장 → {RESULTS_DIR}")
