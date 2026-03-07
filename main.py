import argparse
import sys


def mode_collect(args):
    from data.collector import collect_all, get_stock_list_from_market
    from config.settings import START_DATE, END_DATE

    codes = get_stock_list_from_market() if args.market else None
    collect_all(codes=codes, start=START_DATE, end=END_DATE)


def mode_train(args):
    import numpy as np
    from data.collector import get_stock_daily, get_stock_weekly, get_kospi_index
    from data.preprocessor import build_features, split_train_test, get_scaled_xy, build_lstm_sequences
    from config.settings import KOSPI200_STOCKS, LSTM_FEATURES, LSTM_SEQ_LEN

    import models.ml_model as ml
    import models.lstm_model as lstm

    code = args.code or KOSPI200_STOCKS[0]
    print(f"학습 종목: {code}")

    daily = get_stock_daily(code)
    weekly = get_stock_weekly(code)
    df = build_features(daily, weekly)

    train_df, test_df = split_train_test(df)
    X_train, y_train, X_test, y_test, scaler, feat_cols = get_scaled_xy(train_df, test_df)

    # XGBoost
    print("\n--- XGBoost 학습 ---")
    xgb_model = ml.train(X_train, y_train, tune=args.tune)
    ml.evaluate(xgb_model, X_test, y_test, feature_names=feat_cols)
    ml.save(xgb_model, scaler)

    # LSTM
    print("\n--- LSTM 학습 ---")
    lstm_features = [f for f in LSTM_FEATURES if f in df.columns]
    X_seq, y_seq, lstm_scaler = build_lstm_sequences(df, lstm_features, LSTM_SEQ_LEN)

    split_idx = int(len(X_seq) * 0.8)
    X_tr, X_te = X_seq[:split_idx], X_seq[split_idx:]
    y_tr, y_te = y_seq[:split_idx], y_seq[split_idx:]
    val_split = int(len(X_tr) * 0.9)

    lstm_model = lstm.train(
        X_tr[:val_split], y_tr[:val_split],
        X_tr[val_split:], y_tr[val_split:],
    )
    lstm.evaluate(lstm_model, X_te, y_te)
    lstm.save(lstm_model)


def mode_backtest(args):
    import numpy as np
    from data.collector import get_stock_daily, get_stock_weekly, get_kospi_index
    from data.preprocessor import build_features, split_train_test, get_scaled_xy, build_lstm_sequences
    from analysis.market_cycle import compute_market_cycle
    from models.rule_based import generate_signals
    from trading.signal_generator import generate_signals_for_df
    from trading.backtester import Backtester
    from config.settings import KOSPI200_STOCKS, LSTM_FEATURES, LSTM_SEQ_LEN, TEST_START_DATE

    import models.ml_model as ml
    import models.lstm_model as lstm

    codes = args.codes.split(",") if args.codes else KOSPI200_STOCKS[:5]

    xgb_model, xgb_scaler = ml.load()
    lstm_model = lstm.load()

    try:
        kospi = get_kospi_index()
        market_cycles = compute_market_cycle(kospi)
    except Exception:
        kospi = None
        market_cycles = None

    signals_by_code = {}
    prices_by_code = {}

    for code in codes:
        print(f"신호 생성: {code}")
        try:
            daily = get_stock_daily(code)
            weekly = get_stock_weekly(code)
            df = build_features(daily, weekly)
            _, test_df = split_train_test(df)

            if test_df.empty:
                continue

            feat_cols = [c for c in test_df.columns if c in xgb_scaler.feature_names_in_] \
                if hasattr(xgb_scaler, "feature_names_in_") else []
            if not feat_cols:
                from data.preprocessor import FEATURE_COLS
                feat_cols = [c for c in FEATURE_COLS if c in test_df.columns]

            X_test = xgb_scaler.transform(test_df[feat_cols])
            xgb_probs = ml.predict_proba(xgb_model, X_test)

            lstm_features = [f for f in LSTM_FEATURES if f in df.columns]
            X_seq, _, lstm_sc = build_lstm_sequences(df, lstm_features, LSTM_SEQ_LEN)
            lstm_returns = lstm.predict_return(lstm_model, X_seq)[-len(test_df):]
            if len(lstm_returns) < len(test_df):
                lstm_returns = np.pad(lstm_returns, (len(test_df) - len(lstm_returns), 0))

            rule_sigs = generate_signals(daily, weekly)
            test_rule = rule_sigs.reindex(test_df.index).fillna(0)

            combined = generate_signals_for_df(
                test_df.index,
                test_rule,
                xgb_probs,
                lstm_returns,
                market_cycles,
            )

            signals_by_code[code] = combined
            prices_by_code[code] = test_df[["Close"]]

        except Exception as e:
            print(f"  [경고] {code} 처리 실패: {e}")

    if not signals_by_code:
        print("처리 가능한 종목 없음.")
        return

    bt = Backtester()
    kospi_test = kospi["Close"][kospi.index >= TEST_START_DATE] if kospi is not None else None
    portfolio = bt.run(signals_by_code, prices_by_code)
    metrics = bt.compute_metrics(portfolio, benchmark=kospi_test)
    bt.save_results(portfolio, metrics)


def mode_analyze(args):
    import matplotlib.pyplot as plt
    from data.collector import get_stock_daily, get_stock_weekly
    from analysis.technical import add_all_indicators
    from models.rule_based import generate_signals
    from config.settings import RESULTS_DIR

    code = args.code
    print(f"분석 종목: {code}")

    daily = get_stock_daily(code)
    weekly = get_stock_weekly(code)
    df = add_all_indicators(daily)
    signals = generate_signals(daily, weekly)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # 가격 + MA + 신호
    ax = axes[0]
    ax.plot(df.index, df["Close"], label="Close", linewidth=1)
    ax.plot(df.index, df["MA20"], label="MA20", linewidth=1, alpha=0.7)
    ax.plot(df.index, df["MA60"], label="MA60", linewidth=1, alpha=0.7)
    buy_dates = signals[signals == 1].index
    sell_dates = signals[signals == -1].index
    ax.scatter(buy_dates, df.loc[buy_dates, "Close"], marker="^", color="red", s=40, label="BUY")
    ax.scatter(sell_dates, df.loc[sell_dates, "Close"], marker="v", color="blue", s=40, label="SELL")
    ax.legend(fontsize=8)
    ax.set_title(f"{code} - Price & Signals")

    # RSI
    axes[1].plot(df.index, df["RSI"], color="purple")
    axes[1].axhline(40, linestyle="--", color="green", alpha=0.5)
    axes[1].axhline(75, linestyle="--", color="red", alpha=0.5)
    axes[1].set_title("RSI")
    axes[1].set_ylim(0, 100)

    # MACD
    axes[2].plot(df.index, df["MACD"], label="MACD")
    axes[2].plot(df.index, df["MACD_Signal"], label="Signal")
    axes[2].bar(df.index, df["MACD_Hist"], alpha=0.3, label="Histogram")
    axes[2].legend(fontsize=8)
    axes[2].set_title("MACD")

    # 거래량
    axes[3].bar(df.index, df["Volume"], alpha=0.5, color="gray")
    axes[3].plot(df.index, df["Volume_MA"], color="orange", linewidth=1, label="Vol MA20")
    axes[3].legend(fontsize=8)
    axes[3].set_title("Volume")

    plt.tight_layout()
    out_path = RESULTS_DIR / f"analyze_{code}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"차트 저장 → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="한국 주식 자동매매 분석 시스템")
    parser.add_argument("--mode", choices=["collect", "train", "backtest", "analyze"], required=True)
    parser.add_argument("--code", type=str, default=None, help="종목 코드 (예: 005930)")
    parser.add_argument("--codes", type=str, default=None, help="종목 코드 목록 (쉼표 구분)")
    parser.add_argument("--market", action="store_true", help="시장에서 종목 목록 자동 조회")
    parser.add_argument("--tune", action="store_true", help="XGBoost GridSearch 하이퍼파라미터 튜닝")
    args = parser.parse_args()

    dispatch = {
        "collect": mode_collect,
        "train": mode_train,
        "backtest": mode_backtest,
        "analyze": mode_analyze,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
