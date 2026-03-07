import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"

for d in [CACHE_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Data collection
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
KOSPI_INDEX_CODE = "KS11"  # KOSPI 지수

# Train / test split
TRAIN_END_DATE = "2020-12-31"
TEST_START_DATE = "2021-01-01"

# Top N stocks by market cap to use
TOP_N_STOCKS = 50

# KOSPI200 대표 종목 코드 (시가총액 상위)
KOSPI200_STOCKS = [
    "005930",  # 삼성전자
    "000660",  # SK하이닉스
    "207940",  # 삼성바이오로직스
    "005490",  # POSCO홀딩스
    "035420",  # NAVER
    "000270",  # 기아
    "005380",  # 현대차
    "051910",  # LG화학
    "006400",  # 삼성SDI
    "068270",  # 셀트리온
    "105560",  # KB금융
    "055550",  # 신한지주
    "035720",  # 카카오
    "003550",  # LG
    "028260",  # 삼성물산
    "066570",  # LG전자
    "032830",  # 삼성생명
    "017670",  # SK텔레콤
    "030200",  # KT
    "015760",  # 한국전력
    "034020",  # 두산에너빌리티
    "096770",  # SK이노베이션
    "018260",  # 삼성에스디에스
    "047050",  # 포스코인터내셔널
    "003490",  # 대한항공
    "010950",  # S-Oil
    "011170",  # 롯데케미칼
    "009150",  # 삼성전기
    "000810",  # 삼성화재
    "086790",  # 하나금융지주
    "024110",  # 기업은행
    "139480",  # 이마트
    "009830",  # 한화솔루션
    "011780",  # 금호석유
    "042660",  # 대우조선해양
    "329180",  # 현대중공업
    "010130",  # 고려아연
    "000100",  # 유한양행
    "002380",  # KCC
    "021240",  # 코웨이
]

# Technical indicator parameters
MA_WINDOWS = [5, 20, 60, 120]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
VOLUME_MA_PERIOD = 20

# Rule-based signal thresholds
RSI_OVERSOLD = 40
RSI_OVERBOUGHT = 75
RSI_BUY_MAX = 70
VOLUME_RATIO_THRESHOLD = 1.2  # 평균 대비 120%

# ML model parameters
LABEL_HORIZON = 5  # 5일 후 등락 예측
WEEKLY_LABEL_HORIZON = 4  # 4주 후 등락 예측
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
}

# LSTM parameters
LSTM_SEQ_LEN = 60        # 과거 60일 시퀀스
LSTM_FEATURES = [        # LSTM 입력 feature
    "Close", "Volume", "RSI", "MACD", "MACD_Signal",
    "BB_Upper", "BB_Lower", "MA5", "MA20", "MA60",
]
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_UNITS = [64, 32]    # 2-layer LSTM units
LSTM_DROPOUT = 0.2
LSTM_BUY_THRESHOLD = 0.01  # 예측 수익률 > 1% → 매수

# Signal weights (rule_based : xgboost : lstm)
SIGNAL_WEIGHTS = (0.3, 0.4, 0.3)

# Backtester parameters
INITIAL_CAPITAL = 10_000_000   # 1000만원
MAX_POSITION_RATIO = 0.10      # 종목당 최대 10%
COMMISSION = 0.00015           # 수수료 0.015%
SLIPPAGE = 0.001               # 슬리피지 0.1%
STOP_LOSS = -0.05              # 손절 -5%
TAKE_PROFIT = 0.15             # 익절 +15%
