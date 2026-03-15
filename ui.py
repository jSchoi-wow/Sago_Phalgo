import sys
import os
import threading
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.font_manager as fm

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QTextEdit, QTabWidget,
    QSplitter, QFrame, QProgressBar, QLineEdit, QGroupBox,
    QScrollArea, QGridLayout, QSizePolicy, QDockWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QMetaObject, Q_ARG
from PyQt6.QtGui import QFont, QColor, QPalette, QTextCursor

# ── 한글 폰트 설정 ──────────────────────────────────────────────
def _set_korean_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    for name in candidates:
        if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
_set_korean_font()

# ── 다크 테마 색상 ────────────────────────────────────────────────
BG      = "#1e1e2e"
SURFACE = "#2a2a3e"
ACCENT  = "#7c3aed"
ACCENT2 = "#06b6d4"
SUCCESS = "#10b981"
DANGER  = "#ef4444"
WARNING = "#f59e0b"
TEXT    = "#e2e8f0"
SUBTEXT = "#94a3b8"
BORDER  = "#3f3f5f"

STYLE = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-family: 'Malgun Gothic', sans-serif; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; background: {SURFACE}; }}
QTabBar::tab {{ background: {BG}; color: {SUBTEXT}; padding: 10px 22px; border-radius: 6px 6px 0 0; font-size: 13px; }}
QTabBar::tab:selected {{ background: {SURFACE}; color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QPushButton {{
    background: {ACCENT}; color: white; border: none; border-radius: 8px;
    padding: 10px 20px; font-size: 13px; font-weight: bold;
}}
QPushButton:hover {{ background: #6d28d9; }}
QPushButton:disabled {{ background: #3a3a5a; color: {SUBTEXT}; }}
QPushButton#secondary {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER};
}}
QPushButton#secondary:hover {{ background: {BORDER}; }}
QPushButton#danger {{ background: {DANGER}; }}
QPushButton#danger:hover {{ background: #dc2626; }}
QComboBox {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 6px; padding: 8px 12px; font-size: 13px;
}}
QComboBox::drop-down {{ border: none; }}
QComboBox QAbstractItemView {{ background: {SURFACE}; color: {TEXT}; selection-background-color: {ACCENT}; }}
QLineEdit {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 6px; padding: 8px 12px; font-size: 13px;
}}
QTextEdit {{
    background: #0d0d1a; color: #c8d3f5; border: 1px solid {BORDER};
    border-radius: 8px; padding: 8px; font-family: Consolas, monospace; font-size: 12px;
}}
QLabel {{ color: {TEXT}; }}
QLabel#title {{ font-size: 22px; font-weight: bold; color: {TEXT}; }}
QLabel#subtitle {{ font-size: 13px; color: {SUBTEXT}; }}
QLabel#metric {{ font-size: 26px; font-weight: bold; color: {ACCENT2}; }}
QLabel#metric_label {{ font-size: 11px; color: {SUBTEXT}; }}
QProgressBar {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px;
    height: 8px; text-align: center;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}
QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 8px; margin-top: 12px;
    padding: 12px; color: {SUBTEXT}; font-size: 12px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; }}
QScrollArea {{ border: none; background: transparent; }}
QFrame#card {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
}}
QDockWidget {{
    color: {TEXT}; font-size: 13px; font-weight: bold;
    border: 1px solid {BORDER};
}}
QDockWidget::title {{
    background: {SURFACE}; padding: 6px 10px; border-bottom: 1px solid {BORDER};
}}
"""

STOCK_LIST = {
    "삼성전자":      "005930",
    "SK하이닉스":    "000660",
    "NAVER":         "035420",
    "카카오":        "035720",
    "현대차":        "005380",
    "기아":          "000270",
    "LG화학":        "051910",
    "셀트리온":      "068270",
    "KB금융":        "105560",
    "신한지주":      "055550",
    "포스코홀딩스":  "005490",
    "삼성바이오로직스": "207940",
}


# ── 글로벌 로그 (싱글턴) ─────────────────────────────────────────
class GlobalLog:
    _instance = None

    @classmethod
    def get(cls):
        return cls._instance

    @classmethod
    def set(cls, widget):
        cls._instance = widget

    @classmethod
    def write(cls, msg: str, level: str = "info"):
        inst = cls._instance
        if inst is None:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        colors = {"error": DANGER, "warn": WARNING, "ok": SUCCESS, "info": "#c8d3f5"}
        color  = colors.get(level, colors["info"])
        html   = f'<span style="color:{SUBTEXT}">[{ts}]</span> <span style="color:{color}">{msg}</span>'
        QMetaObject.invokeMethod(
            inst, "append", Qt.ConnectionType.QueuedConnection, Q_ARG(str, html)
        )
        QMetaObject.invokeMethod(
            inst, "moveCursor", Qt.ConnectionType.QueuedConnection,
            Q_ARG(QTextCursor.MoveOperation, QTextCursor.MoveOperation.End)
        )


# ── 워커 스레드 ────────────────────────────────────────────────────
class Worker(QThread):
    log  = pyqtSignal(str, str)   # (message, level)
    done = pyqtSignal(bool, str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn     = fn
        self.args   = args
        self.kwargs = kwargs

    def run(self):
        import io
        from contextlib import redirect_stdout, redirect_stderr

        class Tee(io.StringIO):
            def __init__(self, sig, default_level="info"):
                super().__init__()
                self.sig   = sig
                self.level = default_level

            def write(self, s):
                s = s.rstrip()
                if not s:
                    return len(s) + 1
                lvl = "info"
                sl  = s.lower()
                if any(k in sl for k in ("error", "traceback", "exception", "오류")):
                    lvl = "error"
                elif any(k in sl for k in ("warning", "warn", "경고")):
                    lvl = "warn"
                elif any(k in sl for k in ("완료", "saved", "done", "ok", "success")):
                    lvl = "ok"
                self.sig.emit(s, lvl)
                return len(s)

            def flush(self): pass

        tee_out = Tee(self.log, "info")
        tee_err = Tee(self.log, "error")
        try:
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                self.fn(*self.args, **self.kwargs)
            self.log.emit("✅ 작업 완료", "ok")
            self.done.emit(True, "완료")
        except Exception as e:
            tb = traceback.format_exc()
            for line in tb.splitlines():
                self.log.emit(line, "error")
            self.done.emit(False, str(e))


# ── matplotlib 캔버스 ─────────────────────────────────────────────
class MplCanvas(FigureCanvas):
    def __init__(self, figsize=(10, 4)):
        fig = Figure(figsize=figsize, facecolor=BG, tight_layout=True)
        super().__init__(fig)
        self.setStyleSheet(f"background: {BG};")
        self.fig = fig

    def draw_placeholder(self, msg="데이터 없음"):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(SURFACE)
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                color=SUBTEXT, fontsize=14, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        self.draw()


# ── 메인 윈도우 ───────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sago Phalgo  |  한국 주식 자동매매 분석 시스템")
        self.resize(1440, 960)
        self.setStyleSheet(STYLE)
        self._worker = None

        # ── 중앙 위젯 ─────────────────────────────────────────────
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 12, 16, 4)
        layout.setSpacing(8)

        layout.addWidget(self._make_header())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_collect(),  "📥  데이터 수집")
        self.tabs.addTab(self._tab_train(),    "🧠  모델 학습")
        self.tabs.addTab(self._tab_backtest(), "📊  백테스트")
        self.tabs.addTab(self._tab_analyze(),  "🔍  종목 분석")
        self.tabs.addTab(self._tab_trading(),  "💹  실시간 거래")
        layout.addWidget(self.tabs)

        # 자동매매 엔진 (탭에서 공유)
        self._trader = None

        # 상태바
        self.status_bar = QLabel("준비")
        self.status_bar.setObjectName("subtitle")
        self.progress = QProgressBar()
        self.progress.setFixedHeight(6)
        self.progress.setRange(0, 0)
        self.progress.hide()
        btm = QHBoxLayout()
        btm.addWidget(self.status_bar)
        btm.addStretch()
        btm.addWidget(self.progress)
        layout.addLayout(btm)

        # ── 글로벌 로그 Dock ──────────────────────────────────────
        self._build_log_dock()

    # ── 글로벌 로그 Dock ──────────────────────────────────────────
    def _build_log_dock(self):
        dock = QDockWidget("🖥️  실행 로그  (오류·경고·진행상황 통합)", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        container = QWidget()
        vlay = QVBoxLayout(container)
        vlay.setContentsMargins(4, 4, 4, 4)
        vlay.setSpacing(4)

        # 버튼 행
        btn_row = QHBoxLayout()
        btn_clear = QPushButton("🗑  로그 지우기")
        btn_clear.setObjectName("secondary")
        btn_clear.setFixedHeight(30)
        btn_copy = QPushButton("📋  복사")
        btn_copy.setObjectName("secondary")
        btn_copy.setFixedHeight(30)
        self.log_filter = QComboBox()
        self.log_filter.setFixedHeight(30)
        self.log_filter.setFixedWidth(110)
        self.log_filter.addItems(["전체", "오류만", "경고만"])
        lbl = QLabel("필터:")
        lbl.setObjectName("subtitle")
        btn_row.addWidget(lbl)
        btn_row.addWidget(self.log_filter)
        btn_row.addStretch()
        btn_row.addWidget(btn_copy)
        btn_row.addWidget(btn_clear)
        vlay.addLayout(btn_row)

        self.global_log = QTextEdit()
        self.global_log.setReadOnly(True)
        self.global_log.setMinimumHeight(160)
        self.global_log.setMaximumHeight(260)
        self.global_log.document().setMaximumBlockCount(2000)
        vlay.addWidget(self.global_log)

        GlobalLog.set(self.global_log)

        btn_clear.clicked.connect(self.global_log.clear)
        btn_copy.clicked.connect(lambda: QApplication.clipboard().setText(
            self.global_log.toPlainText()))

        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        GlobalLog.write("Sago Phalgo 시작됨", "ok")

    # ── 헤더 ─────────────────────────────────────────────────────
    def _make_header(self):
        w = QFrame(); w.setObjectName("card"); w.setFixedHeight(68)
        lay = QHBoxLayout(w); lay.setContentsMargins(20, 0, 20, 0)
        title = QLabel("📈  Sago Phalgo"); title.setObjectName("title")
        sub   = QLabel("규칙 기반 + XGBoost + LSTM  |  KOSPI 자동매매 분석")
        sub.setObjectName("subtitle")
        lay.addWidget(title); lay.addWidget(sub); lay.addStretch()
        return w

    # ── 공통: 워커 실행 ──────────────────────────────────────────
    def _start_worker(self, fn, label: str, *args, **kwargs):
        if self._worker and self._worker.isRunning():
            GlobalLog.write("이미 작업이 실행 중입니다. 완료 후 다시 시도하세요.", "warn")
            return
        self.progress.show()
        self.status_bar.setText(f"실행 중: {label}")
        GlobalLog.write(f"▶ {label} 시작", "info")

        self._worker = Worker(fn, *args, **kwargs)
        self._worker.log.connect(self._on_worker_log)
        self._worker.done.connect(lambda ok, msg: self._on_done(ok, msg, label))
        self._worker.start()

    def _on_worker_log(self, msg: str, level: str):
        GlobalLog.write(msg, level)

    def _on_done(self, ok: bool, msg: str, label: str):
        self.progress.hide()
        if ok:
            self.status_bar.setText(f"✅  완료: {label}")
            GlobalLog.write(f"■ {label} 완료", "ok")
        else:
            self.status_bar.setText(f"❌  오류: {msg[:60]}")
            GlobalLog.write(f"■ {label} 실패: {msg}", "error")

    # ── 탭1: 데이터 수집 ─────────────────────────────────────────
    def _tab_collect(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(12)

        grp = QGroupBox("수집 옵션")
        glay = QHBoxLayout(grp)
        glay.addWidget(QLabel("기간:"))
        self.collect_start = QLineEdit("2010-01-01"); self.collect_start.setFixedWidth(120)
        self.collect_end   = QLineEdit("2024-12-31"); self.collect_end.setFixedWidth(120)
        glay.addWidget(self.collect_start); glay.addWidget(QLabel("~"))
        glay.addWidget(self.collect_end);  glay.addStretch()
        lay.addWidget(grp)

        btn = QPushButton("📥  데이터 수집 시작"); btn.setFixedHeight(44)
        btn.clicked.connect(self._run_collect)
        lay.addWidget(btn)

        info = QLabel(
            "• 수집 대상: KOSPI 대표 40개 종목  (일봉 + 주봉)\n"
            "• 저장 위치: data/cache/ 폴더 (CSV)\n"
            "• 이미 수집된 종목은 캐시에서 바로 불러옵니다"
        )
        info.setObjectName("subtitle")
        lay.addWidget(info)
        lay.addStretch()
        return w

    # ── 탭2: 모델 학습 ────────────────────────────────────────────
    def _tab_train(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(12)

        grp = QGroupBox("학습 옵션")
        glay = QHBoxLayout(grp)
        glay.addWidget(QLabel("종목:"))
        self.train_code = QComboBox()
        for name, code in STOCK_LIST.items():
            self.train_code.addItem(f"{name} ({code})", code)
        glay.addWidget(self.train_code); glay.addStretch()
        lay.addWidget(grp)

        btn = QPushButton("🤖  XGBoost + LSTM 학습"); btn.setFixedHeight(44)
        btn.clicked.connect(self._run_train)
        lay.addWidget(btn)

        self.fi_canvas = MplCanvas(figsize=(10, 3))
        self.fi_canvas.draw_placeholder("학습 후 Feature Importance가 여기에 표시됩니다")
        lay.addWidget(self.fi_canvas)
        return w

    # ── 탭3: 백테스트 ─────────────────────────────────────────────
    def _tab_backtest(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(12)

        grp = QGroupBox("백테스트 옵션")
        glay = QHBoxLayout(grp)
        glay.addWidget(QLabel("종목 코드 (쉼표 구분):"))
        self.bt_codes = QLineEdit("005930,000660,035420")
        glay.addWidget(self.bt_codes); glay.addStretch()
        lay.addWidget(grp)

        btn = QPushButton("🚀  백테스트 실행"); btn.setFixedHeight(44)
        btn.clicked.connect(self._run_backtest)
        lay.addWidget(btn)

        self.metric_cards = {}
        metrics_row = QHBoxLayout()
        for key, label in [("total_return","총 수익률"), ("annual_return","연간 수익률"),
                            ("sharpe_ratio","샤프 지수"), ("mdd","MDD"),
                            ("win_rate","승률"), ("n_trades","거래 횟수")]:
            card, val_lbl = self._make_metric_card(label, "-")
            self.metric_cards[key] = val_lbl
            metrics_row.addWidget(card)
        lay.addLayout(metrics_row)

        self.bt_canvas = MplCanvas(figsize=(10, 4))
        self.bt_canvas.draw_placeholder("백테스트 실행 후 수익 곡선이 표시됩니다")
        lay.addWidget(self.bt_canvas)
        return w

    def _make_metric_card(self, label: str, value: str):
        card = QFrame(); card.setObjectName("card"); card.setFixedHeight(88)
        clay = QVBoxLayout(card); clay.setContentsMargins(16, 8, 16, 8)
        lbl = QLabel(label); lbl.setObjectName("metric_label"); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        val = QLabel(value); val.setObjectName("metric");       val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        clay.addWidget(lbl); clay.addWidget(val)
        return card, val

    # ── 탭4: 종목 분석 ────────────────────────────────────────────
    def _tab_analyze(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(12)

        top = QHBoxLayout()
        top.addWidget(QLabel("종목:"))
        self.ana_code = QComboBox()
        for name, code in STOCK_LIST.items():
            self.ana_code.addItem(f"{name} ({code})", code)
        top.addWidget(self.ana_code)
        btn = QPushButton("🔍  분석"); btn.setFixedHeight(38)
        btn.clicked.connect(self._run_analyze)
        top.addWidget(btn); top.addStretch()
        lay.addLayout(top)

        self.ana_canvas = MplCanvas(figsize=(12, 9))
        self.ana_canvas.draw_placeholder("종목을 선택 후 분석 버튼을 눌러주세요")
        lay.addWidget(self.ana_canvas)
        return w

    # ── 수집 실행 ─────────────────────────────────────────────────
    def _run_collect(self):
        start = self.collect_start.text()
        end   = self.collect_end.text()

        def _collect():
            from data.collector import collect_all
            collect_all(start=start, end=end)

        self._start_worker(_collect, "데이터 수집")

    # ── 학습 실행 ─────────────────────────────────────────────────
    def _run_train(self):
        code = self.train_code.currentData()
        name = self.train_code.currentText()

        def _train():
            from data.collector import get_stock_daily, get_stock_weekly
            from data.preprocessor import (build_features, split_train_test,
                                           get_scaled_xy, build_lstm_sequences)
            from config.settings import LSTM_FEATURES, LSTM_SEQ_LEN
            import models.ml_model  as ml
            import models.lstm_model as lstm

            print(f"학습 종목: {name}")
            daily  = get_stock_daily(code)
            weekly = get_stock_weekly(code)
            df     = build_features(daily, weekly)
            train_df, test_df = split_train_test(df)
            X_train, y_train, X_test, y_test, scaler, feat_cols = get_scaled_xy(train_df, test_df)

            print("--- XGBoost 학습 시작 ---")
            xgb_model = ml.train(X_train, y_train)
            ml.evaluate(xgb_model, X_test, y_test, feature_names=feat_cols)
            ml.save(xgb_model, scaler)

            print("--- LSTM 학습 시작 ---")
            lstm_feats = [f for f in LSTM_FEATURES if f in df.columns]
            X_seq, y_seq, _ = build_lstm_sequences(df, lstm_feats, LSTM_SEQ_LEN)
            sp = int(len(X_seq) * 0.8)
            vp = int(sp * 0.9)
            lm = lstm.train(X_seq[:vp], y_seq[:vp], X_seq[vp:sp], y_seq[vp:sp])
            lstm.evaluate(lm, X_seq[sp:], y_seq[sp:])
            lstm.save(lm)

            self._draw_feature_importance(xgb_model, feat_cols)

        self._start_worker(_train, f"모델 학습 ({name})")

    def _draw_feature_importance(self, model, feat_cols):
        importance = model.feature_importances_
        idx   = np.argsort(importance)[-15:]
        names = [feat_cols[i] for i in idx]
        vals  = importance[idx]

        self.fi_canvas.fig.clear()
        ax = self.fi_canvas.fig.add_subplot(111)
        ax.set_facecolor(SURFACE)
        self.fi_canvas.fig.set_facecolor(BG)
        ax.barh(names, vals, color=ACCENT, edgecolor="none")
        ax.set_title("XGBoost Feature Importance (Top 15)", color=TEXT, fontsize=12)
        ax.tick_params(colors=SUBTEXT, labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        self.fi_canvas.draw()

    # ── 백테스트 실행 ─────────────────────────────────────────────
    def _run_backtest(self):
        codes = [c.strip() for c in self.bt_codes.text().split(",") if c.strip()]

        def _bt():
            from data.collector import get_stock_daily, get_stock_weekly, get_kospi_index
            from data.preprocessor import (build_features, split_train_test,
                                           build_lstm_sequences, FEATURE_COLS)
            from analysis.market_cycle import compute_market_cycle
            from models.rule_based import generate_signals
            from trading.signal_generator import generate_signals_for_df
            from trading.backtester import Backtester
            from config.settings import LSTM_FEATURES, LSTM_SEQ_LEN, TEST_START_DATE
            import models.ml_model  as ml
            import models.lstm_model as lstm

            print("모델 로드 중...")
            xgb_model, xgb_scaler = ml.load()
            lstm_model = lstm.load()

            try:
                kospi = get_kospi_index()
                market_cycles = compute_market_cycle(kospi)
                print("KOSPI 지수 로드 완료")
            except Exception as e:
                print(f"KOSPI 지수 로드 실패 (벤치마크 없이 진행): {e}")
                kospi = None; market_cycles = None

            signals_by_code = {}
            prices_by_code  = {}

            for code in codes:
                print(f"신호 생성 중: {code}")
                try:
                    daily  = get_stock_daily(code)
                    weekly = get_stock_weekly(code)
                    df     = build_features(daily, weekly)
                    _, test_df = split_train_test(df)
                    if test_df.empty:
                        print(f"  {code}: 테스트 데이터 없음, 건너뜀")
                        continue

                    feat_cols = [c for c in FEATURE_COLS if c in test_df.columns]
                    X_test    = xgb_scaler.transform(test_df[feat_cols])
                    xgb_probs = ml.predict_proba(xgb_model, X_test)

                    lstm_feats = [f for f in LSTM_FEATURES if f in df.columns]
                    X_seq, _, _ = build_lstm_sequences(df, lstm_feats, LSTM_SEQ_LEN)
                    lstm_ret = lstm.predict_return(lstm_model, X_seq)[-len(test_df):]
                    if len(lstm_ret) < len(test_df):
                        lstm_ret = np.pad(lstm_ret, (len(test_df) - len(lstm_ret), 0))

                    rule_sigs = generate_signals(daily, weekly)
                    test_rule = rule_sigs.reindex(test_df.index).fillna(0)

                    combined = generate_signals_for_df(
                        test_df.index, test_rule, xgb_probs, lstm_ret, market_cycles)
                    signals_by_code[code] = combined
                    prices_by_code[code]  = test_df[["Close"]]
                    print(f"  {code}: 신호 생성 완료 ({len(test_df)}일)")
                except Exception as e:
                    print(f"  {code} 처리 실패: {e}")
                    print(traceback.format_exc())

            if not signals_by_code:
                print("처리 가능한 종목 없음. 모델 학습을 먼저 해주세요.")
                return

            print("백테스트 실행 중...")
            bt = Backtester()
            kospi_test = kospi["Close"][kospi.index >= TEST_START_DATE] if kospi is not None else None
            portfolio  = bt.run(signals_by_code, prices_by_code)
            metrics    = bt.compute_metrics(portfolio, benchmark=kospi_test)
            bt.save_results(portfolio, metrics)

            self._update_metrics(metrics)
            self._draw_portfolio(portfolio, kospi_test)

        self._start_worker(_bt, "백테스트")

    def _update_metrics(self, m: dict):
        def fmt(k, v):
            if k in ("total_return", "annual_return", "mdd"):
                color = SUCCESS if v >= 0 else DANGER
                return f'<span style="color:{color}">{v*100:.2f}%</span>'
            if k == "win_rate":   return f"{v*100:.1f}%"
            if k == "sharpe_ratio": return f"{v:.3f}"
            return str(int(v))

        for key, lbl in self.metric_cards.items():
            lbl.setText(fmt(key, m.get(key, 0)))

    def _draw_portfolio(self, portfolio: pd.DataFrame, benchmark=None):
        self.bt_canvas.fig.clear()
        ax = self.bt_canvas.fig.add_subplot(111)
        ax.set_facecolor(SURFACE)
        self.bt_canvas.fig.set_facecolor(BG)

        init = portfolio["value"].iloc[0]
        pct  = (portfolio["value"] / init - 1) * 100
        ax.plot(portfolio.index, pct, color=ACCENT2, linewidth=2, label="전략")
        ax.fill_between(portfolio.index, pct, alpha=0.15, color=ACCENT2)

        if benchmark is not None:
            bm = benchmark.reindex(portfolio.index, method="ffill").dropna()
            if not bm.empty:
                bm_pct = (bm / bm.iloc[0] - 1) * 100
                ax.plot(bm.index, bm_pct, color=SUBTEXT, linewidth=1.2,
                        linestyle="--", label="KOSPI")

        ax.axhline(0, color=BORDER, linewidth=0.8)
        ax.set_title("포트폴리오 누적 수익률 (%)", color=TEXT, fontsize=13)
        ax.tick_params(colors=SUBTEXT)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT)
        self.bt_canvas.draw()

    # ── 탭5: 실시간 거래 ──────────────────────────────────────────────
    def _tab_trading(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setSpacing(12)

        # ── API 연결 설정 ──────────────────────────────────────────
        conn_grp = QGroupBox("KIS API 연결 설정")
        cgl = QGridLayout(conn_grp)
        cgl.addWidget(QLabel("APP KEY:"),    0, 0)
        self.kis_app_key = QLineEdit(); self.kis_app_key.setPlaceholderText("발급받은 APP KEY 입력")
        cgl.addWidget(self.kis_app_key, 0, 1)

        cgl.addWidget(QLabel("APP SECRET:"), 1, 0)
        self.kis_app_secret = QLineEdit(); self.kis_app_secret.setPlaceholderText("발급받은 APP SECRET 입력")
        self.kis_app_secret.setEchoMode(QLineEdit.EchoMode.Password)
        cgl.addWidget(self.kis_app_secret, 1, 1)

        cgl.addWidget(QLabel("계좌번호:"),   2, 0)
        self.kis_account = QLineEdit(); self.kis_account.setPlaceholderText("예) 12345678  (앞 8자리)")
        cgl.addWidget(self.kis_account, 2, 1)

        cgl.addWidget(QLabel("계좌상품:"),   3, 0)
        self.kis_prod = QLineEdit("01"); self.kis_prod.setFixedWidth(60)
        cgl.addWidget(self.kis_prod, 3, 1)

        self.kis_conn_status = QLabel("● 미연결")
        self.kis_conn_status.setStyleSheet(f"color: {DANGER}; font-weight: bold;")
        cgl.addWidget(self.kis_conn_status, 4, 0, 1, 2)

        btn_conn = QPushButton("🔌  연결 테스트")
        btn_conn.clicked.connect(self._kis_connect)
        cgl.addWidget(btn_conn, 5, 0, 1, 2)
        lay.addWidget(conn_grp)

        # ── 잔고 조회 ──────────────────────────────────────────────
        bal_grp = QGroupBox("계좌 잔고")
        bgl = QVBoxLayout(bal_grp)

        bal_btn_row = QHBoxLayout()
        btn_bal = QPushButton("💰  잔고 조회")
        btn_bal.clicked.connect(self._kis_get_balance)
        bal_btn_row.addWidget(btn_bal); bal_btn_row.addStretch()
        bgl.addLayout(bal_btn_row)

        self.bal_cards = {}
        bal_row = QHBoxLayout()
        for key, label in [("cash", "예수금"), ("total_eval", "평가총액"), ("n_pos", "보유종목")]:
            card, val = self._make_metric_card(label, "-")
            self.bal_cards[key] = val
            bal_row.addWidget(card)
        bgl.addLayout(bal_row)

        self.pos_text = QTextEdit()
        self.pos_text.setReadOnly(True)
        self.pos_text.setFixedHeight(120)
        self.pos_text.setPlaceholderText("잔고 조회 버튼을 누르면 보유 종목이 표시됩니다")
        bgl.addWidget(self.pos_text)
        lay.addWidget(bal_grp)

        # ── 자동매매 ───────────────────────────────────────────────
        auto_grp = QGroupBox("자동매매")
        agl = QVBoxLayout(auto_grp)

        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("감시 종목:"))
        self.trade_codes = QLineEdit("005930,000660,035420")
        opt_row.addWidget(self.trade_codes)
        opt_row.addWidget(QLabel("1회 매수금액(원):"))
        self.trade_amount = QLineEdit("500000"); self.trade_amount.setFixedWidth(100)
        opt_row.addWidget(self.trade_amount)
        agl.addLayout(opt_row)

        btn_row2 = QHBoxLayout()
        self.btn_trade_start = QPushButton("▶  자동매매 시작")
        self.btn_trade_start.clicked.connect(self._kis_start_trading)
        self.btn_trade_stop = QPushButton("■  중지")
        self.btn_trade_stop.setObjectName("danger")
        self.btn_trade_stop.clicked.connect(self._kis_stop_trading)
        self.btn_trade_stop.setEnabled(False)
        btn_row2.addWidget(self.btn_trade_start)
        btn_row2.addWidget(self.btn_trade_stop)
        btn_row2.addStretch()
        agl.addLayout(btn_row2)

        self.trade_status = QLabel("● 정지")
        self.trade_status.setStyleSheet(f"color: {SUBTEXT}; font-weight: bold;")
        agl.addWidget(self.trade_status)

        # 수동 신호 테스트
        sig_row = QHBoxLayout()
        sig_row.addWidget(QLabel("수동 신호 테스트:"))
        self.sig_code = QComboBox()
        for name, code in STOCK_LIST.items():
            self.sig_code.addItem(f"{name} ({code})", code)
        sig_row.addWidget(self.sig_code)
        btn_buy_sig  = QPushButton("매수 신호")
        btn_sell_sig = QPushButton("매도 신호"); btn_sell_sig.setObjectName("danger")
        btn_buy_sig.clicked.connect(lambda: self._manual_signal(1))
        btn_sell_sig.clicked.connect(lambda: self._manual_signal(-1))
        sig_row.addWidget(btn_buy_sig); sig_row.addWidget(btn_sell_sig)
        sig_row.addStretch()
        agl.addLayout(sig_row)
        lay.addWidget(auto_grp)

        # ── 주문 내역 ──────────────────────────────────────────────
        hist_grp = QGroupBox("주문 내역")
        hgl = QVBoxLayout(hist_grp)
        self.order_hist_text = QTextEdit()
        self.order_hist_text.setReadOnly(True)
        self.order_hist_text.setFixedHeight(130)
        self.order_hist_text.setPlaceholderText("주문 발생 시 여기에 표시됩니다")
        hgl.addWidget(self.order_hist_text)
        lay.addWidget(hist_grp)

        lay.addStretch()
        return w

    # ── KIS 연결 테스트 ───────────────────────────────────────────────
    def _kis_connect(self):
        self._apply_kis_config()

        def _conn():
            from trading.kis_api import KISApi
            api = KISApi()
            ok  = api.check_connection()
            if ok:
                self.kis_conn_status.setText("● 연결됨")
                self.kis_conn_status.setStyleSheet(f"color: {SUCCESS}; font-weight: bold;")
                GlobalLog.write("KIS API 연결 성공", "ok")
            else:
                self.kis_conn_status.setText("● 연결 실패")
                self.kis_conn_status.setStyleSheet(f"color: {DANGER}; font-weight: bold;")
                GlobalLog.write("KIS API 연결 실패 – APP_KEY/SECRET 확인", "error")

        import threading
        threading.Thread(target=_conn, daemon=True).start()

    def _apply_kis_config(self):
        """UI 입력값을 kis_config에 반영"""
        import config.kis_config as cfg
        key     = self.kis_app_key.text().strip()
        secret  = self.kis_app_secret.text().strip()
        account = self.kis_account.text().strip()
        prod    = self.kis_prod.text().strip() or "01"
        if key:    cfg.APP_KEY     = key
        if secret: cfg.APP_SECRET  = secret
        if account:
            cfg.ACCOUNT_NO   = account
            cfg.ACCOUNT_PROD = prod

    # ── 잔고 조회 ─────────────────────────────────────────────────────
    def _kis_get_balance(self):
        self._apply_kis_config()

        def _bal():
            from trading.kis_api import KISApi
            try:
                api = KISApi()
                bal = api.get_balance()
                self.bal_cards["cash"].setText(f"{bal['cash']:,}원")
                self.bal_cards["total_eval"].setText(f"{bal['total_eval']:,}원")
                self.bal_cards["n_pos"].setText(f"{len(bal['positions'])}종목")

                lines = []
                for p in bal["positions"]:
                    sign = "+" if p["pnl_pct"] >= 0 else ""
                    lines.append(
                        f"  {p['name']} ({p['code']})  "
                        f"{p['qty']}주  "
                        f"평균 {p['avg_price']:,}원  "
                        f"평가 {p['eval_value']:,}원  "
                        f"({sign}{p['pnl_pct']:.2f}%)"
                    )
                self.pos_text.setPlainText("\n".join(lines) if lines else "보유 종목 없음")
                GlobalLog.write(f"잔고 조회 완료 | 예수금 {bal['cash']:,}원", "ok")
            except Exception as e:
                GlobalLog.write(f"잔고 조회 실패: {e}", "error")

        import threading
        threading.Thread(target=_bal, daemon=True).start()

    # ── 자동매매 시작 / 중지 ──────────────────────────────────────────
    def _kis_start_trading(self):
        self._apply_kis_config()
        codes = [c.strip() for c in self.trade_codes.text().split(",") if c.strip()]
        try:
            amount = int(self.trade_amount.text().strip())
            import config.kis_config as cfg
            cfg.ORDER_AMOUNT = amount
        except ValueError:
            GlobalLog.write("매수금액이 올바르지 않습니다", "error")
            return

        from trading.auto_trader import AutoTrader
        self._trader = AutoTrader(log_fn=lambda msg, *a: GlobalLog.write(msg, "info"))
        self._trader.start(watch_codes=codes)

        self.trade_status.setText("● 실행 중")
        self.trade_status.setStyleSheet(f"color: {SUCCESS}; font-weight: bold;")
        self.btn_trade_start.setEnabled(False)
        self.btn_trade_stop.setEnabled(True)

    def _kis_stop_trading(self):
        if self._trader:
            self._trader.stop()
            self._trader = None
        self.trade_status.setText("● 정지")
        self.trade_status.setStyleSheet(f"color: {SUBTEXT}; font-weight: bold;")
        self.btn_trade_start.setEnabled(True)
        self.btn_trade_stop.setEnabled(False)

    # ── 수동 신호 테스트 ──────────────────────────────────────────────
    def _manual_signal(self, signal: int):
        if not self._trader:
            GlobalLog.write("자동매매를 먼저 시작하세요", "warn")
            return
        code = self.sig_code.currentData()
        name = self.sig_code.currentText()
        side = "매수" if signal == 1 else "매도"
        GlobalLog.write(f"수동 신호: {name} {side}", "info")
        import threading
        threading.Thread(
            target=self._trader.on_signal,
            args=(code, signal, 1.0),
            daemon=True,
        ).start()

    # ── 종목 분석 실행 ─────────────────────────────────────────────
    def _run_analyze(self):
        code = self.ana_code.currentData()
        name = self.ana_code.currentText()

        def _analyze():
            try:
                GlobalLog.write(f"분석 시작: {name}", "info")
                from data.collector import get_stock_daily, get_stock_weekly
                from analysis.technical import add_all_indicators
                from models.rule_based import generate_signals

                daily   = get_stock_daily(code)
                weekly  = get_stock_weekly(code)
                df      = add_all_indicators(daily)
                signals = generate_signals(daily, weekly)
                self._draw_analysis(df, signals, code)
                GlobalLog.write(f"분석 완료: {name}", "ok")
            except Exception as e:
                GlobalLog.write(f"분석 오류: {e}", "error")
                GlobalLog.write(traceback.format_exc(), "error")

        threading.Thread(target=_analyze, daemon=True).start()

    def _draw_analysis(self, df: pd.DataFrame, signals: pd.Series, code: str):
        fig = self.ana_canvas.fig
        fig.clear(); fig.set_facecolor(BG)
        axes = fig.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [3,1,1,1]})
        name = next((n for n, c in STOCK_LIST.items() if c == code), code)

        ax = axes[0]; ax.set_facecolor(SURFACE)
        ax.plot(df.index, df["Close"], color=TEXT,    linewidth=1.2, label="종가")
        ax.plot(df.index, df["MA20"],  color=WARNING,  linewidth=1,  alpha=0.8, label="MA20")
        ax.plot(df.index, df["MA60"],  color=ACCENT2,  linewidth=1,  alpha=0.8, label="MA60")
        ax.fill_between(df.index, df["BB_Upper"], df["BB_Lower"], alpha=0.06, color=ACCENT2)
        buys  = signals[signals ==  1].index
        sells = signals[signals == -1].index
        ax.scatter(buys,  df.loc[buys,  "Close"], marker="^", color=SUCCESS, s=45, zorder=5, label="매수")
        ax.scatter(sells, df.loc[sells, "Close"], marker="v", color=DANGER,  s=45, zorder=5, label="매도")
        ax.set_title(f"{name} ({code})  |  가격 & 매매 신호", color=TEXT, fontsize=13)
        ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=9, loc="upper left")

        ax = axes[1]; ax.set_facecolor(SURFACE)
        ax.plot(df.index, df["RSI"], color="#a78bfa", linewidth=1.2)
        ax.axhline(70, color=DANGER,  linewidth=0.8, linestyle="--")
        ax.axhline(40, color=SUCCESS, linewidth=0.8, linestyle="--")
        ax.fill_between(df.index, df["RSI"], 40, where=df["RSI"]<40, alpha=0.2, color=SUCCESS)
        ax.fill_between(df.index, df["RSI"], 70, where=df["RSI"]>70, alpha=0.2, color=DANGER)
        ax.set_ylim(0, 100); ax.set_title("RSI (14)", color=TEXT, fontsize=10)

        ax = axes[2]; ax.set_facecolor(SURFACE)
        ax.plot(df.index, df["MACD"],        color=ACCENT2, linewidth=1.2, label="MACD")
        ax.plot(df.index, df["MACD_Signal"], color=WARNING,  linewidth=1,  label="Signal")
        colors = [SUCCESS if v >= 0 else DANGER for v in df["MACD_Hist"]]
        ax.bar(df.index, df["MACD_Hist"], color=colors, alpha=0.5, width=1)
        ax.axhline(0, color=BORDER, linewidth=0.6)
        ax.set_title("MACD", color=TEXT, fontsize=10)
        ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

        ax = axes[3]; ax.set_facecolor(SURFACE)
        ax.bar(df.index, df["Volume"], color=ACCENT, alpha=0.5, width=1)
        ax.plot(df.index, df["Volume_MA"], color=WARNING, linewidth=1, label="거래량 MA20")
        ax.set_title("거래량", color=TEXT, fontsize=10)
        ax.legend(facecolor=SURFACE, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)

        for a in axes:
            a.tick_params(colors=SUBTEXT, labelsize=9)
            for spine in a.spines.values():
                spine.set_edgecolor(BORDER)

        axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[3].xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=30, ha="right", color=SUBTEXT)
        fig.subplots_adjust(hspace=0.35, left=0.06, right=0.98, top=0.95, bottom=0.08)
        self.ana_canvas.draw()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
