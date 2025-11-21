import time
import math
import uuid
import requests
import tkinter as tk
from tkinter import scrolledtext, messagebox
from datetime import datetime, timezone, timedelta

# tkcalendar 설치 필요:
# pip install tkcalendar
from tkcalendar import DateEntry

import hmac
import hashlib
from urllib.parse import urlencode
import threading

# ================================
# 0) Binance Futures Public 설정
# ================================
# 백테스트(과거 데이터)는 실서버 퍼블릭 API 사용
FAPI_URL = "https://fapi.binance.com"

# 데모(실시간 자동매매)는 Futures TESTNET 사용
DEMO_FAPI_URL = "https://testnet.binancefuture.com"

# 심볼별 대략적인 최소 수량 / 스텝
SYMBOL_CONFIG = {
    "BTCUSDT": {"step": 0.001, "min_notional": 5.0},
    "ETHUSDT": {"step": 0.01, "min_notional": 5.0},
    "BNBUSDT": {"step": 0.01, "min_notional": 5.0},
    "SOLUSDT": {"step": 0.1, "min_notional": 5.0},
    "XRPUSDT": {"step": 1.0, "min_notional": 5.0},
    "DOGEUSDT": {"step": 1.0, "min_notional": 5.0},
    "LINKUSDT": {"step": 0.1, "min_notional": 5.0},
}


# ================================
# 1) 유틸 함수
# ================================
def ts_from_str(s: str) -> int:
    """'YYYY-MM-DD HH:MM:SS' -> ms timestamp (UTC 기준 가정)"""
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ema_series(values, period):
    if len(values) < period:
        return [None] * len(values)
    emas = [None] * len(values)
    k = 2 / (period + 1)
    ema = sum(values[:period]) / period
    emas[period - 1] = ema
    for i in range(period, len(values)):
        ema = values[i] * k + ema * (1 - k)
        emas[i] = ema
    return emas


def atr_series(highs, lows, closes, period):
    """단순 ATR 계산"""
    if len(closes) < period + 1:
        return [None] * len(closes)

    trs = [0.0]
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    atrs = [None] * len(closes)
    # 초기값: SMA
    first_atr = sum(trs[1 : period + 1]) / period
    atrs[period] = first_atr

    alpha = 1 / period
    for i in range(period + 1, len(closes)):
        atrs[i] = (trs[i] * alpha) + atrs[i - 1] * (1 - alpha)
    return atrs



def fetch_klines(symbol, interval, start_ts, end_ts, log=None):
    """
    바이낸스 선물 kline 여러 페이지로 가져오기 (실서버, 백테스트용)
    """
    limit = 1500
    all_klines = []
    cur = start_ts

    if log:
        log(f"📥 캔들 데이터 가져오는 중... ({symbol}, {interval})\n")

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ts,
            "limit": limit,
        }
        resp = requests.get(FAPI_URL + "/fapi/v1/klines", params=params, timeout=10)
        if resp.status_code != 200:
            raise Exception(f"kline HTTP {resp.status_code}: {resp.text}")
        kl = resp.json()
        if not kl:
            break
        all_klines.extend(kl)
        if len(kl) < limit:
            break
        last_ts = kl[-1][0]
        cur = last_ts + 1
        if cur >= end_ts:
            break

    if log:
        log(f"   → 캔들 개수: {len(all_klines)}\n")
    return all_klines


def fetch_funding(symbol, start_ts, end_ts, log=None):
    """
    fundingRate 이력 가져오기 (실서버, 백테스트용)
    """
    if log:
        log("📥 펀딩 데이터 가져오는 중...\n")

    all_rows = []
    cur = start_ts
    limit = 1000

    while True:
        params = {
            "symbol": symbol,
            "startTime": cur,
            "endTime": end_ts,
            "limit": limit,
        }
        resp = requests.get(FAPI_URL + "/fapi/v1/fundingRate", params=params, timeout=10)
        if resp.status_code != 200:
            raise Exception(f"funding HTTP {resp.status_code}: {resp.text}")
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < limit:
            break
        last_ts = rows[-1]["fundingTime"]
        cur = last_ts + 1
        if cur >= end_ts:
            break

    if log:
        log(f"   → 펀딩 이벤트 개수: {len(all_rows)}\n")
    return all_rows


def build_funding_map(rows):
    """
    fundingTime -> rate 매핑
    """
    return {int(r["fundingTime"]): float(r["fundingRate"]) for r in rows}


def calc_qty(symbol, price, base_notional):
    cfg = SYMBOL_CONFIG.get(symbol, {"step": 0.001, "min_notional": 5.0})
    step = cfg["step"]
    min_notional = cfg["min_notional"]
    target_notional = max(base_notional, min_notional)
    qty = target_notional / price
    qty = math.floor(qty / step) * step
    if qty <= 0:
        return 0.0
    return qty


# ================================
# 1-2) DEMO(테스트넷)용 사인/요청 유틸
# ================================
def demo_signed_request(method: str, path: str, api_key: str, api_secret: str, params: dict):
    """
    Futures TESTNET 에 사인된 요청 보내기 (주문, 계좌조회 등)
    """
    params = dict(params) if params else {}
    params["timestamp"] = int(time.time() * 1000)
    query = urlencode(params, doseq=True)
    signature = hmac.new(api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
    headers = {"X-MBX-APIKEY": api_key}
    url = DEMO_FAPI_URL + path + "?" + query + "&signature=" + signature

    if method == "GET":
        r = requests.get(url, headers=headers, timeout=10)
    elif method == "POST":
        r = requests.post(url, headers=headers, timeout=10)
    elif method == "DELETE":
        r = requests.delete(url, headers=headers, timeout=10)
    else:
        raise ValueError(f"지원하지 않는 메서드: {method}")

    if r.status_code != 200:
        raise Exception(f"HTTP {r.status_code}: {r.text}")
    return r.json()


def demo_place_market_order(
    symbol,
    side,
    qty,
    api_key,
    api_secret,
    reduce_only=False,
    position_side=None,
    client_order_id=None,
):
    """
    Futures TESTNET에 마켓 주문 전송
    side: 'BUY' or 'SELL'
    """
    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": qty,
        "recvWindow": 5000,
    }
    if reduce_only:
        params["reduceOnly"] = "true"
    if position_side:
        params["positionSide"] = position_side
    if client_order_id:
        params["newClientOrderId"] = client_order_id
    return demo_signed_request("POST", "/fapi/v1/order", api_key, api_secret, params)


def demo_fetch_klines(symbol, interval, limit=500):
    """
    TESTNET에서 최신 kline 가져오기 (실시간 매매용)
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    r = requests.get(DEMO_FAPI_URL + "/fapi/v1/klines", params=params, timeout=10)
    if r.status_code != 200:
        raise Exception(f"demo kline HTTP {r.status_code}: {r.text}")
    return r.json()


def demo_get_htf_trend(symbol, htf_interval, htf_ema_period):
    """
    TESTNET 상위 TF에서 마지막 캔들의 추세 (close >= EMA ? True : False)
    """
    limit = max(htf_ema_period + 50, 400)
    kl = demo_fetch_klines(symbol, htf_interval, limit=limit)
    closes = [float(k[4]) for k in kl]
    emas = ema_series(closes, htf_ema_period)
    if not emas or emas[-1] is None:
        return None
    return closes[-1] >= emas[-1]


# ================================
# 2) 백테스트 로직
# ================================
def backtest_symbol(
    symbol,
    interval,
    start_str,
    end_str,
    ema_short,
    ema_long,
    tp_pct,
    sl_pct,
    base_notional,
    leverage,
    taker_fee_pct,
    slippage_pct,
    init_balance,
    use_htf,
    htf_interval,
    htf_ema_period,
    use_time_filter,
    hour_start,
    hour_end,
    log,
):
    """
    단일 심볼 백테스트
    → 결과 dict를 리턴해서 나중에 심볼별 비교에 사용
    """
    log("\n" + "=" * 40 + "\n")
    log(f"📊 현실 백테스트 시작 [{symbol}]\n")
    log(f"심볼: {symbol}, 인터벌: {interval}\n")
    log(f"기간: {start_str} ~ {end_str}\n")
    log(f"EMA: {ema_short}/{ema_long}, TP: {tp_pct*100:.2f}%, SL: {sl_pct*100:.2f}%\n")
    log(f"1회 진입금: {base_notional} USDT, 레버리지: {leverage}x\n")
    log(f"taker 수수료: {taker_fee_pct:.3f}%, 슬리피지: {slippage_pct:.3f}%\n")
    log(f"초기 잔고: {init_balance:.2f} USDT\n")
    if use_htf:
        log(f"상위TF 필터: 사용 ({htf_interval}, EMA {htf_ema_period})\n")
    else:
        log("상위TF 필터: 미사용\n")
    if use_time_filter:
        log(f"시간대 필터: 사용 (UTC {hour_start} ~ {hour_end})\n")
    else:
        log("시간대 필터: 미사용 (UTC 0~24)\n")

    start_ts = ts_from_str(start_str)
    end_ts = ts_from_str(end_str)

    # 1) 메인 타임프레임 캔들
    klines = fetch_klines(symbol, interval, start_ts, end_ts, log)
    if not klines:
        log("⚠ 캔들이 없어서 스킵됩니다.\n")
        return {
            "symbol": symbol,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "gross_pnl": 0.0,
            "funding_pnl": 0.0,
            "fee_sum": 0.0,
            "net_pnl": 0.0,
            "final_balance": init_balance,
            "max_dd": 0.0,
        }

    closes = [float(k[4]) for k in klines]
    open_times = [int(k[0]) for k in klines]

    # 2) 상위 타임프레임 캔들 + EMA 필터
    trend_flag = None
    if use_htf:
        log("📡 상위 타임프레임 데이터 가져오는 중...\n")
        htf_kl = fetch_klines(symbol, htf_interval, start_ts, end_ts, log)
        if not htf_kl:
            log("⚠ 상위TF 캔들이 없어 트렌드 필터는 비활성화됩니다.\n")

            def get_trend_for(_):
                return None
        else:
            htf_closes = [float(k[4]) for k in htf_kl]
            htf_times = [int(k[0]) for k in htf_kl]
            htf_emas = ema_series(htf_closes, htf_ema_period)
            trend_flag = []
            for i in range(len(htf_closes)):
                if htf_emas[i] is None:
                    trend_flag.append(None)
                else:
                    trend_flag.append(htf_closes[i] >= htf_emas[i])

            # 메인 타임프레임 시간에 맞게 상위 트렌드 찾기
            def get_trend_for(ts):
                idx = None
                for j in range(len(htf_times)):
                    if htf_times[j] <= ts:
                        idx = j
                    else:
                        break
                if idx is None:
                    return None
                return trend_flag[idx]
    else:

        def get_trend_for(_):
            return None

    # 3) 펀딩 데이터
    funding_rows = fetch_funding(symbol, start_ts, end_ts, log)
    funding_map = build_funding_map(funding_rows)

    fee_rate = taker_fee_pct / 100.0
    slip_rate = slippage_pct / 100.0

    balance = init_balance
    max_balance = init_balance
    max_dd = 0.0

    position_side = None  # "LONG" / "SHORT"
    entry_price = 0.0
    qty = 0.0

    gross_pnl_sum = 0.0
    fee_sum = 0.0
    funding_pnl_sum = 0.0

    wins = 0
    losses = 0
    trades = 0

    short_ema = ema_series(closes, ema_short)
    long_ema = ema_series(closes, ema_long)

    prev_short = None
    prev_long = None

    # 펀딩을 kline에 반영하기 위해 fundingTime 근처 캔들에서 처리
    funding_times_sorted = sorted(funding_map.keys())
    funding_idx = 0

    log("\n📊 백테스트 계산 중...\n")

    for i in range(len(closes)):
        price_open = float(klines[i][1])
        price_close = closes[i]
        ts = open_times[i]

        # 시간대 필터
        if use_time_filter:
            hour_utc = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).hour
            if not (hour_start <= hour_utc < hour_end):
                time_filter_block_entry = True
            else:
                time_filter_block_entry = False
        else:
            time_filter_block_entry = False

        # 상위 TF 필터
        trend = get_trend_for(ts)
        # EMA 값
        s = short_ema[i]
        l = long_ema[i]

        # 펀딩 처리 (포지션 있을 때만)
        while funding_idx < len(funding_times_sorted) and funding_times_sorted[funding_idx] <= ts:
            f_ts = funding_times_sorted[funding_idx]
            rate = funding_map[f_ts]
            if position_side is not None and qty > 0:
                notional = entry_price * qty
                if position_side == "LONG":
                    pnl_f = -notional * rate
                else:
                    pnl_f = notional * rate
                balance += pnl_f
                funding_pnl_sum += pnl_f
                log(f"[펀딩] ts={f_ts}, rate={rate:.6f}, pnl={pnl_f:.4f}, balance={balance:.4f}\n")
            funding_idx += 1

        # EMA 교차 시그널 계산
        signal = None
        if s is not None and l is not None:
            if prev_short is not None and prev_long is not None:
                # 상향 교차 -> LONG
                if prev_short <= prev_long and s > l:
                    signal = "LONG"
                # 하향 교차 -> SHORT
                elif prev_short >= prev_long and s < l:
                    signal = "SHORT"
        prev_short = s
        prev_long = l

        # 상위 TF 트렌드 필터 적용
        if use_htf and trend is not None:
            if trend:  # 상승 추세
                if signal == "SHORT":
                    signal = None
            else:  # 하락 추세
                if signal == "LONG":
                    signal = None

        # 포지션 관리
        if position_side is None:
            # 진입 없음 상태
            if signal and not time_filter_block_entry:
                # 진입
                q = calc_qty(symbol, price_open, base_notional)
                if q <= 0:
                    log(f"[경고] idx={i}, 수량이 0 이하라 진입 안함.\n")
                    continue

                side = signal
                # 슬리피지 반영한 실제 체결가
                if side == "LONG":
                    fill_price = price_open * (1 + slip_rate)
                else:
                    fill_price = price_open * (1 - slip_rate)

                notional = fill_price * q
                fee = notional * fee_rate
                balance -= fee
                fee_sum += fee

                entry_price = fill_price
                qty = q
                position_side = side
                trades += 1

                log(
                    f"[진입] idx={i}, side={side}, price={fill_price:.2f}, "
                    f"qty={qty}, fee={fee:.4f}, balance={balance:.4f}\n"
                )
        else:
            # 포지션 보유 상태: TP/SL or 반대 시그널
            side = position_side
            ep = entry_price

            # 현재가 + 슬리피지
            if side == "LONG":
                tp_price = ep * (1 + tp_pct)
                sl_price = ep * (1 - sl_pct)
                price_for_calc = price_close
                if price_for_calc >= tp_price or (signal == "SHORT"):
                    exit_price = price_close * (1 - slip_rate)
                    reason = "TP" if price_for_calc >= tp_price else "Reverse"
                elif price_for_calc <= sl_price:
                    exit_price = price_close * (1 - slip_rate)
                    reason = "SL"
                else:
                    reason = None
            else:  # SHORT
                tp_price = ep * (1 - tp_pct)
                sl_price = ep * (1 + sl_pct)
                price_for_calc = price_close
                if price_for_calc <= tp_price or (signal == "LONG"):
                    exit_price = price_close * (1 + slip_rate)
                    reason = "TP" if price_for_calc <= tp_price else "Reverse"
                elif price_for_calc >= sl_price:
                    exit_price = price_close * (1 + slip_rate)
                    reason = "SL"
                else:
                    reason = None

            if reason is not None:
                notional_entry = ep * qty
                notional_exit = exit_price * qty
                # 가격차익
                if side == "LONG":
                    gross = notional_exit - notional_entry
                else:
                    gross = notional_entry - notional_exit

                fee_exit = notional_exit * fee_rate
                fee_sum += fee_exit
                balance += gross - fee_exit
                gross_pnl_sum += gross

                if gross >= 0:
                    wins += 1
                else:
                    losses += 1

                log(
                    f"[청산] idx={i}, side={side}, entry={ep:.2f}, exit={exit_price:.2f}, "
                    f"reason={reason}, gross={gross:.4f}, fee={fee_exit:.4f}, balance={balance:.4f}\n"
                )

                # max DD 업데이트
                if balance > max_balance:
                    max_balance = balance
                dd = (max_balance - balance) / max_balance if max_balance > 0 else 0
                if dd > max_dd:
                    max_dd = dd

                # 포지션 리셋
                position_side = None
                qty = 0.0
                entry_price = 0.0

    net_pnl = gross_pnl_sum + funding_pnl_sum - fee_sum
    win_rate = (wins / trades * 100.0) if trades > 0 else 0.0

    log("\n===== 백테스트 결과 요약 =====\n")
    log(f"[{symbol}]\n")
    log(f"총 거래 수: {trades}\n")
    log(f"승: {wins}, 패: {losses}, 승률: {win_rate:.2f}%\n")
    log(f"가격차익 합(Gross): {gross_pnl_sum:.4f} USDT\n")
    log(f"펀딩 PnL 합:         {funding_pnl_sum:.4f} USDT\n")
    log(f"수수료 합:          {fee_sum:.4f} USDT\n")
    log(f"총 순손익(Net):      {net_pnl:.4f} USDT\n")
    log(f"최초 잔고:          {init_balance:.4f} USDT\n")
    log(f"최종 잔고:          {balance:.4f} USDT\n")
    log(f"최대 드로우다운:    {max_dd*100:.2f}%\n")
    log("========================================\n")

    # 요약 결과를 dict로 리턴
    return {
        "symbol": symbol,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "gross_pnl": gross_pnl_sum,
        "funding_pnl": funding_pnl_sum,
        "fee_sum": fee_sum,
        "net_pnl": net_pnl,
        "final_balance": balance,
        "max_dd": max_dd,
    }


# ================================
# 3) Tkinter GUI
# ================================
class BacktestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Binance Futures 멀티 심볼 백테스터 + DEMO 자동매매")
        self.root.geometry("1000x760")

        # DEMO 자동매매 상태
        self.demo_running = False
        self.demo_thread = None
        self.demo_states = {}  # 심볼별 포지션 상태
        self.demo_daily_loss = 0.0
        self.demo_daily_reset_date = datetime.utcnow().date()
        self.demo_paused_by_loss = False

        # 오늘(UTC) 기준으로 기본 날짜 계산
        today_utc = datetime.utcnow().date()
        default_end = today_utc
        default_start = today_utc - timedelta(days=30)  # 한 달 = 30일 기준

        # 상단 설정 프레임
        cfg = tk.LabelFrame(root, text="백테스트 / 전략 설정", padx=5, pady=5)
        cfg.pack(fill="x", padx=10, pady=5)

        # 0행: 심볼 목록 / 인터벌
        tk.Label(cfg, text="심볼들(콤마 구분):", width=18, anchor="e").grid(
            row=0, column=0, padx=5, pady=3
        )
        self.symbols_entry = tk.Entry(cfg, width=40)
        self.symbols_entry.insert(0, "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT")
        self.symbols_entry.grid(row=0, column=1, sticky="w", padx=5)

        tk.Label(cfg, text="인터벌:", width=10, anchor="e").grid(
            row=0, column=2, padx=5, pady=3
        )
        self.interval_entry = tk.Entry(cfg, width=8)
        self.interval_entry.insert(0, "5m")
        self.interval_entry.grid(row=0, column=3, sticky="w")

        # 1행: 시작 날짜 (달력)
        tk.Label(cfg, text="시작 날짜(UTC):", width=18, anchor="e").grid(
            row=1, column=0, padx=5, pady=3
        )
        self.start_cal = DateEntry(
            cfg,
            width=12,
            year=default_start.year,
            month=default_start.month,
            day=default_start.day,
            date_pattern="yyyy-mm-dd",
        )
        self.start_cal.grid(row=1, column=1, sticky="w", padx=5)

        # 2행: 종료 날짜 (달력)
        tk.Label(cfg, text="종료 날짜(UTC):", width=18, anchor="e").grid(
            row=2, column=0, padx=5, pady=3
        )
        self.end_cal = DateEntry(
            cfg,
            width=12,
            year=default_end.year,
            month=default_end.month,
            day=default_end.day,
            date_pattern="yyyy-mm-dd",
        )
        self.end_cal.grid(row=2, column=1, sticky="w", padx=5)

        # 3행: EMA / TP/SL
        tk.Label(cfg, text="EMA 단기/장기:", width=18, anchor="e").grid(
            row=3, column=0, padx=5, pady=3
        )
        self.ema_short_entry = tk.Entry(cfg, width=6)
        self.ema_short_entry.insert(0, "20")
        self.ema_short_entry.grid(row=3, column=1, sticky="w")
        self.ema_long_entry = tk.Entry(cfg, width=6)
        self.ema_long_entry.insert(0, "80")
        self.ema_long_entry.grid(row=3, column=1, padx=60, sticky="w")

        tk.Label(cfg, text="TP/SL(%):", width=10, anchor="e").grid(
            row=3, column=2, padx=5, pady=3
        )
        self.tp_entry = tk.Entry(cfg, width=6)
        self.tp_entry.insert(0, "0.8")
        self.tp_entry.grid(row=3, column=3, sticky="w")
        self.sl_entry = tk.Entry(cfg, width=6)
        self.sl_entry.insert(0, "0.4")
        self.sl_entry.grid(row=3, column=3, padx=60, sticky="w")

        # 4행: 자금 / 레버리지 / 수수료
        tk.Label(cfg, text="1회 진입금(USDT):", width=18, anchor="e").grid(
            row=4, column=0, padx=5, pady=3
        )
        self.base_notional_entry = tk.Entry(cfg, width=10)
        self.base_notional_entry.insert(0, "50")
        self.base_notional_entry.grid(row=4, column=1, sticky="w")

        tk.Label(cfg, text="레버리지(x):", width=10, anchor="e").grid(
            row=4, column=2, padx=5, pady=3
        )
        self.lev_entry = tk.Entry(cfg, width=6)
        self.lev_entry.insert(0, "5")
        self.lev_entry.grid(row=4, column=3, sticky="w")

        tk.Label(cfg, text="수수료/슬리피지(%):", width=18, anchor="e").grid(
            row=5, column=0, padx=5, pady=3
        )
        self.fee_entry = tk.Entry(cfg, width=8)
        self.fee_entry.insert(0, "0.04")
        self.fee_entry.grid(row=5, column=1, sticky="w")
        self.slip_entry = tk.Entry(cfg, width=8)
        self.slip_entry.insert(0, "0.02")
        self.slip_entry.grid(row=5, column=1, padx=60, sticky="w")

        tk.Label(cfg, text="초기 잔고(각 심볼):", width=18, anchor="e").grid(
            row=5, column=2, padx=5, pady=3
        )
        self.init_bal_entry = tk.Entry(cfg, width=10)
        self.init_bal_entry.insert(0, "100")
        self.init_bal_entry.grid(row=5, column=3, sticky="w")

        # 6행: 상위 TF
        self.use_htf_var = tk.IntVar(value=1)
        chk_htf = tk.Checkbutton(cfg, text="상위 TF EMA 필터 사용", variable=self.use_htf_var)
        chk_htf.grid(row=6, column=0, padx=5, pady=3, sticky="e")

        tk.Label(cfg, text="상위TF 인터벌:", width=14, anchor="e").grid(
            row=6, column=1, padx=5, pady=3, sticky="w"
        )
        self.htf_interval_entry = tk.Entry(cfg, width=8)
        self.htf_interval_entry.insert(0, "1h")
        self.htf_interval_entry.grid(row=6, column=1, padx=110, sticky="w")

        tk.Label(cfg, text="상위TF EMA 기간:", width=14, anchor="e").grid(
            row=6, column=2, padx=5, pady=3
        )
        self.htf_ema_entry = tk.Entry(cfg, width=8)
        self.htf_ema_entry.insert(0, "200")
        self.htf_ema_entry.grid(row=6, column=3, sticky="w")

        # 7행: 시간대 필터
        self.use_time_var = tk.IntVar(value=0)
        chk_time = tk.Checkbutton(cfg, text="시간대 필터 사용(UTC)", variable=self.use_time_var)
        chk_time.grid(row=7, column=0, padx=5, pady=3, sticky="e")

        tk.Label(cfg, text="시작 시간:", width=14, anchor="e").grid(
            row=7, column=1, padx=5, pady=3, sticky="w"
        )
        self.hour_start_entry = tk.Entry(cfg, width=6)
        self.hour_start_entry.insert(0, "0")
        self.hour_start_entry.grid(row=7, column=1, padx=90, sticky="w")

        tk.Label(cfg, text="종료 시간:", width=10, anchor="e").grid(
            row=7, column=2, padx=5, pady=3
        )
        self.hour_end_entry = tk.Entry(cfg, width=6)
        self.hour_end_entry.insert(0, "24")
        self.hour_end_entry.grid(row=7, column=3, sticky="w")

        # 8행: DEMO API Key/Secret
        tk.Label(cfg, text="DEMO API Key:", width=18, anchor="e").grid(
            row=8, column=0, padx=5, pady=3
        )
        self.api_key_entry = tk.Entry(cfg, width=40)
        self.api_key_entry.grid(row=8, column=1, columnspan=3, sticky="w", padx=5)

        tk.Label(cfg, text="DEMO Secret:", width=18, anchor="e").grid(
            row=9, column=0, padx=5, pady=3
        )
        self.api_secret_entry = tk.Entry(cfg, width=40, show="*")
        self.api_secret_entry.grid(row=9, column=1, columnspan=3, sticky="w", padx=5)

        # 10행: 리스크/체크 설정
        tk.Label(cfg, text="일일 손실 한도(USDT):", width=18, anchor="e").grid(
            row=10, column=0, padx=5, pady=3
        )
        self.daily_loss_limit_entry = tk.Entry(cfg, width=10)
        self.daily_loss_limit_entry.insert(0, "100")
        self.daily_loss_limit_entry.grid(row=10, column=1, sticky="w")

        tk.Label(cfg, text="최대 1회 진입 노출:", width=18, anchor="e").grid(
            row=10, column=2, padx=5, pady=3
        )
        self.max_trade_notional_entry = tk.Entry(cfg, width=10)
        self.max_trade_notional_entry.insert(0, "150")
        self.max_trade_notional_entry.grid(row=10, column=3, sticky="w")

        tk.Label(cfg, text="포트폴리오 노출 한도:", width=18, anchor="e").grid(
            row=11, column=0, padx=5, pady=3
        )
        self.max_portfolio_notional_entry = tk.Entry(cfg, width=10)
        self.max_portfolio_notional_entry.insert(0, "300")
        self.max_portfolio_notional_entry.grid(row=11, column=1, sticky="w")

        tk.Label(cfg, text="ATR 필터(period / %):", width=18, anchor="e").grid(
            row=11, column=2, padx=5, pady=3
        )
        self.atr_period_entry = tk.Entry(cfg, width=6)
        self.atr_period_entry.insert(0, "14")
        self.atr_period_entry.grid(row=11, column=3, sticky="w")
        self.atr_threshold_entry = tk.Entry(cfg, width=6)
        self.atr_threshold_entry.insert(0, "1.5")
        self.atr_threshold_entry.grid(row=11, column=3, padx=60, sticky="w")

        tk.Label(cfg, text="허용 슬리피지%(체결)", width=18, anchor="e").grid(
            row=12, column=0, padx=5, pady=3
        )
        self.slippage_limit_entry = tk.Entry(cfg, width=10)
        self.slippage_limit_entry.insert(0, "0.5")
        self.slippage_limit_entry.grid(row=12, column=1, sticky="w")

        self.hedge_mode_var = tk.IntVar(value=0)
        chk_hedge = tk.Checkbutton(cfg, text="Hedge 모드(positionSide 사용)", variable=self.hedge_mode_var)
        chk_hedge.grid(row=12, column=2, padx=5, pady=3, sticky="w")

        # 버튼들
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)

        self.start_btn = tk.Button(
            btn_frame, text="백테스트 시작", width=18, command=self.start_backtest
        )
        self.start_btn.pack(side="left", padx=5)

        self.demo_start_btn = tk.Button(
            btn_frame, text="DEMO 자동매매 시작", width=18, command=self.start_demo_trading
        )
        self.demo_start_btn.pack(side="left", padx=5)

        self.demo_stop_btn = tk.Button(
            btn_frame, text="DEMO 자동매매 정지", width=18, command=self.stop_demo_trading
        )
        self.demo_stop_btn.pack(side="left", padx=5)

        # 로그 영역
        log_frame = tk.LabelFrame(root, text="로그 / 결과", padx=5, pady=5)
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.log_area = scrolledtext.ScrolledText(log_frame, state="disabled")
        self.log_area.pack(fill="both", expand=True)

    def append_log(self, text: str):
        self.log_area.configure(state="normal")
        self.log_area.insert(tk.END, text)
        self.log_area.see(tk.END)
        self.log_area.configure(state="disabled")
        self.root.update_idletasks()

    # ------------------------------
    # 백테스트 실행
    # ------------------------------
    def start_backtest(self):
        try:
            symbols_raw = self.symbols_entry.get().strip()
            symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
            if not symbols:
                messagebox.showwarning("입력 오류", "심볼을 하나 이상 입력해 주세요.")
                return

            interval = self.interval_entry.get().strip()

            # 달력에서 날짜 가져오기
            s_date = self.start_cal.get_date()  # datetime.date
            e_date = self.end_cal.get_date()

            start_str = f"{s_date.year:04d}-{s_date.month:02d}-{s_date.day:02d} 00:00:00"
            end_str = f"{e_date.year:04d}-{e_date.month:02d}-{e_date.day:02d} 23:59:59"

            # 날짜 유효성 체크
            ds = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
            de = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
            if ds >= de:
                messagebox.showwarning("날짜 오류", "시작 날짜는 종료 날짜보다 이전이어야 합니다.")
                return

            ema_short = int(self.ema_short_entry.get().strip())
            ema_long = int(self.ema_long_entry.get().strip())
            tp_pct = float(self.tp_entry.get().strip()) / 100.0
            sl_pct = float(self.sl_entry.get().strip()) / 100.0

            base_notional = float(self.base_notional_entry.get().strip())
            leverage = float(self.lev_entry.get().strip())
            taker_fee_pct = float(self.fee_entry.get().strip())
            slippage_pct = float(self.slip_entry.get().strip())
            init_balance = float(self.init_bal_entry.get().strip())

            use_htf = bool(self.use_htf_var.get())
            htf_interval = self.htf_interval_entry.get().strip()
            htf_ema_period = int(self.htf_ema_entry.get().strip())

            use_time_filter = bool(self.use_time_var.get())
            hour_start = int(self.hour_start_entry.get().strip())
            hour_end = int(self.hour_end_entry.get().strip())

        except Exception as e:
            messagebox.showerror("입력 오류", f"입력 값을 확인해 주세요.\n{e}")
            return

        # 로그 초기화
        self.log_area.configure(state="normal")
        self.log_area.delete("1.0", tk.END)
        self.log_area.configure(state="disabled")

        self.append_log("🚀 멀티 심볼 EMA 전략 백테스트 시작\n")
        self.append_log(f"대상 심볼: {', '.join(symbols)}\n")
        self.append_log(f"기간(UTC): {start_str} ~ {end_str}\n")

        summaries = []

        # 심볼별로 따로 백테스트 수행
        for sym in symbols:
            try:
                result = backtest_symbol(
                    symbol=sym,
                    interval=interval,
                    start_str=start_str,
                    end_str=end_str,
                    ema_short=ema_short,
                    ema_long=ema_long,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                    base_notional=base_notional,
                    leverage=leverage,
                    taker_fee_pct=taker_fee_pct,
                    slippage_pct=slippage_pct,
                    init_balance=init_balance,
                    use_htf=use_htf,
                    htf_interval=htf_interval,
                    htf_ema_period=htf_ema_period,
                    use_time_filter=use_time_filter,
                    hour_start=hour_start,
                    hour_end=hour_end,
                    log=self.append_log,
                )
                summaries.append(result)
            except Exception as e:
                self.append_log(f"❌ [{sym}] 백테스트 중 에러: {e}\n")

        # ===== 심볼별 결과 비교 요약 =====
        if summaries:
            # Net PnL 기준 정렬
            summaries_sorted_net = sorted(
                summaries, key=lambda x: x["net_pnl"], reverse=True
            )

            self.append_log("\n\n🏁 심볼별 성과 요약 (Net PnL 기준 내림차순)\n")
            self.append_log(
                "-------------------------------------------------------------------------------\n"
            )
            header = (
                f"{'순위':>4} {'심볼':<8} {'NetPnL':>12} {'승률%':>8} "
                f"{'거래수':>8} {'최종잔고':>12} {'MDD%':>8}\n"
            )
            self.append_log(header)
            self.append_log(
                "-------------------------------------------------------------------------------\n"
            )

            for idx, r in enumerate(summaries_sorted_net, start=1):
                line = (
                    f"{idx:>4} "
                    f"{r['symbol']:<8} "
                    f"{r['net_pnl']:>12.4f} "
                    f"{r['win_rate']:>8.2f} "
                    f"{r['trades']:>8d} "
                    f"{r['final_balance']:>12.4f} "
                    f"{(r['max_dd']*100):>8.2f}\n"
                )
                self.append_log(line)

            self.append_log(
                "-------------------------------------------------------------------------------\n"
            )

            best = summaries_sorted_net[0]
            self.append_log(
                f"\n🔥 이번 세팅에서 Net PnL 기준 1등 심볼: {best['symbol']} "
                f"(Net {best['net_pnl']:.4f} USDT, 승률 {best['win_rate']:.2f}%, "
                f"최종잔고 {best['final_balance']:.4f} USDT)\n"
            )

        self.append_log("\n✅ 모든 심볼 백테스트 완료.\n")

    # ------------------------------
    # DEMO 자동매매 시작/중지
    # ------------------------------
    def start_demo_trading(self):
        if self.demo_running:
            messagebox.showinfo("알림", "DEMO 자동매매가 이미 실행 중입니다.")
            return

        api_key = self.api_key_entry.get().strip()
        api_secret = self.api_secret_entry.get().strip()
        if not api_key or not api_secret:
            messagebox.showwarning("API 오류", "DEMO API Key와 Secret을 입력해 주세요.")
            return

        try:
            symbols_raw = self.symbols_entry.get().strip()
            symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]
            if not symbols:
                messagebox.showwarning("입력 오류", "심볼을 하나 이상 입력해 주세요.")
                return

            interval = self.interval_entry.get().strip()
            ema_short = int(self.ema_short_entry.get().strip())
            ema_long = int(self.ema_long_entry.get().strip())
            tp_pct = float(self.tp_entry.get().strip()) / 100.0
            sl_pct = float(self.sl_entry.get().strip()) / 100.0
            base_notional = float(self.base_notional_entry.get().strip())

            use_htf = bool(self.use_htf_var.get())
            htf_interval = self.htf_interval_entry.get().strip()
            htf_ema_period = int(self.htf_ema_entry.get().strip())

            use_time_filter = bool(self.use_time_var.get())
            hour_start = int(self.hour_start_entry.get().strip())
            hour_end = int(self.hour_end_entry.get().strip())

            daily_loss_limit = float(self.daily_loss_limit_entry.get().strip())
            max_trade_notional = float(self.max_trade_notional_entry.get().strip())
            max_portfolio_notional = float(self.max_portfolio_notional_entry.get().strip())
            atr_period = int(self.atr_period_entry.get().strip())
            atr_threshold_pct = float(self.atr_threshold_entry.get().strip())
            slippage_limit_pct = float(self.slippage_limit_entry.get().strip())
            hedge_mode = bool(self.hedge_mode_var.get())
        except Exception as e:
            messagebox.showerror("입력 오류", f"입력 값을 확인해 주세요.\n{e}")
            return

        # 상태 초기화
        self.demo_states = {
            sym: {
                "position_side": None,
                "qty": 0.0,
                "entry_price": 0.0,
                "last_kline_time": None,
                "estimated_pnl": 0.0,
            }
            for sym in symbols
        }
        self.demo_daily_loss = 0.0
        self.demo_daily_reset_date = datetime.utcnow().date()
        self.demo_paused_by_loss = False

        self.demo_cfg = {
            "symbols": symbols,
            "interval": interval,
            "ema_short": ema_short,
            "ema_long": ema_long,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "base_notional": base_notional,
            "use_htf": use_htf,
            "htf_interval": htf_interval,
            "htf_ema_period": htf_ema_period,
            "use_time_filter": use_time_filter,
            "hour_start": hour_start,
            "hour_end": hour_end,
            "api_key": api_key,
            "api_secret": api_secret,
            "daily_loss_limit": daily_loss_limit,
            "max_trade_notional": max_trade_notional,
            "max_portfolio_notional": max_portfolio_notional,
            "atr_period": atr_period,
            "atr_threshold_pct": atr_threshold_pct,
            "slippage_limit_pct": slippage_limit_pct,
            "hedge_mode": hedge_mode,
        }

        self.demo_running = True

        self.append_log(
            "\n🚨 DEMO 자동매매 시작 (Binance Futures TESTNET - https://demo.binance.com)\n"
        )
        self.append_log(
            "⚠️ 반드시 테스트넷(Futures Testnet)용 API KEY / SECRET만 사용하세요. "
            "실계좌 키 절대 넣지 마세요.\n"
        )

        # 백그라운드 스레드로 실행
        self.demo_thread = threading.Thread(
            target=self.demo_loop, args=(), daemon=True
        )
        self.demo_thread.start()

    def stop_demo_trading(self):
        if not self.demo_running:
            messagebox.showinfo("알림", "DEMO 자동매매가 실행 중이 아닙니다.")
            return
        self.demo_running = False
        self.append_log("\n🛑 DEMO 자동매매 중지 요청.\n")

    def _reset_daily_loss_if_needed(self):
        today = datetime.utcnow().date()
        if today != self.demo_daily_reset_date:
            self.demo_daily_reset_date = today
            self.demo_daily_loss = 0.0
            self.demo_paused_by_loss = False
            self.append_log("🌅 새 UTC 일자 시작: 일일 손실 한도 리셋\n")

    def demo_loop(self):
        """
        DEMO 자동매매 메인 루프
        일정 주기로 각 심볼에 대해 최신 캔들 받아서
        EMA 교차 → 시그널 → TP/SL/반대시그널에 따라 시장가 주문
        + 3초마다 상태 로그 출력
        """
        cfg = self.demo_cfg
        long_period = max(cfg["ema_short"], cfg["ema_long"])
        loop_cnt = 0

        while self.demo_running:
            loop_cnt += 1
            self._reset_daily_loss_if_needed()

            # 일일 손실 한도 초과 시 거래 중단
            if self.demo_paused_by_loss:
                self.append_log(
                    f"[DEMO 루프 {loop_cnt}] 일일 손실 한도 초과로 거래 일시 중지 중\n"
                )
            else:
                # 각 심볼 한 번씩 처리
                for sym in cfg["symbols"]:
                    try:
                        self.demo_run_symbol_once(sym, long_period)
                    except Exception as e:
                        self.append_log(f"❌ [DEMO] {sym} 처리 중 오류: {e}\n")

            # --- 상태 로그 출력 (3초마다) ---
            status_list = []
            for sym in cfg["symbols"]:
                st = self.demo_states.get(sym, {})
                side = st.get("position_side")
                qty = st.get("qty", 0.0)
                pnl = st.get("estimated_pnl", 0.0)
                if side is None:
                    status_list.append(f"{sym}: FLAT (PnL≈{pnl:.2f})")
                else:
                    status_list.append(f"{sym}: {side} {qty} (PnL≈{pnl:.2f})")
            status_str = " | ".join(status_list)

            now_utc = datetime.utcnow().strftime("%H:%M:%S")
            self.append_log(
                f"[DEMO 루프 {loop_cnt}] {now_utc} UTC / {status_str} / 일일PnL={self.demo_daily_loss:.2f}\n"
            )

            # 3초 간격으로 동작 (중간에 정지 누르면 바로 탈출)
            for _ in range(3):
                if not self.demo_running:
                    break
                time.sleep(1)

        self.append_log("🧹 DEMO 자동매매 루프 종료.\n")

    def _estimate_portfolio_notional(self):
        total = 0.0
        for st in self.demo_states.values():
            if st.get("position_side") and st.get("qty", 0.0) > 0:
                total += st.get("entry_price", 0.0) * st.get("qty", 0.0)
        return total

    def _should_block_by_atr(self, symbol, highs, lows, closes):
        cfg = self.demo_cfg
        atrs = atr_series(highs, lows, closes, cfg["atr_period"])
        last_atr = atrs[-1]
        last_close = closes[-1]
        if last_atr is None:
            return False
        atr_pct = (last_atr / last_close) * 100 if last_close else 0
        if atr_pct >= cfg["atr_threshold_pct"]:
            self.append_log(
                f"[DEMO] {symbol} 변동성 {atr_pct:.2f}% ≥ 설정 {cfg['atr_threshold_pct']:.2f}% → 진입 스킵\n"
            )
            return True
        return False

    def _update_loss_and_check_limit(self, pnl_delta: float):
        self.demo_daily_loss += pnl_delta
        cfg = self.demo_cfg
        if self.demo_daily_loss <= -cfg["daily_loss_limit"]:
            self.demo_paused_by_loss = True
            self.append_log(
                f"🚫 일일 손실 한도 {-cfg['daily_loss_limit']:.2f} USDT 초과 → 당일 거래 중지\n"
            )

    def _generate_client_order_id(self, prefix: str) -> str:
        return f"{prefix}-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"

    def demo_run_symbol_once(self, symbol, long_period):
        """
        단일 심볼에 대해 1회 시그널 체크 & 필요 시 주문
        """
        cfg = self.demo_cfg
        state = self.demo_states[symbol]
        api_key = cfg["api_key"]
        api_secret = cfg["api_secret"]

        interval = cfg["interval"]
        ema_short = cfg["ema_short"]
        ema_long = cfg["ema_long"]
        tp_pct = cfg["tp_pct"]
        sl_pct = cfg["sl_pct"]
        base_notional = cfg["base_notional"]

        use_htf = cfg["use_htf"]
        htf_interval = cfg["htf_interval"]
        htf_ema_period = cfg["htf_ema_period"]
        use_time_filter = cfg["use_time_filter"]
        hour_start = cfg["hour_start"]
        hour_end = cfg["hour_end"]

        atr_period = cfg["atr_period"]

        # 최신 캔들 가져오기 (TESTNET)
        limit = max(long_period + 5, atr_period + 50, 120)
        kl = demo_fetch_klines(symbol, interval, limit=limit)
        if len(kl) < max(long_period + 2, atr_period + 2):
            self.append_log(f"[DEMO] {symbol} 캔들 개수 부족으로 스킵\n")
            return

        closes = [float(k[4]) for k in kl]
        highs = [float(k[2]) for k in kl]
        lows = [float(k[3]) for k in kl]
        open_times = [int(k[0]) for k in kl]
        close_times = [int(k[6]) for k in kl]

        if any(math.isnan(v) for v in closes):
            self.append_log(f"[DEMO] {symbol} 데이터에 NaN이 있어 스킵\n")
            return

        if self._should_block_by_atr(symbol, highs, lows, closes):
            return

        short_emas = ema_series(closes, ema_short)
        long_emas = ema_series(closes, ema_long)

        # 마지막 캔들 기준 (실시간이라 완전히 닫힌게 아닐 수 있지만, 간단히 사용)
        idx = len(closes) - 1
        prev_idx = idx - 1

        if (
            short_emas[idx] is None
            or long_emas[idx] is None
            or short_emas[prev_idx] is None
            or long_emas[prev_idx] is None
        ):
            return

        kline_time = open_times[idx]
        # 같은 캔들을 이미 처리했으면 스킵
        if state["last_kline_time"] == kline_time:
            return
        state["last_kline_time"] = kline_time

        # 시간대 필터 (UTC 기준)
        if use_time_filter:
            hour_utc = datetime.fromtimestamp(close_times[idx] / 1000, tz=timezone.utc).hour
            if not (hour_start <= hour_utc < hour_end):
                time_filter_block_entry = True
            else:
                time_filter_block_entry = False
        else:
            time_filter_block_entry = False

        s_prev = short_emas[prev_idx]
        l_prev = long_emas[prev_idx]
        s = short_emas[idx]
        l = long_emas[idx]

        # EMA 교차 시그널 계산
        signal = None
        if s_prev is not None and l_prev is not None:
            if s_prev <= l_prev and s > l:
                signal = "LONG"
            elif s_prev >= l_prev and s < l:
                signal = "SHORT"

        # 상위TF 추세 필터
        if use_htf:
            trend = demo_get_htf_trend(symbol, htf_interval, htf_ema_period)
            if trend is not None:
                if trend:  # 상승 추세
                    if signal == "SHORT":
                        signal = None
                else:  # 하락 추세
                    if signal == "LONG":
                        signal = None

        last_price = closes[idx]

        # 포지션 상태
        pos_side = state["position_side"]
        qty = state["qty"]
        ep = state["entry_price"]

        # 포트폴리오 노출 체크
        total_notional = self._estimate_portfolio_notional()

        # 포지션 없는 상태: 새 진입
        if pos_side is None:
            if signal and not time_filter_block_entry and not self.demo_paused_by_loss:
                q = calc_qty(symbol, last_price, base_notional)
                if q <= 0:
                    self.append_log(f"[DEMO 경고] {symbol}: 수량 0 이하라 진입 생략.\n")
                    return

                trade_notional = last_price * q
                if trade_notional > cfg["max_trade_notional"]:
                    self.append_log(
                        f"[DEMO] {symbol} 1회 노출 {trade_notional:.2f} > 설정 {cfg['max_trade_notional']:.2f} → 진입 스킵\n"
                    )
                    return
                if total_notional + trade_notional > cfg["max_portfolio_notional"]:
                    self.append_log(
                        f"[DEMO] 포트폴리오 노출 초과 예상({total_notional + trade_notional:.2f} > {cfg['max_portfolio_notional']:.2f}) → 진입 스킵\n"
                    )
                    return

                side = "BUY" if signal == "LONG" else "SELL"
                try:
                    resp = demo_place_market_order(
                        symbol,
                        side,
                        q,
                        api_key,
                        api_secret,
                        position_side="LONG" if (cfg["hedge_mode"] and signal == "LONG") else "SHORT" if cfg["hedge_mode"] else None,
                        client_order_id=self._generate_client_order_id("entry"),
                    )
                except Exception as e:
                    self.append_log(f"❌ [DEMO 주문 실패] {symbol} {side} {q}: {e}\n")
                    return

                filled_price = float(resp.get("avgPrice") or resp.get("price", last_price))
                slippage_real = abs(filled_price - last_price) / last_price * 100 if last_price else 0
                if slippage_real > cfg["slippage_limit_pct"]:
                    self.append_log(
                        f"🚧 [DEMO] {symbol} 슬리피지 {slippage_real:.2f}%가 한도 {cfg['slippage_limit_pct']:.2f}% 초과 → 즉시 청산\n"
                    )
                    try:
                        demo_place_market_order(
                            symbol,
                            "SELL" if side == "BUY" else "BUY",
                            q,
                            api_key,
                            api_secret,
                            reduce_only=True,
                            position_side="LONG" if (cfg["hedge_mode"] and signal == "LONG") else "SHORT" if cfg["hedge_mode"] else None,
                            client_order_id=self._generate_client_order_id("slip-close"),
                        )
                    except Exception as close_err:
                        self.append_log(f"❌ [DEMO] 슬리피지 청산 실패: {close_err}\n")
                    return

                state["position_side"] = signal  # LONG/SHORT
                state["qty"] = q
                state["entry_price"] = filled_price
                state["estimated_pnl"] = 0.0

                self.append_log(
                    f"[DEMO 진입] {symbol} side={signal}, price≈{filled_price:.2f}, qty={q}\n"
                )
            return

        # 포지션 보유 상태: TP/SL/반대 시그널 체크
        if pos_side == "LONG":
            tp_price = ep * (1 + tp_pct)
            sl_price = ep * (1 - sl_pct)
            need_close = False
            reason = None

            if last_price >= tp_price:
                need_close = True
                reason = "TP"
            elif last_price <= sl_price:
                need_close = True
                reason = "SL"
            elif signal == "SHORT":
                need_close = True
                reason = "Reverse"

            if need_close:
                side = "SELL"  # LONG 포지션 청산은 SELL
                try:
                    resp = demo_place_market_order(
                        symbol,
                        side,
                        qty,
                        api_key,
                        api_secret,
                        reduce_only=True,
                        position_side="LONG" if cfg["hedge_mode"] else None,
                        client_order_id=self._generate_client_order_id("close"),
                    )
                except Exception as e:
                    self.append_log(
                        f"❌ [DEMO 청산 실패] {symbol} {side} {qty}: {e}\n"
                    )
                    return

                exit_price = float(resp.get("avgPrice") or resp.get("price", last_price))
                pnl = (exit_price - ep) * qty
                state["estimated_pnl"] += pnl
                self._update_loss_and_check_limit(pnl)

                self.append_log(
                    f"[DEMO 청산] {symbol} LONG ep={ep:.2f} -> fill≈{exit_price:.2f}, "
                    f"reason={reason}, qty={qty}, PnL≈{pnl:.4f}\n"
                )
                state["position_side"] = None
                state["qty"] = 0.0
                state["entry_price"] = 0.0

                # Reverse면 바로 반대 방향 진입 (옵션)
                if reason == "Reverse" and not time_filter_block_entry and not self.demo_paused_by_loss:
                    new_signal = "SHORT"
                    q2 = calc_qty(symbol, last_price, base_notional)
                    if q2 > 0:
                        try:
                            resp2 = demo_place_market_order(
                                symbol,
                                "SELL",
                                q2,
                                api_key,
                                api_secret,
                                position_side="SHORT" if cfg["hedge_mode"] else None,
                                client_order_id=self._generate_client_order_id("re-enter"),
                            )
                        except Exception as e2:
                            self.append_log(
                                f"❌ [DEMO 리버스 진입 실패] {symbol} SHORT {q2}: {e2}\n"
                            )
                            return

                        filled2 = float(resp2.get("avgPrice") or resp2.get("price", last_price))
                        state["position_side"] = new_signal
                        state["qty"] = q2
                        state["entry_price"] = filled2
                        state["estimated_pnl"] = 0.0
                        self.append_log(
                            f"[DEMO 리버스 진입] {symbol} SHORT price≈{filled2:.2f}, qty={q2}\n"
                        )

        else:  # SHORT 포지션
            tp_price = ep * (1 - tp_pct)
            sl_price = ep * (1 + sl_pct)
            need_close = False
            reason = None

            if last_price <= tp_price:
                need_close = True
                reason = "TP"
            elif last_price >= sl_price:
                need_close = True
                reason = "SL"
            elif signal == "LONG":
                need_close = True
                reason = "Reverse"

            if need_close:
                side = "BUY"  # SHORT 포지션 청산은 BUY
                try:
                    resp = demo_place_market_order(
                        symbol,
                        side,
                        qty,
                        api_key,
                        api_secret,
                        reduce_only=True,
                        position_side="SHORT" if cfg["hedge_mode"] else None,
                        client_order_id=self._generate_client_order_id("close"),
                    )
                except Exception as e:
                    self.append_log(
                        f"❌ [DEMO 청산 실패] {symbol} {side} {qty}: {e}\n"
                    )
                    return

                exit_price = float(resp.get("avgPrice") or resp.get("price", last_price))
                pnl = (ep - exit_price) * qty
                state["estimated_pnl"] += pnl
                self._update_loss_and_check_limit(pnl)

                self.append_log(
                    f"[DEMO 청산] {symbol} SHORT ep={ep:.2f} -> fill≈{exit_price:.2f}, "
                    f"reason={reason}, qty={qty}, PnL≈{pnl:.4f}\n"
                )
                state["position_side"] = None
                state["qty"] = 0.0
                state["entry_price"] = 0.0

                if reason == "Reverse" and not time_filter_block_entry and not self.demo_paused_by_loss:
                    new_signal = "LONG"
                    q2 = calc_qty(symbol, last_price, base_notional)
                    if q2 > 0:
                        try:
                            resp2 = demo_place_market_order(
                                symbol,
                                "BUY",
                                q2,
                                api_key,
                                api_secret,
                                position_side="LONG" if cfg["hedge_mode"] else None,
                                client_order_id=self._generate_client_order_id("re-enter"),
                            )
                        except Exception as e2:
                            self.append_log(
                                f"❌ [DEMO 리버스 진입 실패] {symbol} LONG {q2}: {e2}\n"
                            )
                            return
                        filled2 = float(resp2.get("avgPrice") or resp2.get("price", last_price))
                        state["position_side"] = new_signal
                        state["qty"] = q2
                        state["entry_price"] = filled2
                        state["estimated_pnl"] = 0.0
                        self.append_log(
                            f"[DEMO 리버스 진입] {symbol} LONG price≈{filled2:.2f}, qty={q2}\n"
                        )


# ================================
# 4) 실행
# ================================
if __name__ == "__main__":
    root = tk.Tk()
    app = BacktestGUI(root)
    root.mainloop()
