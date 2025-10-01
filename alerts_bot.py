# -*- coding: utf-8 -*-
"""
alerts_bot.py – Moduláris szabálymotor (RSI/BB) Twelve Data (részvény) + CoinGecko (kripto) + Discord
- Idősíkok: 1h / 4h / 1d
- Könnyen bővíthető RULES lista a fájl tetején (enabled flaggel kapcsolható szabályok)

Secrets (GitHub → Settings → Secrets → Actions):
  TWELVEDATA_API_KEY, COINGECKO_DEMO_API_KEY (vagy COINGECKO_PRO_API_KEY), DISCORD_WEBHOOK_URL
"""

import os, time, json, logging, datetime as dt
from typing import List, Dict, Optional, Any
import pandas as pd
import requests

# -----------------------------
# KONFIG + SZABÁLYOK (ITT SZERKESZTESZ A JÖVŐBEN)
# -----------------------------
CONFIG = {
    "stocks": {
        "timeframes": ["1h", "4h", "1d"],    # egyszerre több idősíkon is mehet a szkennelés
        "rsi_period": 14,
        "bb_period": 20,
        "bb_stddev": 2.0,
        "outputsize": 500,
        "batch_size": 25,
        "sleep_between_batches_sec": 2.0,
        "files": ["sp500.csv", "russell2000.csv"],
        "max_symbols": None,
    },
    "crypto": {
        "timeframes": ["1h", "4h", "1d"],
        "vs_currency": "usd",
        "market_pages": 4,                   # 4*250=1000 coin
        "per_page": 250,
        "min_24h_volume": 500_000,
        "top_n_for_rsi": 200,
        "rsi_period": 14,
        "bb_period": 20,
        "bb_stddev": 2.0,
        "sleep_between_market_pages_sec": 1.0,
        "sleep_between_chart_calls_sec": 2.2
    },
    "alerts": {
        "discord_webhook": os.getenv("DISCORD_WEBHOOK_URL", "").strip(),
        "cooldown_minutes": 45,
        "embed_color_stock": 0x2e7d32,
        "embed_color_crypto": 0x1565c0
    },
    "logging": {"level": "INFO"}
}

# ---- SZABÁLYOK: itt adj hozzá új kéréseket (RSI, BB, kombinációk) ----
# type: "rsi_below" | "bb_touch" | "composite"
# scope: "stocks" | "crypto" | "both"
# enabled: True -> fut | False -> NEM fut
RULES: List[Dict[str, Any]] = [
    {
        "name": "RSI<50 (alap) [1h]",
        "type": "rsi_below",
        "threshold": 50,
        "timeframe": "1h",
        "scope": "both",
        "enabled": True
    },
    {
        "name": "RSI<40 & BB-Lower (4h)",
        "type": "composite",
        "all": [
            {"type": "rsi_below", "threshold": 40, "timeframe": "4h"},
            {"type": "bb_touch", "band": "lower", "timeframe": "4h", "period": 20, "stddev": 2.0}
        ],
        "scope": "crypto",
        "enabled": False
    },
    {
        "name": "RSI<45 (1d) – csak részvények",
        "type": "rsi_below",
        "threshold": 45,
        "timeframe": "1d",
        "scope": "stocks",
        "enabled": False
    },
    {
        "name": "BB lower érintés (4h) – mindkét piac",
        "type": "bb_touch",
        "band": "lower",
        "timeframe": "4h",
        "period": 20,
        "stddev": 2.0,
        "scope": "both",
        "enabled": False
    },
]

# -----------------------------
# LOGOLÁS
# -----------------------------
logging.basicConfig(level=getattr(logging, CONFIG["logging"]["level"]),
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rule_engine_bot")

# -----------------------------
# KÖZÖS SEGÉDEK (RSI/BB/Discord/Cache)
# -----------------------------
CACHE_FILE = "alert_cache.json"

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    d = s.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, 1e-12)
    return 100 - (100 / (1 + rs))

def compute_bbands(series: pd.Series, period=20, stddev=2.0):
    s = pd.to_numeric(series, errors="coerce").dropna()
    ma = s.rolling(window=period, min_periods=period).mean()
    sd = s.rolling(window=period, min_periods=period).std(ddof=0)
    upper = ma + stddev * sd
    lower = ma - stddev * sd
    return lower, ma, upper

def load_cache() -> Dict[str, float]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache: Dict[str, float]):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass

def can_alert(cache: Dict[str, float], key: str, cooldown_minutes: int) -> bool:
    return (time.time() - cache.get(key, 0)) > cooldown_minutes * 60

def mark_alert(cache: Dict[str, float], key: str):
    cache[key] = time.time()

def send_discord_alert(symbol: str, rule_name: str, rsi_value: Optional[float], price: Optional[float],
                       kind: str, timeframe: str):
    url = CONFIG["alerts"]["discord_webhook"]
    if not url:
        logger.error("Nincs beállítva DISCORD_WEBHOOK_URL")
        return
    color = CONFIG["alerts"]["embed_color_stock"] if kind == "stocks" else CONFIG["alerts"]["embed_color_crypto"]
    title = f"{symbol} – {rule_name} [{timeframe}]"
    price_str = f"{price:,.4f}" if (price is not None) else "n.a."
    rsi_str = f"{rsi_value:.2f}" if (rsi_value is not None) else "n.a."
    now_utc = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    payload = {"embeds": [{"title": title,
                            "description": f"**RSI**: `{rsi_str}`\n**Ár**: `{price_str}`\n**Időkeret**: `{timeframe}`\n`{now_utc}`",
                            "color": color}]}
    try:
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code in (200, 201, 202, 204):
            logger.info(f"Discord jelzés: {symbol} – {rule_name} [{timeframe}]")
        elif r.status_code == 429:
            retry = float(r.headers.get("Retry-After", "1"))
            time.sleep(retry + 0.5)
            requests.post(url, json=payload, timeout=20)
        else:
            logger.error(f"Discord hiba {r.status_code}: {r.text}")
    except Exception as e:
        logger.exception(f"Discord hívás hiba: {e}")

# -----------------------------
# UNIVERZUM (részvény/kripto)
# -----------------------------
def load_universe(files: List[str], max_symbols: Optional[int]) -> List[str]:
    out = []
    for f in files:
        if os.path.exists(f):
            s = pd.read_csv(f, header=None).iloc[:, 0].astype(str).str.strip().tolist()
            out += [t for t in s if t and t.upper() != "SYMBOL"]
        else:
            logger.warning(f"Hiányzik: {f}")
    out = sorted(list(dict.fromkeys([t.strip().upper() for t in out if t.strip()])))
    return out[:max_symbols] if max_symbols else out

# -----------------------------
# RÉSZVÉNY – Twelve Data
# -----------------------------
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
TD_BASE = "https://api.twelvedata.com"

def td_series_batch(symbols: List[str], interval: str, outputsize: int) -> Dict[str, pd.Series]:
    """ Több szimbólumra OHLC time_series, interval: '1h' | '4h' | '1day' -> close sorozat """
    params = {"symbol": ",".join(symbols), "interval": interval, "outputsize": outputsize,
              "order": "asc", "apikey": TWELVEDATA_API_KEY}
    out: Dict[str, pd.Series] = {}
    try:
        r = requests.get(f"{TD_BASE}/time_series", params=params, timeout=30)
        data = r.json()
        for sym in symbols:
            sub = data.get(sym)
            if not sub or "values" not in sub:
                continue
            df = pd.DataFrame(sub["values"]).rename(columns=str.lower)
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime").set_index("datetime")
            out[sym] = df["close"].dropna()
    except Exception as e:
        logger.exception(f"Twelve Data hiba: {e}")
    return out

# -----------------------------
# KRIPTO – CoinGecko
# -----------------------------
CG_DEMO_KEY = os.getenv("COINGECKO_DEMO_API_KEY", "").strip()
CG_PRO_KEY  = os.getenv("COINGECKO_PRO_API_KEY", "").strip()

def cg_base() -> str:
    return "https://pro-api.coingecko.com/api/v3" if CG_PRO_KEY else "https://api.coingecko.com/api/v3"
def cg_headers() -> Dict[str, str]:
    return {"x-cg-pro-api-key": CG_PRO_KEY} if CG_PRO_KEY else ({"x-cg-demo-api-key": CG_DEMO_KEY} if CG_DEMO_KEY else {})

def cg_markets(vs: str, per_page: int, pages: int, min_vol: float) -> List[dict]:
    base, headers = cg_base(), cg_headers()
    out: List[dict] = []
    for p in range(1, pages + 1):
        params = {"vs_currency": vs, "order": "market_cap_desc", "per_page": per_page, "page": p,
                  "sparkline": "false", "price_change_percentage": "24h"}
        try:
            r = requests.get(f"{base}/coins/markets", params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                time.sleep(2.5)
                r = requests.get(f"{base}/coins/markets", params=params, headers=headers, timeout=30)
            data = r.json()
            if isinstance(data, list):
                out += [row for row in data if float(row.get("total_volume", 0) or 0) >= min_vol]
        except Exception as e:
            logger.debug(f"CoinGecko markets hiba p={p}: {e}")
        time.sleep(CONFIG["crypto"]["sleep_between_market_pages_sec"])
    return out

def cg_series(coin_id: str, vs: str, timeframe: str) -> Optional[pd.Series]:
    """ 1h: hourly(7d); 4h: hourly(14d)→resample 4H; 1d: daily(365d) – CG auto granularitás """
    base, headers = cg_base(), cg_headers()
    days = 7 if timeframe=="1h" else (14 if timeframe=="4h" else 365)
    try:
        r = requests.get(f"{base}/coins/{coin_id}/market_chart",
                         params={"vs_currency": vs, "days": days},
                         headers=headers, timeout=30)
        if r.status_code == 429:
            time.sleep(2.5)
            r = requests.get(f"{base}/coins/{coin_id}/market_chart",
                             params={"vs_currency": vs, "days": days},
                             headers=headers, timeout=30)
        data = r.json(); prices = data.get("prices")
        if not prices: return None
        df = pd.DataFrame(prices, columns=["ts","price"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.sort_values("ts").set_index("ts")
        if timeframe=="1h":
            s = df["price"].resample("1H").last().dropna()
        elif timeframe=="4h":
            s = df["price"].resample("1H").last().dropna().resample("4H").last().dropna()
        else:
            s = df["price"].resample("1D").last().dropna()
        return s
    except Exception:
        return None

# -----------------------------
# SZABÁLYKIÉRTÉKELÉS
# -----------------------------
def eval_rsi_below(series: pd.Series, thr: float, period: int) -> Optional[float]:
    if series is None or len(series) < period + 5:
        return None
    rsi = compute_rsi(series, period)
    last = float(rsi.iloc[-1])
    return last if last < thr else None

def eval_bb_touch(series: pd.Series, period: int, stddev: float, band: str) -> bool:
    if series is None or len(series) < period + 5:
        return False
    lower, mid, upper = compute_bbands(series, period, stddev)
    close = series.iloc[-1]
    lb = None if pd.isna(lower.iloc[-1]) else float(lower.iloc[-1])
    ub = None if pd.isna(upper.iloc[-1]) else float(upper.iloc[-1])
    return (band == "lower" and lb is not None and close <= lb) or (band == "upper" and ub is not None and close >= ub)

def run_rules_for(kind: str, symbol: str, get_series_fn, get_price_fn) -> int:
    cache, cooldown = load_cache(), CONFIG["alerts"]["cooldown_minutes"]
    alerts = 0
    # mely idősíkok kellenek (CSAK az enabled szabályokhoz)?
    needed: set = set()
    for r in RULES:
        if r.get("enabled", True) is False:
            continue
        if r["scope"] not in ("both", kind):
            continue
        if r["type"] == "composite":
            needed |= {c["timeframe"] for c in r["all"]}
        else:
            needed.add(r["timeframe"])
    # előtöltjük a sorozatokat
    series_cache: Dict[str, pd.Series] = {tf: get_series_fn(tf) for tf in needed}
    cur_price = get_price_fn()
    # szabályok futtatása (CSAK az enabled szabályok)
    for r in RULES:
        if r.get("enabled", True) is False:
            continue
        if r["scope"] not in ("both", kind):
            continue
        fired, rep_rsi, tf = False, None, None
        if r["type"] == "composite":
            oks, rsis, tfs = [], [], []
            for c in r["all"]:
                s = series_cache.get(c["timeframe"])
                tf = c["timeframe"]; tfs.append(tf)
                if c["type"] == "rsi_below":
                    val = eval_rsi_below(s, float(c["threshold"]), CONFIG[kind]["rsi_period"])
                    oks.append(val is not None); rsis.append(val)
                elif c["type"] == "bb_touch":
                    ok = eval_bb_touch(s,
                                       int(c.get("period", CONFIG[kind]["bb_period"])),
                                       float(c.get("stddev", CONFIG[kind]["bb_stddev"])),
                                       c.get("band", "lower"))
                    oks.append(ok); rsis.append(None)
            fired = all(oks)
            tf = tfs[0] if tfs else None
            rep_rsi = next((x for x in rsis if x is not None), None)
        else:
            s = series_cache.get(r["timeframe"])
            tf = r["timeframe"]
            if r["type"] == "rsi_below":
                rep_rsi = eval_rsi_below(s, float(r["threshold"]), CONFIG[kind]["rsi_period"])
                fired = rep_rsi is not None
            elif r["type"] == "bb_touch":
                fired = eval_bb_touch(s,
                                      int(r.get("period", CONFIG[kind]["bb_period"])),
                                      float(r.get("stddev", CONFIG[kind]["bb_stddev"])),
                                      r.get("band", "lower"))
        if fired:
            key = f"{kind}::{symbol}::{r['name']}::{tf}"
            if can_alert(cache, key, cooldown):
                send_discord_alert(symbol, r["name"], rep_rsi, cur_price, kind, tf or "")
                mark_alert(cache, key); save_cache(cache); alerts += 1
    return alerts

# -----------------------------
# FUTTATÁS – RÉSZVÉNY & KRIPTO
# -----------------------------
def run_stocks() -> int:
    if not TWELVEDATA_API_KEY:
        logger.error("Hiányzik a TWELVEDATA_API_KEY")
        return 0
    syms = load_universe(CONFIG["stocks"]["files"], CONFIG["stocks"]["max_symbols"])
    if not syms:
        logger.warning("Részvény tickerlista üres.")
        return 0
    alerts = 0
    # az összes érintett idősík (CSAK enabled + stocks/both scope)
    tfs = set()
    for r in RULES:
        if r.get("enabled", True) is False:
            continue
        if r["scope"] in ("both", "stocks"):
            if r["type"] == "composite":
                tfs |= {c["timeframe"] for c in r["all"]}
            else:
                tfs.add(r["timeframe"])
    interval_map = {"1h": "1h", "4h": "4h", "1d": "1day"}
    for tf in sorted(tfs):
        td_interval = interval_map.get(tf, "1h")
        logger.info(f"Részvények ({tf}) szkennelése...")
        # batch
        for i in range(0, len(syms), CONFIG["stocks"]["batch_size"]):
            batch = syms[i:i + CONFIG["stocks"]["batch_size"]]
            series_map = td_series_batch(batch, td_interval, CONFIG["stocks"]["outputsize"])
            for sym in batch:
                s = series_map.get(sym)
                get_series = lambda tf_=tf, s_=s: s_
                get_price  = lambda: float(s.iloc[-1]) if s is not None and len(s) else None
                alerts += run_rules_for("stocks", sym, get_series, get_price)
            if i + CONFIG["stocks"]["batch_size"] < len(syms):
                time.sleep(CONFIG["stocks"]["sleep_between_batches_sec"])
    return alerts

def run_crypto() -> int:
    rows = cg_markets(CONFIG["crypto"]["vs_currency"], CONFIG["crypto"]["per_page"],
                      CONFIG["crypto"]["market_pages"], CONFIG["crypto"]["min_24h_volume"])
    if not rows:
        logger.warning("Kripto univerzum üres.")
        return 0
    rows = sorted(rows, key=lambda r: r.get("market_cap", 0) or 0, reverse=True)[:CONFIG["crypto"]["top_n_for_rsi"]]
    # idősíkok, amik a szabályokban érintik a kriptót (CSAK enabled + crypto/both)
    tfs = set()
    for r in RULES:
        if r.get("enabled", True) is False:
            continue
        if r["scope"] in ("both", "crypto"):
            if r["type"] == "composite":
                tfs |= {c["timeframe"] for c in r["all"]}
            else:
                tfs.add(r["timeframe"])
    alerts = 0
    for row in rows:
        cid = row.get("id")
        sym = (row.get("symbol") or "").upper()
        cur_price = float(row.get("current_price", 0) or 0)
        # előre lekérjük minden szükséges tf sorozatát
        series_cache: Dict[str, pd.Series] = {}
        for tf in sorted(tfs):
            series_cache[tf] = cg_series(cid, CONFIG["crypto"]["vs_currency"], tf) if cid else None
            time.sleep(CONFIG["crypto"]["sleep_between_chart_calls_sec"])
        get_series = lambda tf: series_cache.get(tf)
        get_price  = lambda: cur_price
        alerts += run_rules_for("crypto", sym, get_series, get_price)
    return alerts

# -----------------------------
# MAIN
# -----------------------------
def main():
    start = dt.datetime.now()
    logger.info("=== Rule Engine futás indul ===")
    total = 0
    try:
        total += run_stocks()
    except Exception as e:
        logger.exception(f"Részvény-hiba: {e}")
    try:
        total += run_crypto()
    except Exception as e:
        logger.exception(f"Kripto-hiba: {e}")
    dur = dt.datetime.now() - start
    logger.info(f"=== Kész. Jelzések: {total}, futási idő: {dur} ===")

if __name__ == "__main__":
    main()
