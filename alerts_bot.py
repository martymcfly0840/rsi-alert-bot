# -*- coding: utf-8 -*-
"""
alerts_bot.py – Modular rule engine (RSI/BB) for Stocks (Twelve Data) & Crypto (CoinGecko) + Discord alerts
- Timeframes: 1h / 4h / 1d
- RULES list at the top with `enabled` flags to toggle strategies on/off
- Option B1: frequent runs, but max 1 alert per bar (per-bar dedupe key)
- Filters: exclude stablecoins & NFT tokens from the crypto universe

GitHub Secrets required:
  TWELVEDATA_API_KEY
  COINGECKO_DEMO_API_KEY  (or)  COINGECKO_PRO_API_KEY
  DISCORD_WEBHOOK_URL
"""

import os
import time
import json
import logging
import datetime as dt
from typing import List, Dict, Optional, Any

import pandas as pd
import requests

# -----------------------------
# CONFIG + RULES (EDIT THESE)
# -----------------------------
CONFIG = {
    "stocks": {
        # We can scan multiple timeframes in one run if your rules use them
        "timeframes": ["1h", "4h", "1d"],
        "rsi_period": 14,
        "bb_period": 20,
        "bb_stddev": 2.0,
        "outputsize": 500,                 # bars per request (enough for RSI/BB & backfill)
        "batch_size": 25,
        "sleep_between_batches_sec": 2.0,
        "files": ["sp500.csv", "russell2000.csv"],
        "max_symbols": None,               # cap if needed
    },
    "crypto": {
        "timeframes": ["1h", "4h", "1d"],
        "vs_currency": "usd",
        "market_pages": 4,                 # 4 * 250 = 1000 top coins by mcap
        "per_page": 250,
        "min_24h_volume": 500_000,         # drop illiquid coins early
        "top_n_for_rsi": 200,              # compute indicators on top N after filters
        "rsi_period": 14,
        "bb_period": 20,
        "bb_stddev": 2.0,
        "sleep_between_market_pages_sec": 1.0,
        "sleep_between_chart_calls_sec": 2.2,

        # ---- Filters for the crypto universe ----
        "filters": {
            "exclude_stablecoins": True,
            "exclude_nft_tokens": True,

            # Symbols considered stable (lowercase). Extend as needed.
            "stable_symbol_list": [
                "usdt","usdc","tusd","dai","frax","usdd","gusd","lusd","susd",
                "usde","usdp","pax","fdusd","pyusd","alusd","cusd","usdy","usdx",
                "ousd","usd1","rlusd"
            ],

            # If id/name contains any of these substrings -> likely stable (lowercase)
            "stable_id_or_name_contains": ["usd", "stable", "pegged"],

            # (Meaningful if vs_currency="usd")
            "stable_price_band": [0.96, 1.04],        # ~±4% band
            "max_abs_24h_change_pct_for_stable": 1.0  # |24h%| <= 1% -> likely stable
        }
    },
    "alerts": {
        "discord_webhook": os.getenv("DISCORD_WEBHOOK_URL", "").strip(),
        "cooldown_minutes": 45,            # still applies, but per-bar dedupe will be stricter
        "embed_color_stock": 0x2e7d32,
        "embed_color_crypto": 0x1565c0
    },
    "logging": {"level": "INFO"}
}

# ---- RULES: add new conditions here (toggle with enabled=True/False) ----
# type: "rsi_below" | "bb_touch" | "composite"
# scope: "stocks" | "crypto" | "both"
# timeframe: "1h" | "4h" | "1d"
RULES: List[Dict[str, Any]] = [
    {
        "name": "RSI<50 (base) [1h]",
        "type": "rsi_below",
        "threshold": 50,
        "timeframe": "1h",
        "scope": "both",
        "enabled": True
    },
    {
        "name": "RSI<45 & BB-Lower (4h)",
        "type": "composite",
        "all": [
            {"type": "rsi_below", "threshold": 45, "timeframe": "4h"},
            {"type": "bb_touch", "band": "lower", "timeframe": "4h", "period": 20, "stddev": 2.0}
        ],
        "scope": "both",
        "enabled": True
    },
    # Examples (disabled by default)
    {
        "name": "RSI<45 (1d) – stocks only",
        "type": "rsi_below",
        "threshold": 45,
        "timeframe": "1d",
        "scope": "stocks",
        "enabled": False
    },
    {
        "name": "BB lower touch (4h) – both",
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
# LOGGING
# -----------------------------
logging.basicConfig(level=getattr(logging, CONFIG["logging"]["level"]),
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rule_engine_bot")

# -----------------------------
# COMMON HELPERS
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
        logger.error("DISCORD_WEBHOOK_URL is not set")
        return
    color = CONFIG["alerts"]["embed_color_stock"] if kind == "stocks" else CONFIG["alerts"]["embed_color_crypto"]
    title = f"{symbol} – {rule_name} [{timeframe}]"
    price_str = f"{price:,.4f}" if (price is not None) else "n.a."
    rsi_str = f"{rsi_value:.2f}" if (rsi_value is not None) else "n.a."
    now_utc = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    payload = {
        "embeds": [{
            "title": title,
            "description": f"**RSI**: `{rsi_str}`\n**Price**: `{price_str}`\n**TF**: `{timeframe}`\n`{now_utc}`",
            "color": color
        }]
    }
    try:
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code in (200, 201, 202, 204):
            logger.info(f"Discord alert sent: {symbol} – {rule_name} [{timeframe}]")
        elif r.status_code == 429:
            retry = float(r.headers.get("Retry-After", "1"))
            time.sleep(retry + 0.5)
            requests.post(url, json=payload, timeout=20)
        else:
            logger.error(f"Discord error {r.status_code}: {r.text}")
    except Exception as e:
        logger.exception(f"Discord call failed: {e}")

def bar_id_for_series(series: pd.Series, timeframe: str) -> Optional[str]:
    """Return a stable identifier for the last bar of a series, per timeframe (for per-bar dedupe)."""
    if series is None or series.empty:
        return None
    last_idx = series.index[-1]
    if timeframe == "1h":
        bid = last_idx.floor("1H")
    elif timeframe == "4h":
        bid = last_idx.floor("4H")
    else:
        bid = last_idx.floor("1D")
    return bid.isoformat()

# -----------------------------
# UNIVERSE LOADING
# -----------------------------
def load_universe(files: List[str], max_symbols: Optional[int]) -> List[str]:
    out = []
    for f in files:
        if os.path.exists(f):
            s = pd.read_csv(f, header=None).iloc[:, 0].astype(str).str.strip().tolist()
            out += [t for t in s if t and t.upper() != "SYMBOL"]
        else:
            logger.warning(f"Missing file: {f}")
    out = sorted(list(dict.fromkeys([t.strip().upper() for t in out if t.strip()])))
    return out[:max_symbols] if max_symbols else out

# -----------------------------
# STOCKS (Twelve Data)
# -----------------------------
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
TD_BASE = "https://api.twelvedata.com"

def td_series_batch(symbols: List[str], interval: str, outputsize: int) -> Dict[str, pd.Series]:
    """Fetch close series for multiple symbols; interval: '1h' | '4h' | '1day'."""
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
        logger.exception(f"Twelve Data error: {e}")
    return out

# -----------------------------
# CRYPTO (CoinGecko)
# -----------------------------
CG_DEMO_KEY = os.getenv("COINGECKO_DEMO_API_KEY", "").strip()
CG_PRO_KEY  = os.getenv("COINGECKO_PRO_API_KEY", "").strip()

def cg_base() -> str:
    return "https://pro-api.coingecko.com/api/v3" if CG_PRO_KEY else "https://api.coingecko.com/api/v3"

def cg_headers() -> Dict[str, str]:
    return {"x-cg-pro-api-key": CG_PRO_KEY} if CG_PRO_KEY else ({"x-cg-demo-api-key": CG_DEMO_KEY} if CG_DEMO_KEY else {})

# ---- NEW: Filters for crypto universe ----
def is_probable_stablecoin(row: dict, fcfg: dict) -> bool:
    sym = (row.get("symbol") or "").lower()
    cid = (row.get("id") or "").lower()
    name = (row.get("name") or "").lower()
    price = row.get("current_price")
    chg24 = row.get("price_change_percentage_24h")

    # 1) explicit symbols
    if sym in set(fcfg.get("stable_symbol_list", [])):
        return True

    # 2) id/name substrings
    needles = tuple(x.lower() for x in fcfg.get("stable_id_or_name_contains", []))
    if any(n in cid for n in needles) or any(n in name for n in needles):
        return True

    # 3) price band + low |24h%| if vs_currency is usd
    try:
        low, high = fcfg.get("stable_price_band", [0.98, 1.02])
        if price is not None and low <= float(price) <= high:
            limit = float(fcfg.get("max_abs_24h_change_pct_for_stable", 1.0))
            if chg24 is not None and abs(float(chg24)) <= limit:
                return True
    except Exception:
        pass
    return False

def is_probable_nft_token(row: dict) -> bool:
    cid = (row.get("id") or "").lower()
    name = (row.get("name") or "").lower()
    return ("nft" in cid) or ("nft" in name)

def cg_markets(vs: str, per_page: int, pages: int, min_vol: float) -> List[dict]:
    base, headers = cg_base(), cg_headers()
    out: List[dict] = []

    # Filter switches
    fcfg = CONFIG["crypto"].get("filters", {})
    flag_excl_stable = bool(fcfg.get("exclude_stablecoins", True))
    flag_excl_nft     = bool(fcfg.get("exclude_nft_tokens", True))

    dropped_stable = 0
    dropped_nft = 0
    kept = 0

    for p in range(1, pages + 1):
        params = {
            "vs_currency": vs,
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": p,
            "sparkline": "false",
            "price_change_percentage": "24h",
        }
        try:
            r = requests.get(f"{base}/coins/markets", params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                time.sleep(2.5)
                r = requests.get(f"{base}/coins/markets", params=params, headers=headers, timeout=30)
            data = r.json()
            if isinstance(data, list):
                for row in data:
                    # base filter: min volume
                    if float(row.get("total_volume", 0) or 0) < min_vol:
                        continue

                    # stablecoin filter
                    if flag_excl_stable and is_probable_stablecoin(row, fcfg):
                        dropped_stable += 1
                        continue

                    # NFT token filter
                    if flag_excl_nft and is_probable_nft_token(row):
                        dropped_nft += 1
                        continue

                    out.append(row)
                    kept += 1
        except Exception as e:
            logger.debug(f"CoinGecko markets error p={p}: {e}")

        time.sleep(CONFIG["crypto"]["sleep_between_market_pages_sec"])

    logger.info(f"CG universe filter → kept: {kept}, dropped stable: {dropped_stable}, dropped NFT: {dropped_nft}")
    return out

def cg_series(coin_id: str, vs: str, timeframe: str) -> Optional[pd.Series]:
    """ 1h: hourly(7d); 4h: hourly(14d)→resample 4H; 1d: daily(365d). Granularity is automatic by 'days'. """
    base, headers = cg_base(), cg_headers()
    days = 7 if timeframe == "1h" else (14 if timeframe == "4h" else 365)
    try:
        r = requests.get(f"{base}/coins/{coin_id}/market_chart",
                         params={"vs_currency": vs, "days": days},
                         headers=headers, timeout=30)
        if r.status_code == 429:
            time.sleep(2.5)
            r = requests.get(f"{base}/coins/{coin_id}/market_chart",
                             params={"vs_currency": vs, "days": days},
                             headers=headers, timeout=30)
        data = r.json()
        prices = data.get("prices")
        if not prices:
            return None
        df = pd.DataFrame(prices, columns=["ts", "price"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.sort_values("ts").set_index("ts")
        if timeframe == "1h":
            s = df["price"].resample("1H").last().dropna()
        elif timeframe == "4h":
            s = df["price"].resample("1H").last().dropna().resample("4H").last().dropna()
        else:  # 1d
            s = df["price"].resample("1D").last().dropna()
        return s
    except Exception:
        return None

# -----------------------------
# RULE EVALUATION
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

    # Which TFs do we need (only enabled rules + correct scope)?
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

    # Prefetch series for those TFs
    series_cache: Dict[str, pd.Series] = {tf: get_series_fn(tf) for tf in needed}
    cur_price = get_price_fn()

    # Evaluate rules (only enabled, correct scope)
    for r in RULES:
        if r.get("enabled", True) is False:
            continue
        if r["scope"] not in ("both", kind):
            continue

        fired, rep_rsi, tf = False, None, None

        if r["type"] == "composite":
            oks, rsis, tfs = [], [], []
            for c in r["all"]:
                tf = c["timeframe"]
                s = series_cache.get(tf)
                if c["type"] == "rsi_below":
                    val = eval_rsi_below(s, float(c["threshold"]), CONFIG[kind]["rsi_period"])
                    oks.append(val is not None); rsis.append(val)
                elif c["type"] == "bb_touch":
                    ok = eval_bb_touch(
                        s,
                        int(c.get("period", CONFIG[kind]["bb_period"])),
                        float(c.get("stddev", CONFIG[kind]["bb_stddev"])),
                        c.get("band", "lower")
                    )
                    oks.append(ok); rsis.append(None)
                tfs.append(tf)
            fired = all(oks)
            tf = tfs[0] if tfs else None
            rep_rsi = next((x for x in rsis if x is not None), None)

        else:
            tf = r["timeframe"]
            s = series_cache.get(tf)
            if r["type"] == "rsi_below":
                rep_rsi = eval_rsi_below(s, float(r["threshold"]), CONFIG[kind]["rsi_period"])
                fired = rep_rsi is not None
            elif r["type"] == "bb_touch":
                fired = eval_bb_touch(
                    s,
                    int(r.get("period", CONFIG[kind]["bb_period"])),
                    float(r.get("stddev", CONFIG[kind]["bb_stddev"])),
                    r.get("band", "lower")
                )

        if fired:
            # Option B1: per-bar dedupe key -> one alert per symbol/rule/TF per bar
            series_for_tf = series_cache.get(tf)
            bar_id = bar_id_for_series(series_for_tf, tf) or "na"
            key = f"{kind}::{symbol}::{r['name']}::{tf}::{bar_id}"

            if can_alert(cache, key, cooldown):
                send_discord_alert(symbol, r["name"], rep_rsi, cur_price, kind, tf or "")
                mark_alert(cache, key); save_cache(cache); alerts += 1

    return alerts

# -----------------------------
# RUNNERS (STOCKS & CRYPTO)
# -----------------------------
def run_stocks() -> int:
    if not TWELVEDATA_API_KEY:
        logger.error("Missing TWELVEDATA_API_KEY")
        return 0

    syms = load_universe(CONFIG["stocks"]["files"], CONFIG["stocks"]["max_symbols"])
    if not syms:
        logger.warning("Stocks universe is empty.")
        return 0

    alerts = 0

    # Which TFs are actually needed by enabled rules for stocks?
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
        logger.info(f"Stocks scanning ({tf}) ...")

        # Batches
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
        logger.warning("Crypto universe is empty after filters.")
        return 0

    rows = sorted(rows, key=lambda r: r.get("market_cap", 0) or 0, reverse=True)
    rows = rows[:CONFIG["crypto"]["top_n_for_rsi"]]
    logger.info(f"Crypto scanning with {len(rows)} coins (post-filter).")

    # Which TFs are actually needed by enabled rules for crypto?
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

        # Prefetch series for needed TFs (respect rate limit)
        series_cache: Dict[str, pd.Series] = {}
        for tf in sorted(tfs):
            series_cache[tf] = cg_series(cid, CONFIG["crypto"]["vs_currency"], tf) if cid else None
            time.sleep(CONFIG["crypto"]["sleep_between_chart_calls_sec"])

        def get_series_fn(tf: str) -> Optional[pd.Series]:
            return series_cache.get(tf)

        def get_price_fn() -> Optional[float]:
            return cur_price

        alerts += run_rules_for("crypto", sym, get_series_fn, get_price_fn)

    return alerts

# -----------------------------
# MAIN
# -----------------------------
def main():
    start = dt.datetime.now()
    logger.info("=== Rule Engine start ===")
    total = 0
    try:
        total += run_stocks()
    except Exception as e:
        logger.exception(f"Stocks runner error: {e}")
    try:
        total += run_crypto()
    except Exception as e:
        logger.exception(f"Crypto runner error: {e}")
    dur = dt.datetime.now() - start
    logger.info(f"=== Done. Alerts: {total}, duration: {dur} ===")

if __name__ == "__main__":
    main()
