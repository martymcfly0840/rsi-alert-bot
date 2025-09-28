
# -*- coding: utf-8 -*-
"""
Discord RSI<50 riasztások – S&P500 + Russell2000 (Twelve Data, 1p) és Crypto (CoinGecko, 5p)
Futtatás: GitHub Actions (cron) vagy lokális Python
Szerző: Kiss, Bálint részére

Környezeti változók (GitHub Secrets):
  - TWELVEDATA_API_KEY
  - COINGECKO_DEMO_API_KEY (opcionális, Demo)
  - COINGECKO_PRO_API_KEY  (opcionális, Pro)
  - DISCORD_WEBHOOK_URL

Fájlok (repo gyökerében):
  - sp500.csv           # egysoros ticker lista (fejléc nélkül)
  - russell2000.csv     # egysoros ticker lista (fejléc nélkül)
  - alert_cache.json    # futások között jelzés-cooldown állapot (Actions workflow visszamenti)

Megjegyzések:
  - CoinGecko Demo: root URL api.coingecko.com, header: x-cg-demo-api-key; ~30 req/perc, 10k/hó
  - CoinGecko Pro:  root URL pro-api.coingecko.com, header: x-cg-pro-api-key
  - Twelve Data:     batch /time_series támogatott, 1p OHLC "date=today" paraméterrel
"""

import os
import time
import json
import math
import logging
import datetime as dt
from typing import List, Dict, Optional

import pandas as pd
import requests

# -----------------------------
# Beállítások
# -----------------------------
CONFIG = {
    "stocks": {
        "interval": "1min",
        "rsi_period": 14,
        "outputsize": 160,                 # ~2.5 óra perces adat elég az RSI-hez
        "batch_size": 25,                  # Twelve Data batch méret
        "sleep_between_batches_sec": 2.0,  # batch-ek közt pici szünet
        "files": ["sp500.csv", "russell2000.csv"],
        "max_symbols": None               # None = teljes lista
    },
    "crypto": {
        "vs_currency": "usd",
        "market_pages": 4,                 # 4*250 = 1000 coin univerzum
        "per_page": 250,
        "min_24h_volume": 500_000,         # USD
        "top_n_for_rsi": 200,              # RSI számítás jelöltek száma (rate-limit barát)
        "rsi_period": 14,
        "sleep_between_market_pages_sec": 1.0,
        "sleep_between_chart_calls_sec": 2.2  # ~27 calls/perc
    },
    "alerts": {
        "discord_webhook": os.getenv("DISCORD_WEBHOOK_URL", "").strip(),
        "cooldown_minutes": 45,            # ugyanarra a tickerre ennyi ideig ne jelezzen
        "embed_color_stock": 0x2e7d32,
        "embed_color_crypto": 0x1565c0
    },
    "logging": {"level": "INFO"}
}

logging.basicConfig(level=getattr(logging, CONFIG["logging"]["level"]),
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rsi_alerts")

# -----------------------------
# Segédfüggvények
# -----------------------------

def load_universe_from_files(files: List[str], max_symbols: Optional[int] = None) -> List[str]:
    tickers = []
    for f in files:
        if os.path.exists(f):
            try:
                s = pd.read_csv(f, header=None).iloc[:, 0].astype(str).str.strip().tolist()
                tickers.extend([t for t in s if t and t.upper() != "SYMBOL"])
            except Exception as e:
                logger.warning(f"Hiba a fájl olvasásakor: {f}: {e}")
        else:
            logger.warning(f"Hiányzik a fájl: {f} (kihagyva)")
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    tickers = sorted(list(dict.fromkeys(tickers)))
    if max_symbols:
        tickers = tickers[:max_symbols]
    return tickers


def chunked(lst: List[str], size: int) -> List[List[str]]:
    return [lst[i:i+size] for i in range(0, len(lst), size)]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, 1e-12)
    return 100 - (100 / (1 + rs))


# Discord
CACHE_FILE = "alert_cache.json"

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


def send_discord_alert(symbol: str, rsi_value: float, price: Optional[float], kind: str, timeframe: str):
    url = CONFIG["alerts"]["discord_webhook"]
    if not url:
        logger.error("Nincs beállítva DISCORD_WEBHOOK_URL")
        return

    color = CONFIG["alerts"]["embed_color_stock"] if kind == "stock" else CONFIG["alerts"]["embed_color_crypto"]
    title = f"{symbol} – RSI {timeframe} < 50"
    price_str = f"{price:,.4f}" if (price is not None) else "n.a."
    now_utc = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    payload = {
        "content": None,
        "embeds": [
            {
                "title": title,
                "description": f"**RSI**: `{rsi_value:.2f}`\n**Ár**: `{price_str}`\n**Időkeret**: `{timeframe}`\n**Forrás**: {'Twelve Data' if kind=='stock' else 'CoinGecko'}\n`{now_utc}`",
                "color": color,
            }
        ],
        "username": "RSI Alert Bot",
        "attachments": []
    }

    try:
        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code in (200, 201, 202, 204):
            logger.info(f"Discord riasztás elküldve: {symbol} (RSI={rsi_value:.2f})")
        elif resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "1"))
            logger.warning(f"Discord 429 – várakozás {retry_after:.1f}s, újrapróbálás...")
            time.sleep(retry_after + 0.5)
            requests.post(url, json=payload, timeout=20)
        else:
            logger.error(f"Discord hiba {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.exception(f"Discord hívás hiba: {e}")


# -----------------------------
# Twelve Data – részvények
# -----------------------------
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
TD_BASE = "https://api.twelvedata.com"

def td_time_series_batch(symbols: List[str], interval: str, outputsize: int, date_today: bool = True) -> Dict[str, dict]:
    params = {
        "symbol": ",".join(symbols),
        "interval": interval,
        "outputsize": outputsize,
        "order": "asc",
        "apikey": TWELVEDATA_API_KEY,
    }
    if date_today:
        params["date"] = "today"
    try:
        r = requests.get(f"{TD_BASE}/time_series", params=params, timeout=30)
        return r.json()
    except Exception as e:
        logger.exception(f"Twelve Data hiba: {e}")
        return {}


def scan_stocks_rsi_lt50():
    if not TWELVEDATA_API_KEY:
        logger.error("Hiányzik a TWELVEDATA_API_KEY")
        return 0

    uni = load_universe_from_files(CONFIG["stocks"]["files"], CONFIG["stocks"]["max_symbols"])
    if not uni:
        logger.warning("Tickerlista üres. sp500.csv / russell2000.csv?")
        return 0

    cache = load_cache()
    cooldown = CONFIG["alerts"]["cooldown_minutes"]
    chunks = chunked(uni, CONFIG["stocks"]["batch_size"])
    logger.info(f"Részvény szkennelés: {len(uni)} ticker, {len(chunks)} batch")

    alerts = 0
    for i, batch in enumerate(chunks, start=1):
        data = td_time_series_batch(batch, CONFIG["stocks"]["interval"], CONFIG["stocks"]["outputsize"], True)
        for sym in batch:
            sub = data.get(sym)
            if not sub or "values" not in sub:
                continue
            try:
                df = pd.DataFrame(sub["values"]).rename(columns=str.lower)
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.sort_values("datetime").reset_index(drop=True)
                if len(df) < CONFIG["stocks"]["rsi_period"] + 5:
                    continue
                rsi = compute_rsi(df["close"], CONFIG["stocks"]["rsi_period"])
                last_rsi = float(rsi.iloc[-1])
                last_close = float(df["close"].iloc[-1])
                if last_rsi < 50.0:
                    key = f"stock::{sym}"
                    if can_alert(cache, key, cooldown):
                        send_discord_alert(sym, last_rsi, last_close, kind="stock", timeframe="1m")
                        mark_alert(cache, key)
                        alerts += 1
            except Exception as e:
                logger.debug(f"{sym} feldolgozás hiba: {e}")
        save_cache(cache)
        if i < len(chunks):
            time.sleep(CONFIG["stocks"]["sleep_between_batches_sec"])
    return alerts


# -----------------------------
# CoinGecko – kripto
# -----------------------------
CG_DEMO_KEY = os.getenv("COINGECKO_DEMO_API_KEY", "").strip()
CG_PRO_KEY  = os.getenv("COINGECKO_PRO_API_KEY", "").strip()


def get_cg_base() -> str:
    # Pro kulccsal a PRO hostot használjuk
    return "https://pro-api.coingecko.com/api/v3" if CG_PRO_KEY else "https://api.coingecko.com/api/v3"


def cg_headers() -> Dict[str, str]:
    if CG_PRO_KEY:
        return {"x-cg-pro-api-key": CG_PRO_KEY}
    elif CG_DEMO_KEY:
        return {"x-cg-demo-api-key": CG_DEMO_KEY}
    return {}


def cg_coins_markets_pages(vs_currency: str, per_page: int, pages: int, min_vol: float) -> List[dict]:
    base = get_cg_base()
    headers = cg_headers()
    out: List[dict] = []
    for p in range(1, pages + 1):
        params = {
            "vs_currency": vs_currency,
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
                out.extend([row for row in data if float(row.get("total_volume", 0) or 0) >= min_vol])
        except Exception as e:
            logger.debug(f"CoinGecko markets hiba p={p}: {e}")
        time.sleep(CONFIG["crypto"]["sleep_between_market_pages_sec"])
    return out


def cg_market_chart_prices(coin_id: str, vs_currency: str) -> Optional[pd.Series]:
    base = get_cg_base()
    headers = cg_headers()
    params = {"vs_currency": vs_currency, "days": 1}
    try:
        r = requests.get(f"{base}/coins/{coin_id}/market_chart", params=params, headers=headers, timeout=30)
        if r.status_code == 429:
            time.sleep(2.5)
            r = requests.get(f"{base}/coins/{coin_id}/market_chart", params=params, headers=headers, timeout=30)
        data = r.json()
        prices = data.get("prices")
        if not prices:
            return None
        df = pd.DataFrame(prices, columns=["ts", "price"])  # 5p granularitás (Demo/Pro auto)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.sort_values("ts").reset_index(drop=True)
        return df["price"]
    except Exception:
        return None


def scan_crypto_rsi_lt50():
    cache = load_cache()
    cooldown = CONFIG["alerts"]["cooldown_minutes"]

    uni_rows = cg_coins_markets_pages(
        vs_currency=CONFIG["crypto"]["vs_currency"],
        per_page=CONFIG["crypto"]["per_page"],
        pages=CONFIG["crypto"]["market_pages"],
        min_vol=CONFIG["crypto"]["min_24h_volume"],
    )
    if not uni_rows:
        logger.warning("CoinGecko nem adott vissza adatot az univerzumra.")
        return 0

    # válogatás market cap szerint, majd limit
    uni_rows = sorted(uni_rows, key=lambda r: r.get("market_cap", 0) or 0, reverse=True)
    uni_rows = uni_rows[:CONFIG["crypto"]["top_n_for_rsi"]]

    alerts = 0
    for row in uni_rows:
        cid = row.get("id")
        sym = (row.get("symbol") or "").upper()
        cur_price = float(row.get("current_price", 0) or 0)
        series = cg_market_chart_prices(cid, CONFIG["crypto"]["vs_currency"]) if cid else None
        if series is None or len(series) < CONFIG["crypto"]["rsi_period"] + 5:
            time.sleep(CONFIG["crypto"]["sleep_between_chart_calls_sec"])
            continue
        rsi = compute_rsi(series, CONFIG["crypto"]["rsi_period"])
        last_rsi = float(rsi.iloc[-1])
        if last_rsi < 50.0:
            key = f"crypto::{sym}"
            if can_alert(cache, key, cooldown):
                send_discord_alert(sym, last_rsi, cur_price, kind="crypto", timeframe="5m")
                mark_alert(cache, key)
                alerts += 1
        save_cache(cache)
        time.sleep(CONFIG["crypto"]["sleep_between_chart_calls_sec"])
    return alerts


# -----------------------------
# Main
# -----------------------------

def main():
    start = dt.datetime.now()
    logger.info("=== RSI<50 alert futtatás indul ===")

    total_alerts = 0
    try:
        a = scan_stocks_rsi_lt50()
        total_alerts += int(a or 0)
    except Exception as e:
        logger.exception(f"Részvény-szkennelés hiba: {e}")

    try:
        b = scan_crypto_rsi_lt50()
        total_alerts += int(b or 0)
    except Exception as e:
        logger.exception(f"Kripto-szkennelés hiba: {e}")

    dur = dt.datetime.now() - start
    logger.info(f"=== Kész. Riasztások száma: {total_alerts}, futási idő: {dur} ===")


if __name__ == "__main__":
    main()
