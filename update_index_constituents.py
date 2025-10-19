# -*- coding: utf-8 -*-
"""
update_index_constituents.py  (v2)
- S&P 500: Wikipedia -> fallback Slickcharts
- Russell 2000: FTSE Russell (LSEG) -> fallback Slickcharts -> fallback TradingView
- Részletes LOG + Artifacts támogatás (CSV-k generálása akkor is, ha az egyik forrás bukik)
- Soha nem dob exit kódot, ha legalább az egyik index sikerül (S&P vagy R2000).
"""

import io
import os
import sys
import json
import time
import logging
from typing import List, Optional

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("update_constituents")

HDRS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) M365-Copilot/1.0 (+https://github.com/)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
TIMEOUT = 45

def _clean_tickers(sym_series: pd.Series) -> List[str]:
    """
    Alap normalizálás, duplikátumszűrés, nagybetűsítés. Nem konvertáljuk a '.'-ot '-'-ra (pl. BRK.B marad).
    """
    syms = (
        sym_series.astype(str)
        .str.strip()
        .replace({"": pd.NA})
        .dropna()
        .tolist()
    )
    seen = set()
    out = []
    for s in syms:
        u = s.upper()
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def _write_csv(path: str, symbols: List[str]) -> None:
    pd.DataFrame(symbols).to_csv(path, header=False, index=False)
    log.info(f"Írva: {path} ({len(symbols)} sor)")

# ------------------- S&P 500 -------------------

def fetch_sp500_from_wikipedia() -> Optional[List[str]]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        r = requests.get(url, headers=HDRS, timeout=TIMEOUT)
        r.raise_for_status()
        tables = pd.read_html(r.text)  # lxml / html5lib
        for tbl in tables:
            cols = [str(c).lower() for c in tbl.columns]
            if "symbol" in cols:
                syms = _clean_tickers(tbl[tbl.columns[cols.index("symbol")]])
                if len(syms) > 400:
                    log.info(f"SP500 Wikipedia OK: {len(syms)} ticker")
                    return syms
        log.warning("Wikipedia SP500: nem találtam 'Symbol' oszlopot.")
        return None
    except Exception as e:
        log.warning(f"Wikipedia SP500 hiba: {e}")
        return None

def fetch_sp500_from_slickcharts() -> Optional[List[str]]:
    url = "https://www.slickcharts.com/sp500"
    try:
        r = requests.get(url, headers=HDRS, timeout=TIMEOUT)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        for tbl in tables:
            if "Symbol" in tbl.columns:
                syms = _clean_tickers(tbl["Symbol"])
                if len(syms) > 400:
                    log.info(f"SP500 Slickcharts OK: {len(syms)} ticker")
                    return syms
        log.warning("Slickcharts SP500: nem találtam 'Symbol' oszlopot.")
        return None
    except Exception as e:
        log.warning(f"Slickcharts SP500 hiba: {e}")
        return None

def get_sp500() -> List[str]:
    syms = fetch_sp500_from_wikipedia()
    if not syms:
        log.info("SP500 fallback: Slickcharts")
        syms = fetch_sp500_from_slickcharts() or []
    return syms

# ------------------- Russell 2000 -------------------

def fetch_r2000_from_ftse() -> Optional[List[str]]:
    """
    FTSE Russell (LSEG): Constituents & Weights
    """
    url = "https://research.ftserussell.com/Analytics/factsheets/Home/DownloadConstituentsWeights/?indexdetails=US2000"
    try:
        r = requests.get(url, headers=HDRS, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.content

        # Próbáljuk CSV-ként
        def try_read_csv():
            try:
                return pd.read_csv(io.BytesIO(data))
            except Exception:
                return None

        # Próbáljuk Excelként
        def try_read_xlsx():
            try:
                return pd.read_excel(io.BytesIO(data))
            except Exception:
                return None

        df = try_read_csv()
        if df is None or df.empty:
            df = try_read_xlsx()

        if df is None or df.empty:
            log.warning("FTSE R2000: üres/betöltési hiba.")
            return None

        cols_l = [str(c).lower() for c in df.columns]
        for cand in ("ticker", "symbol", "constituent ticker", "bbg ticker", "ric"):
            if cand in cols_l:
                series = df[df.columns[cols_l.index(cand)]]
                syms = _clean_tickers(series)
                if len(syms) > 1000:
                    log.info(f"R2000 FTSE OK: {len(syms)} ticker")
                    return syms
        log.warning(f"FTSE R2000: nem találtam ticker oszlopot. Oszlopok: {df.columns.tolist()[:6]}...")
        return None
    except Exception as e:
        log.warning(f"FTSE R2000 hiba: {e}")
        return None

def fetch_r2000_from_slickcharts() -> Optional[List[str]]:
    url = "https://www.slickcharts.com/russell2000"
    try:
        r = requests.get(url, headers=HDRS, timeout=TIMEOUT)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        for tbl in tables:
            if "Symbol" in tbl.columns:
                syms = _clean_tickers(tbl["Symbol"])
                if len(syms) > 1000:
                    log.info(f"R2000 Slickcharts OK: {len(syms)} ticker")
                    return syms
        log.warning("Slickcharts R2000: nem találtam 'Symbol' oszlopot.")
        return None
    except Exception as e:
        log.warning(f"Slickcharts R2000 hiba: {e}")
        return None

def fetch_r2000_from_tradingview() -> Optional[List[str]]:
    """
    TradingView komponens oldal (szerveroldali render), jellemzően tartalmaz Symbol oszlopot.
    """
    url = "https://www.tradingview.com/symbols/CBOEFTSE-RUT/components/"
    try:
        r = requests.get(url, headers=HDRS, timeout=TIMEOUT)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        for tbl in tables:
            for col in tbl.columns:
                if str(col).strip().lower() in ("symbol", "ticker"):
                    syms = _clean_tickers(tbl[col])
                    if len(syms) > 500:  # lehet, hogy nem teljes, de részhalmaznak jó
                        log.info(f"R2000 TradingView részlista OK: {len(syms)} ticker")
                        return syms
        log.warning("TradingView R2000: nem találtam Symbol/Ticker oszlopot.")
        return None
    except Exception as e:
        log.warning(f"TradingView R2000 hiba: {e}")
        return None

def get_russell2000() -> List[str]:
    for fn in (fetch_r2000_from_ftse, fetch_r2000_from_slickcharts, fetch_r2000_from_tradingview):
        syms = fn()
        if syms:
            return syms
        time.sleep(1.0)  # kis szünet a források között
    return []

# ------------------- MAIN -------------------

def main():
    ok_any = False

    sp = get_sp500()
    if sp:
        _write_csv("sp500.csv", sp)
        ok_any = True
    else:
        log.error("SP500 lista üres – nem írok fájlt.")

    r2k = get_russell2000()
    if r2k:
        _write_csv("russell2000.csv", r2k)
        ok_any = True
    else:
        log.error("Russell2000 lista üres – nem írok fájlt.")

    # Artifacts mappába is mentsünk (ha a workflow feltölti)
    try:
        os.makedirs("artifacts", exist_ok=True)
        if sp:
            _write_csv(os.path.join("artifacts", "sp500.csv"), sp)
        if r2k:
            _write_csv(os.path.join("artifacts", "russell2000.csv"), r2k)
    except Exception as e:
        log.warning(f"Artifacts mentés hiba: {e}")

    if not ok_any:
        log.error("Sem SP500, sem R2000 nem sikerült – exit 1")
        sys.exit(1)

if __name__ == "__main__":
    main()
