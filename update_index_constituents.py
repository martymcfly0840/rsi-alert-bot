# -*- coding: utf-8 -*-
"""
update_index_constituents.py
Dinamikusan frissíti az sp500.csv és russell2000.csv fájlokat megbízható nyilvános forrásokból.

Források:
- S&P 500: Wikipedia elsődlegesen, fallback: Slickcharts
- Russell 2000: FTSE Russell (LSEG) Constituents & Weights elsődlegesen, fallback: Slickcharts

Megjegyzés:
- A CSV-k 1 oszloposak, fejléc nélkül, minden sor egy ticker.
- Külön figyelünk a speciális jelölésekre (pl. BRK.B); alapértelmezésben a forrás jelölését hagyjuk meg.
"""

import io
import sys
import time
import json
import random
import logging
from typing import List, Optional

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("update_constituents")

HDRS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) M365-Copilot/1.0 (+https://github.com/)",
}

# ---------- Segédfüggvények ----------

def _clean_tickers(sym_series: pd.Series) -> List[str]:
    """
    Normalizálás minimálisan: whitespace, üresek szűrése, duplikátumok elhagyása.
    Nem cserélünk '.' -> '-' karaktert, mert a Twelve Data US tickereknél a pont is megszokott (pl. BRK.B).
    Ha másik adatforráshoz szükséges, itt lehet finomítani.
    """
    syms = (
        sym_series.astype(str)
        .str.strip()
        .replace({"": pd.NA})
        .dropna()
        .tolist()
    )
    # Duplikátum elhagyás, eredeti sorrendben
    seen = set()
    out = []
    for s in syms:
        u = s.upper()
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def _write_csv(path: str, symbols: List[str]) -> None:
    df = pd.DataFrame(symbols)
    df.to_csv(path, header=False, index=False)
    log.info(f"Írva: {path} ({len(symbols)} sor)")

# ---------- S&P 500 letöltés ----------

def fetch_sp500_from_wikipedia() -> Optional[List[str]]:
    """
    Wikipedia: List of S&P 500 companies – a 'Symbol' oszlopot olvassuk.
    https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        resp = requests.get(url, headers=HDRS, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)  # html5lib/lxml szükséges
        # A fő "constituents" tábla általában az első vagy az első kettő között van.
        # Keresünk 'Symbol' nevű oszlopot.
        for tbl in tables:
            cols = [c.lower() for c in tbl.columns]
            if "symbol" in cols:
                return _clean_tickers(tbl[tbl.columns[cols.index("symbol")]])
        return None
    except Exception as e:
        log.warning(f"Wikipedia SP500 hiba: {e}")
        return None

def fetch_sp500_from_slickcharts() -> Optional[List[str]]:
    """
    Slickcharts S&P 500
    https://www.slickcharts.com/sp500
    """
    url = "https://www.slickcharts.com/sp500"
    try:
        resp = requests.get(url, headers=HDRS, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        # Az első táblában általában 'Symbol' kolumna van
        for tbl in tables:
            if "Symbol" in tbl.columns:
                return _clean_tickers(tbl["Symbol"])
        return None
    except Exception as e:
        log.warning(f"Slickcharts SP500 hiba: {e}")
        return None

def get_sp500() -> List[str]:
    syms = fetch_sp500_from_wikipedia()
    if not syms:
        log.info("Wikipedia sikertelen vagy nem talált Symbol oszlopot; fallback: Slickcharts S&P 500.")
        syms = fetch_sp500_from_slickcharts() or []
    return syms

# ---------- Russell 2000 letöltés ----------

def fetch_r2000_from_ftse() -> Optional[List[str]]:
    """
    FTSE Russell (LSEG) – Constituents & Weights (US2000)
    Publikus letöltő endpoint. Lehet, hogy XLSX/CSV jön vissza – próbáljuk olvasni dinamikusan.
    https://research.ftserussell.com/Analytics/factsheets/Home/DownloadConstituentsWeights/?indexdetails=US2000
    """
    url = "https://research.ftserussell.com/Analytics/factsheets/Home/DownloadConstituentsWeights/?indexdetails=US2000"
    try:
        resp = requests.get(url, headers=HDRS, timeout=45)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()
        data = resp.content

        # Próbáljuk először CSV-ként
        try:
            df = pd.read_csv(io.BytesIO(data))
        except Exception:
            df = None

        # Ha nem CSV, próbáljuk Excelként
        if df is None or df.empty or ("constituents" in [c.lower() for c in df.columns]):
            try:
                df = pd.read_excel(io.BytesIO(data))
            except Exception:
                pass

        if df is None or df.empty:
            # Próba: ha plain text-szerű, megkíséreljük a 'Symbol' / 'Ticker' oszlopot felismerni
            try:
                text = resp.text
                # Tömbös, whitespace-szel tagolt lehet; ez már eseti – hagyjuk inkább None-nak
            except Exception:
                pass
            return None

        # Keresünk 'Ticker' / 'Symbol' jellegű oszlopot
        cols_l = [c.lower() for c in df.columns]
        for cand in ("ticker", "symbol", "constituent ticker", "ric", "bbg ticker", "isin"):
            if cand in cols_l:
                series = df[df.columns[cols_l.index(cand)]]
                syms = _clean_tickers(series)
                if syms:
                    return syms

        # Ha nem találtunk ismert oszlopot, adjuk fel
        return None
    except Exception as e:
        log.warning(f"FTSE Russell R2000 hiba: {e}")
        return None

def fetch_r2000_from_slickcharts() -> Optional[List[str]]:
    """
    Slickcharts Russell 2000
    https://www.slickcharts.com/russell2000
    (Általában teljes táblát ad vissza HTML-ben.)
    """
    url = "https://www.slickcharts.com/russell2000"
    try:
        resp = requests.get(url, headers=HDRS, timeout=45)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        for tbl in tables:
            # a táblák egyikében jellemzően 'Symbol' oszlop van
            if "Symbol" in tbl.columns:
                syms = _clean_tickers(tbl["Symbol"])
                if len(syms) > 1000:  # sanity check
                    return syms
        return None
    except Exception as e:
        log.warning(f"Slickcharts R2000 hiba: {e}")
        return None

def get_russell2000() -> List[str]:
    syms = fetch_r2000_from_ftse()
    if not syms:
        log.info("FTSE Russell letöltés sikertelen / ismeretlen formátum; fallback: Slickcharts Russell 2000.")
        syms = fetch_r2000_from_slickcharts() or []
    return syms

# ---------- Main ----------

def main():
    sp = get_sp500()
    if sp:
        _write_csv("sp500.csv", sp)
    else:
        log.error("SP500 lista üres – nem írok fájlt.")

    r2k = get_russell2000()
    if r2k:
        _write_csv("russell2000.csv", r2k)
    else:
        log.error("Russell2000 lista üres – nem írok fájlt.")

if __name__ == "__main__":
    main()
