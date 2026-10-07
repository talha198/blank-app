"""KSE-100 data loading and sensory-feature engineering.

Data sources, in order of preference:
  1. A local CSV (Yahoo export, investing.com export, or PSX dump).
  2. yfinance, trying several tickers Yahoo has used for the KSE-100.
  3. A seeded synthetic series (only if explicitly allowed), so the pipeline
     can be smoke-tested offline. Results on synthetic data mean nothing.

All features are computed causally: the value at row t only uses data up to
and including the close of day t. The target is the *next* day's log return.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

KSE100_TICKERS = ("^KSE100", "^KSE", "KSE100.KA")
PRICE_COLS = ("Open", "High", "Low", "Close", "Volume")

FEATURE_COLS = ["log_ret", "rsi", "volatility", "vwap_dev", "volume_z", "hl_range"]
TARGET_COL = "target"

_COLUMN_ALIASES = {
    "date": "Date",
    "time": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "price": "Close",
    "adj close": "Adj Close",
    "volume": "Volume",
    "vol.": "Volume",
    "vol": "Volume",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _parse_number(value) -> float:
    """Parse '1,234.5', '250.3M', '1.2B', '-' into floats."""
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if value is None:
        return np.nan
    s = str(value).strip().replace(",", "")
    if s in {"", "-", "nan", "NaN", "null"}:
        return np.nan
    m = re.fullmatch(r"(-?[\d.]+)\s*([KMB]?)", s, flags=re.IGNORECASE)
    if not m:
        return np.nan
    mult = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9}[m.group(2).upper()]
    return float(m.group(1)) * mult


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: _COLUMN_ALIASES.get(str(c).strip().lower(), c) for c in df.columns})
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "Date"

    if "Close" not in df.columns:
        raise ValueError(f"No Close/Price column found; columns are {list(df.columns)}")
    for col in PRICE_COLS:
        if col not in df.columns:
            df[col] = np.nan if col != "Volume" else 0.0
        df[col] = df[col].map(_parse_number) if df[col].dtype == object else df[col].astype(float)

    # Fill missing OHLC from Close so downstream indicators always work.
    for col in ("Open", "High", "Low"):
        df[col] = df[col].fillna(df["Close"])
    df["Volume"] = df["Volume"].fillna(0.0)

    df = df[list(PRICE_COLS)].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[df["Close"] > 0]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a local OHLCV CSV with forgiving column names and number formats."""
    raw = pd.read_csv(path, thousands=",")
    return _standardize_ohlcv(raw)


def download_yfinance(
    tickers: tuple[str, ...] = KSE100_TICKERS,
    start: str = "2010-01-01",
    end: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Try each ticker until one returns data. Raises RuntimeError if none do."""
    import yfinance as yf

    errors = []
    for ticker in tickers:
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        except Exception as exc:  # network errors, delisted symbols, rate limits
            errors.append(f"{ticker}: {exc}")
            continue
        if raw is None or raw.empty:
            errors.append(f"{ticker}: empty response")
            continue
        if isinstance(raw.columns, pd.MultiIndex):  # yfinance >= 0.2.40 returns (field, ticker)
            raw.columns = raw.columns.get_level_values(0)
        return _standardize_ohlcv(raw.reset_index()), ticker
    raise RuntimeError("yfinance returned no KSE-100 data:\n  " + "\n  ".join(errors))


def synthetic_kse100(n_days: int = 3500, seed: int = 7, start: str = "2011-01-03") -> pd.DataFrame:
    """GARCH(1,1)-style random walk with KSE-100-like level and volatility.

    For offline smoke tests only. There is no real signal in it.
    """
    rng = np.random.default_rng(seed)
    omega, alpha, beta, mu = 2e-6, 0.10, 0.88, 4e-4
    var = omega / (1 - alpha - beta)
    rets = np.empty(n_days)
    for t in range(n_days):
        eps = rng.standard_t(df=5) / np.sqrt(5 / 3)
        rets[t] = mu + np.sqrt(var) * eps
        var = omega + alpha * (rets[t] - mu) ** 2 + beta * var
    close = 11_000 * np.exp(np.cumsum(rets))
    intraday = np.abs(rng.normal(0, 0.006, n_days)) * close
    high = close + intraday * rng.uniform(0.2, 1.0, n_days)
    low = close - intraday * rng.uniform(0.2, 1.0, n_days)
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n_days))
    volume = rng.lognormal(mean=np.log(2.5e8), sigma=0.45, size=n_days)
    idx = pd.bdate_range(start=start, periods=n_days, name="Date")
    return pd.DataFrame(
        {"Open": open_, "High": np.maximum.reduce([high, open_, close]),
         "Low": np.minimum.reduce([low, open_, close]), "Close": close, "Volume": volume},
        index=idx,
    )


def load_kse100(
    csv_path: str | Path | None = None,
    start: str = "2010-01-01",
    end: str | None = None,
    tickers: tuple[str, ...] = KSE100_TICKERS,
    cache_path: str | Path | None = "data/kse100.csv",
    allow_synthetic: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Load KSE-100 OHLCV. Returns (dataframe, human-readable source label)."""
    if csv_path:
        df = load_csv(csv_path)
        source = f"CSV: {csv_path}"
    else:
        try:
            df, ticker = download_yfinance(tickers, start=start, end=end)
            source = f"yfinance: {ticker}"
            if cache_path:
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_path)
                log.info("Cached download to %s", cache_path)
        except (RuntimeError, ImportError) as exc:
            if cache_path and Path(cache_path).exists():
                log.warning("Download failed, using cached %s", cache_path)
                df, source = load_csv(cache_path), f"cached CSV: {cache_path}"
            elif allow_synthetic:
                log.warning("%s\nFalling back to SYNTHETIC data.", exc)
                return synthetic_kse100(), "SYNTHETIC (GARCH random walk, no real signal)"
            else:
                raise

    df = df.loc[start:end] if end else df.loc[start:]
    if len(df) < 300:
        raise ValueError(f"Only {len(df)} rows of KSE-100 data; need at least 300.")
    return df, source


# --------------------------------------------------------------------------- #
# Sensory features
# --------------------------------------------------------------------------- #
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(loss.notna())


def rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling VWAP of the typical price. Falls back to equal weights when
    the feed has no volume (Yahoo often reports 0 volume for indices)."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].where(df["Volume"] > 0)
    if vol.notna().mean() < 0.5:
        return typical.rolling(window).mean()
    vol = vol.fillna(vol.rolling(window, min_periods=1).median())
    return (typical * vol).rolling(window).sum() / vol.rolling(window).sum()


def build_features(
    df: pd.DataFrame,
    rsi_window: int = 14,
    vol_window: int = 20,
    vwap_window: int = 20,
) -> pd.DataFrame:
    """Raw (un-normalized) sensory channels plus the 1-day-forward target."""
    close = df["Close"]
    out = pd.DataFrame(index=df.index)
    out["log_ret"] = np.log(close).diff()
    out["rsi"] = (rsi(close, rsi_window) - 50) / 50  # -> [-1, 1]
    out["volatility"] = out["log_ret"].rolling(vol_window).std() * np.sqrt(252)
    out["vwap_dev"] = close / rolling_vwap(df, vwap_window) - 1
    log_vol = np.log(df["Volume"].where(df["Volume"] > 0))
    if log_vol.notna().mean() < 0.5:
        out["volume_z"] = 0.0
    else:
        mean = log_vol.rolling(vol_window).mean()
        std = log_vol.rolling(vol_window).std()
        out["volume_z"] = ((log_vol - mean) / std).fillna(0.0)
    out["hl_range"] = np.log(df["High"] / df["Low"])
    out[TARGET_COL] = out["log_ret"].shift(-1)
    out["close"] = close
    # Keep the last row (target NaN) out: it has nothing to be scored against.
    return out.dropna(subset=FEATURE_COLS + [TARGET_COL])


class FeatureScaler:
    """Z-score using statistics from the training window only (no look-ahead),
    then clip so a crash day cannot saturate every neuron."""

    def __init__(self, clip: float = 4.0):
        self.clip = clip
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "FeatureScaler":
        self.mean_ = X.mean()
        self.std_ = X.std().replace(0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        z = (X - self.mean_) / self.std_
        return np.clip(z.to_numpy(dtype=np.float64), -self.clip, self.clip)
