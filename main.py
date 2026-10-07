"""Fly Brain vs KSE-100: run the full pipeline end to end.

    python main.py                                   # yfinance, synthetic connectome
    python main.py --csv data/kse100.csv             # local price history
    python main.py --flywire connections.csv.gz      # real FlyWire sub-network
    python main.py --adjacency fly_2000.npz          # pre-distilled matrix

Steps: load KSE-100 -> causal sensory features -> fly-brain reservoir states
-> ridge readout fit on the first 80% -> threshold strategy on the last 20%
-> metrics table + fly_brain_vs_kse100.png.
"""

from __future__ import annotations

import argparse
import logging
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.psx_data import FEATURE_COLS, TARGET_COL, FeatureScaler, build_features, load_kse100
from src.reservoir import (
    FlyBrainReservoir,
    load_adjacency_npz,
    load_flywire_connections,
    synthetic_connectome,
)

TRADING_DAYS = 252
RIDGE_ALPHAS = np.logspace(-2, 5, 15)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = p.add_argument_group("data")
    data.add_argument("--csv", help="local OHLCV CSV instead of yfinance")
    data.add_argument("--start", default="2010-01-01")
    data.add_argument("--end", default=None)
    data.add_argument("--allow-synthetic", action="store_true",
                      help="fall back to a synthetic price series if no data source works")

    brain = p.add_argument_group("connectome / reservoir")
    src = brain.add_mutually_exclusive_group()
    src.add_argument("--flywire", help="FlyWire Codex connections.csv(.gz)")
    src.add_argument("--adjacency", help="pre-distilled signed adjacency .npz")
    brain.add_argument("--neurons", type=int, default=2000)
    brain.add_argument("--spectral-radius", type=float, default=0.9)
    brain.add_argument("--leak-rate", type=float, default=0.3)
    brain.add_argument("--input-scaling", type=float, default=0.5)
    brain.add_argument("--washout", type=int, default=100)
    brain.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    brain.add_argument("--seed", type=int, default=42)

    strat = p.add_argument_group("strategy")
    strat.add_argument("--train-frac", type=float, default=0.8)
    strat.add_argument("--threshold", type=float, default=0.002, help="predicted-return trigger (0.002 = 0.2%%)")
    strat.add_argument("--cost-bps", type=float, default=10.0, help="cost per unit of position change, in bps")
    strat.add_argument("--plot", default="fly_brain_vs_kse100.png")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Strategy & metrics
# --------------------------------------------------------------------------- #
def positions_from_predictions(pred: np.ndarray, threshold: float, allow_short: bool) -> np.ndarray:
    """+1 above +threshold, -1 (or 0 = cash) below -threshold, otherwise hold
    the previous position so a quiet day doesn't force a round trip."""
    pos = np.zeros(len(pred))
    current = 0.0
    for t, p in enumerate(pred):
        if p > threshold:
            current = 1.0
        elif p < -threshold:
            current = -1.0 if allow_short else 0.0
        pos[t] = current
    return pos


def backtest(positions: np.ndarray, fwd_log_ret: np.ndarray, cost_bps: float) -> np.ndarray:
    """Daily net simple returns. Position at row t is set at the close of day t
    and earns day t -> t+1."""
    simple = np.expm1(fwd_log_ret)
    turnover = np.abs(np.diff(np.r_[0.0, positions]))
    return positions * simple - turnover * cost_bps / 1e4


def performance(daily: np.ndarray, positions: np.ndarray | None = None) -> dict:
    equity = np.cumprod(1 + daily)
    years = len(daily) / TRADING_DAYS
    std = daily.std(ddof=1)
    out = {
        "Total return": equity[-1] - 1,
        "CAGR": equity[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "Ann. vol": std * np.sqrt(TRADING_DAYS),
        "Sharpe": daily.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else np.nan,
        "Max drawdown": (equity / np.maximum.accumulate(equity) - 1).min(),
    }
    if positions is not None:
        active = positions != 0
        out["Time in market"] = active.mean()
        out["Trades"] = int((np.diff(np.r_[0.0, positions]) != 0).sum())
        out["Hit rate"] = (daily[active] > 0).mean() if active.any() else np.nan
    return out


def signal_quality(pred: np.ndarray, actual: np.ndarray) -> dict:
    return {
        "Directional accuracy": float((np.sign(pred) == np.sign(actual)).mean()),
        "Information coeff. (corr)": float(np.corrcoef(pred, actual)[0, 1]) if pred.std() > 0 else np.nan,
        "Out-of-sample R^2": float(1 - ((actual - pred) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()),
        "Pred. std (bps)": float(pred.std() * 1e4),
        "Days |pred| > threshold": None,  # filled in by caller
    }


class RidgeReadout:
    """Standardize -> ridge, with alpha picked by walk-forward CV inside the
    training window. One SVD per fold scores every alpha at once, which is
    far cheaper than refitting a 2000-wide ridge per alpha."""

    def __init__(self, alphas: np.ndarray = RIDGE_ALPHAS, n_splits: int = 5):
        self.alphas = alphas
        self.n_splits = n_splits

    @staticmethod
    def _ridge_path(X, y, alphas):
        """Coefficients for every alpha: (n_alphas, n_features), plus intercept."""
        x_mean, y_mean = X.mean(axis=0), y.mean()
        U, s, Vt = np.linalg.svd(X - x_mean, full_matrices=False)
        uty = U.T @ (y - y_mean)
        coefs = np.stack([Vt.T @ (s / (s**2 + a) * uty) for a in alphas])
        return coefs, y_mean - coefs @ x_mean

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeReadout":
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        mse = np.zeros(len(self.alphas))
        for tr, va in TimeSeriesSplit(n_splits=self.n_splits).split(Xs):
            coefs, icpt = self._ridge_path(Xs[tr], y[tr], self.alphas)
            mse += ((Xs[va] @ coefs.T + icpt - y[va, None]) ** 2).mean(axis=0)
        self.alpha_ = float(self.alphas[np.argmin(mse)])
        coefs, icpt = self._ridge_path(Xs, y, [self.alpha_])
        self.coef_, self.intercept_ = coefs[0], float(icpt[0])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.scaler_.transform(X) @ self.coef_ + self.intercept_


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#52514e"]  # blue, orange, aqua, neutral ink


def make_plot(curves: dict[str, pd.Series], title: str, subtitle: str, path: str) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, ax_dd) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.08})
    fig.patch.set_facecolor("#fcfcfb")
    for a in (ax, ax_dd):
        a.set_facecolor("#fcfcfb")
        a.grid(axis="y", color="#e4e3df", linewidth=0.8)
        a.tick_params(colors="#52514e")
        for s in ("left", "bottom"):
            a.spines[s].set_color("#b5b4ae")

    for (name, equity), color in zip(curves.items(), SERIES_COLORS):
        lw = 2.4 if name.startswith("Fly Brain") else 1.8
        ls = "--" if name.startswith("Buy & Hold") else "-"
        ax.plot(equity.index, (equity - 1) * 100, color=color, lw=lw, ls=ls, label=name)
        ax.annotate(f"{(equity.iloc[-1] - 1) * 100:+.1f}%", xy=(equity.index[-1], (equity.iloc[-1] - 1) * 100),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color="#0b0b0b")
        dd = (equity / equity.cummax() - 1) * 100
        ax_dd.plot(dd.index, dd, color=color, lw=1.4, ls=ls)

    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.set_ylabel("Cumulative return (%)", color="#52514e")
    ax.legend(loc="upper left", frameon=False)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color="#0b0b0b", pad=24)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color="#52514e")
    ax_dd.set_ylabel("Drawdown (%)", color="#52514e")
    ax.margins(x=0.06)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("fly-brain")

    # 1-2. Data and sensory features --------------------------------------
    prices, source = load_kse100(args.csv, start=args.start, end=args.end, allow_synthetic=args.allow_synthetic)
    feats = build_features(prices)
    n = len(feats)
    split = int(n * args.train_frac)
    if split <= args.washout + 250 or n - split < 60:
        raise SystemExit(f"Not enough data ({n} rows) for washout={args.washout} and train_frac={args.train_frac}")
    log.info("Data: %s | %s -> %s | %d rows (train %d, test %d)",
             source, feats.index[0].date(), feats.index[-1].date(), n, split, n - split)

    scaler = FeatureScaler().fit(feats.iloc[:split][FEATURE_COLS])  # train stats only
    U = scaler.transform(feats[FEATURE_COLS])
    y = feats[TARGET_COL].to_numpy()

    # 3. Connectome ---------------------------------------------------------
    if args.flywire:
        connectome = load_flywire_connections(args.flywire, n_neurons=args.neurons)
    elif args.adjacency:
        connectome = load_adjacency_npz(args.adjacency)
    else:
        connectome = synthetic_connectome(n_neurons=args.neurons, seed=args.seed)
    stats = connectome.summary()
    log.info("Connectome: %s | %d neurons, %d edges, reciprocity %.2f, %.0f%% inhibitory edges",
             stats["source"], stats["neurons"], stats["synaptic_edges"],
             stats["reciprocity"], 100 * stats["inhibitory_edge_fraction"])

    # 4. Drive the reservoir. It is fixed and causal, so one pass over the full
    #    series is equivalent to running train then continuing into test.
    reservoir = FlyBrainReservoir(connectome, n_inputs=U.shape[1],
                                  spectral_radius_target=args.spectral_radius, leak_rate=args.leak_rate,
                                  input_scaling=args.input_scaling, seed=args.seed, backend=args.backend)
    states = reservoir.run(U)
    log.info("Reservoir: raw spectral radius %.2f -> %.2f, %d sensory neurons, mean |x| = %.3f",
             reservoir.rho_raw, reservoir.spectral_radius, len(reservoir.sensory_idx), np.abs(states).mean())

    # 5. Ridge readout on [neural state, raw input] ---------------------------
    X = np.hstack([states, U])
    train = slice(args.washout, split)
    test = slice(split, n)
    fly_model = RidgeReadout().fit(X[train], y[train])
    pred_fly = fly_model.predict(X[test])

    # Same readout without the brain: does the reservoir add anything?
    lin_model = RidgeReadout().fit(U[train], y[train])
    pred_lin = lin_model.predict(U[test])

    # 6. Strategy -------------------------------------------------------------
    y_test = y[test]
    idx = feats.index[test]
    pos_lc = positions_from_predictions(pred_fly, args.threshold, allow_short=False)
    pos_ls = positions_from_predictions(pred_fly, args.threshold, allow_short=True)
    pos_lin = positions_from_predictions(pred_lin, args.threshold, allow_short=False)
    pos_bh = np.ones(len(y_test))

    runs = {
        "Fly Brain (long/cash)": (pos_lc, backtest(pos_lc, y_test, args.cost_bps)),
        "Fly Brain (long/short)": (pos_ls, backtest(pos_ls, y_test, args.cost_bps)),
        "Ridge, no brain (long/cash)": (pos_lin, backtest(pos_lin, y_test, args.cost_bps)),
        "Buy & Hold KSE-100": (pos_bh, backtest(pos_bh, y_test, 0.0)),
    }
    table = pd.DataFrame({name: performance(r, p) for name, (p, r) in runs.items()})

    quality = pd.DataFrame({
        "Fly Brain": signal_quality(pred_fly, y_test),
        "Ridge, no brain": signal_quality(pred_lin, y_test),
    })
    quality.loc["Days |pred| > threshold"] = [
        int((np.abs(pred_fly) > args.threshold).sum()), int((np.abs(pred_lin) > args.threshold).sum())]

    # Output ------------------------------------------------------------------
    fmts = {"Total return": "{:+.1%}", "CAGR": "{:+.1%}", "Max drawdown": "{:+.1%}", "Ann. vol": "{:.1%}",
            "Time in market": "{:.0%}", "Hit rate": "{:.1%}", "Sharpe": "{:.2f}", "Trades": "{:.0f}"}
    shown = table.copy().astype(object)
    for row in shown.index:
        fmt = fmts[row]
        shown.loc[row] = [fmt.format(v) if pd.notna(v) else "-" for v in table.loc[row]]

    print("\n" + "=" * 78)
    print(f" FLY BRAIN RESERVOIR vs KSE-100   test window {idx[0].date()} -> {idx[-1].date()} ({len(idx)} days)")
    print(f" data: {source}")
    print(f" brain: {stats['source']}, {stats['neurons']} neurons, {stats['synaptic_edges']} synapses, "
          f"rho={reservoir.spectral_radius}, leak={args.leak_rate}")
    print(f" readout: ridge alpha={fly_model.alpha_:.3g} (brain), {lin_model.alpha_:.3g} (no brain); "
          f"threshold ±{args.threshold:.2%}; cost {args.cost_bps:g} bps")
    print("=" * 78)
    print(shown.to_string())
    print("\nSignal quality (test set, 1-day forward log return):")
    print(quality.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nSharpe uses a 0% risk-free rate; Pakistani T-bill yields are far from zero, "
          "so time in cash is understated.")
    if source.startswith("SYNTHETIC"):
        print("WARNING: synthetic prices - these numbers are a smoke test, not a result.")

    curves = {name: pd.Series(np.cumprod(1 + r), index=idx) for name, (_, r) in runs.items()}
    fly_sr, bh_sr = table.loc["Sharpe", "Fly Brain (long/cash)"], table.loc["Sharpe", "Buy & Hold KSE-100"]
    make_plot(
        curves,
        title="A fruit-fly connectome vs the KSE-100",
        subtitle=(f"Out-of-sample {idx[0].date()} – {idx[-1].date()}  ·  {stats['neurons']:,}-neuron "
                  f"{'FlyWire' if args.flywire else 'connectome'} reservoir + ridge readout  ·  "
                  f"Sharpe {fly_sr:.2f} vs {bh_sr:.2f} buy & hold  ·  {args.cost_bps:g} bps costs"
                  + ("  ·  SYNTHETIC DATA" if source.startswith("SYNTHETIC") else "")),
        path=args.plot,
    )
    print(f"\nPlot saved to {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
