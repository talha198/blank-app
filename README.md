# Fly Brain vs KSE-100

An Echo State Network whose recurrent weights are a fruit-fly connectome,
used to predict next-day returns of Pakistan's KSE-100 index.

```
KSE-100 OHLCV ─► sensory features ─► Fly Brain Reservoir ─► ridge readout ─► long / cash / short
 (yfinance/CSV)   log ret, RSI, vol,    2,000 neurons, tanh,    fit on first 80%    threshold ±0.2%
                  VWAP gap, volume      leaky, ρ = 0.9          test on last 20%
```

## Run it

```bash
pip install -r requirements.txt
python main.py                              # pulls ^KSE100 (then ^KSE) from yfinance
python main.py --csv data/kse100.csv        # or a local CSV (Yahoo / investing.com / PSX export)
python main.py --flywire connections.csv.gz # real FlyWire sub-network instead of the synthetic one
python main.py --help                       # all knobs: neurons, leak, threshold, costs, torch backend
```

Output: a metrics table in the terminal and `fly_brain_vs_kse100.png`.

## Files

| File | What it does |
|---|---|
| `src/psx_data.py` | Loads KSE-100 from a CSV or yfinance (caches to `data/kse100.csv`); builds causal features; z-scores them using **training-window statistics only**. |
| `src/reservoir.py` | `FlyBrainReservoir` (leaky `tanh` ESN, spectral radius 0.9, inputs only into afferent-like "sensory" neurons, numpy or torch). Topology from FlyWire, a `.npz` matrix, or `synthetic_connectome`. |
| `main.py` | End-to-end pipeline, backtest, metrics, plot. |

## Using the real fly brain

Download `connections.csv.gz` from [FlyWire Codex](https://codex.flywire.ai/api/download)
(free account; columns `pre_root_id, post_root_id, neuropil, syn_count, nt_type`).
`load_flywire_connections` streams it twice, keeps the N neurons with the most
synapses (≥ 5 synapses per edge, the FlyWire convention), signs each edge by the
presynaptic neurotransmitter (ACh +, GABA/Glu −) and compresses counts with `log1p`.
Save the distilled matrix with `save_adjacency_npz` and reuse it via `--adjacency`.

Without that file the pipeline uses `synthetic_connectome`: a random graph with
FlyWire-like statistics (neuropil modules, hub neurons, lognormal synapse counts,
reciprocal pairs, Dale's law). It's labelled as synthetic everywhere it appears.

## Reading the results honestly

- **Baseline included.** "Ridge, no brain" is the same readout on the raw features.
  If the fly brain doesn't beat it, the reservoir isn't adding anything.
- **The 0.2% threshold is strict.** A ridge model of daily returns predicts small
  numbers; on many runs no prediction clears ±0.2% and the strategy sits in cash.
  The table reports how many days crossed the threshold. Lower it with `--threshold`.
- **Costs** default to 10 bps per unit of position change (`--cost-bps`).
- **Sharpe** assumes a 0% risk-free rate; Pakistani T-bill yields are high, so this
  flatters buy & hold relative to cash.
- **Shorting** the KSE-100 isn't practical for most investors; long/cash is the
  headline strategy and long/short is shown for comparison.
- One train/test split on one index is a demo, not evidence of an edge. Not financial advice.
