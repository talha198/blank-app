"""Fly Brain Reservoir: an Echo State Network whose recurrent weights are a
Drosophila connectome (or a synthetic graph with the same statistics).

Three ways to get a topology:
  * ``load_flywire_connections`` - distil an N-neuron sub-network from the
    FlyWire Codex ``connections.csv(.gz)`` export (columns ``pre_root_id``,
    ``post_root_id``, ``syn_count`` and optionally ``nt_type``).
  * ``load_adjacency_npz`` - a pre-distilled signed sparse matrix saved with
    ``scipy.sparse.save_npz`` (rows = presynaptic, cols = postsynaptic).
  * ``synthetic_connectome`` - a FlyWire-like random graph (neuropil modules,
    heavy-tailed degrees and synapse counts, reciprocity, Dale's law with
    ACh/GABA/Glu identities) for when the real data isn't on disk.

Dynamics (leaky integrator):
    x[t+1] = (1 - a) * x[t] + a * tanh(W x[t] + W_in u[t] + b)
with W rescaled so its spectral radius is 0.9 by default.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, eigs

log = logging.getLogger(__name__)

# Sign of each neurotransmitter as FlyWire labels them. In the fly, glutamate
# acts mostly through the inhibitory GluCl channel, so GLUT counts as -1.
# Monoamines are modulatory; we treat them as weak excitation.
NT_SIGN = {"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0, "DA": 0.5, "SER": 0.5, "OCT": 0.5}
NT_FRACTIONS = {"ACH": 0.62, "GABA": 0.18, "GLUT": 0.15, "DA": 0.02, "SER": 0.015, "OCT": 0.015}


@dataclass
class Connectome:
    """Signed, weighted directed graph. ``adjacency[i, j]`` is the synapse
    weight from presynaptic neuron i onto postsynaptic neuron j."""

    adjacency: sp.csr_matrix
    nt_types: np.ndarray
    source: str
    modules: np.ndarray | None = None
    neuron_ids: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_neurons(self) -> int:
        return self.adjacency.shape[0]

    def to_networkx(self) -> nx.DiGraph:
        g = nx.from_scipy_sparse_array(self.adjacency, create_using=nx.DiGraph)
        nx.set_node_attributes(g, dict(enumerate(self.nt_types)), "nt_type")
        return g

    def summary(self) -> dict:
        g = self.to_networkx()
        n, m = g.number_of_nodes(), g.number_of_edges()
        largest_scc = max((len(c) for c in nx.strongly_connected_components(g)), default=0)
        weights = self.adjacency.data
        return {
            "source": self.source,
            "neurons": n,
            "synaptic_edges": m,
            "density": m / (n * (n - 1)) if n > 1 else 0.0,
            "mean_out_degree": m / n if n else 0.0,
            "reciprocity": nx.reciprocity(g) if m else 0.0,
            "largest_scc_fraction": largest_scc / n if n else 0.0,
            "inhibitory_edge_fraction": float((weights < 0).mean()) if m else 0.0,
        }


# --------------------------------------------------------------------------- #
# Topology sources
# --------------------------------------------------------------------------- #
def synthetic_connectome(
    n_neurons: int = 2000,
    n_modules: int = 12,
    mean_out_degree: float = 12.0,
    p_within_module: float = 0.8,
    reciprocity: float = 0.12,
    seed: int = 0,
) -> Connectome:
    """FlyWire-flavoured random graph. Not the real fly, but it shares the
    features that matter for reservoir dynamics: modular neuropils, hub
    neurons (lognormal degree propensities), lognormal synapse counts,
    over-represented reciprocal pairs and Dale's-law signed outputs."""
    rng = np.random.default_rng(seed)

    sizes = rng.dirichlet(np.full(n_modules, 2.0))
    modules = rng.choice(n_modules, size=n_neurons, p=sizes)
    out_prop = rng.lognormal(0.0, 1.0, n_neurons)
    in_prop = rng.lognormal(0.0, 0.8, n_neurons)
    n_out = rng.poisson(mean_out_degree * out_prop / out_prop.mean())

    members = [np.flatnonzero(modules == k) for k in range(n_modules)]
    p_global = in_prop / in_prop.sum()
    p_local = [in_prop[m] / in_prop[m].sum() if len(m) else None for m in members]

    pre, post = [], []
    for i in range(n_neurons):
        k = min(n_out[i], n_neurons - 1)
        if k == 0:
            continue
        n_local = rng.binomial(k, p_within_module) if len(members[modules[i]]) > 1 else 0
        local = rng.choice(members[modules[i]], size=n_local, p=p_local[modules[i]])
        glob = rng.choice(n_neurons, size=k - n_local, p=p_global)
        targets = np.unique(np.concatenate([local, glob]))
        targets = targets[targets != i]
        pre.append(np.full(len(targets), i))
        post.append(targets)
    pre, post = np.concatenate(pre), np.concatenate(post)

    # Over-represent reciprocal connections, as seen in FlyWire.
    back = rng.random(len(pre)) < reciprocity
    pre, post = np.concatenate([pre, post[back]]), np.concatenate([post, pre[back]])

    nt_names = np.array(list(NT_FRACTIONS))
    nt_types = rng.choice(nt_names, size=n_neurons, p=list(NT_FRACTIONS.values()))
    syn_count = 5 + np.floor(rng.lognormal(1.2, 1.0, len(pre)))

    adjacency = sp.coo_matrix((syn_count, (pre, post)), shape=(n_neurons, n_neurons)).tocsr()
    adjacency.sum_duplicates()
    adjacency = _sign_and_compress(adjacency, nt_types)
    return Connectome(adjacency, nt_types, "synthetic FlyWire-like graph", modules=modules,
                      meta={"seed": seed, "n_modules": n_modules})


def load_flywire_connections(
    path: str | Path,
    n_neurons: int = 2000,
    min_syn: int = 5,
    chunksize: int = 2_000_000,
) -> Connectome:
    """Distil the ``n_neurons`` most strongly connected neurons from a FlyWire
    Codex connections export. Two streaming passes, so the full ~1 GB table
    never has to sit in memory at once."""
    usecols = lambda c: c in {"pre_root_id", "post_root_id", "syn_count", "nt_type"}  # noqa: E731

    # Pass 1: total synapse degree per neuron (in + out).
    degree = pd.Series(dtype=np.float64)
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk[chunk["syn_count"] >= min_syn]
        d = chunk.groupby("pre_root_id")["syn_count"].sum().add(
            chunk.groupby("post_root_id")["syn_count"].sum(), fill_value=0)
        degree = degree.add(d, fill_value=0)
    keep = degree.nlargest(n_neurons).index.to_numpy()
    index = pd.Series(np.arange(len(keep)), index=keep)

    # Pass 2: edges among the selected neurons.
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk[(chunk["syn_count"] >= min_syn)
                      & chunk["pre_root_id"].isin(index.index)
                      & chunk["post_root_id"].isin(index.index)]
        parts.append(chunk)
    edges = pd.concat(parts, ignore_index=True)
    if "nt_type" not in edges.columns:
        edges["nt_type"] = "ACH"
    edges["nt_type"] = edges["nt_type"].fillna("ACH").str.upper()

    # FlyWire lists one row per (pre, post, neuropil); merge neuropils.
    pairs = edges.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"].sum()
    # A neuron's transmitter is the one carrying most of its output synapses.
    nt_by_neuron = (edges.groupby(["pre_root_id", "nt_type"])["syn_count"].sum()
                    .reset_index().sort_values("syn_count")
                    .drop_duplicates("pre_root_id", keep="last")
                    .set_index("pre_root_id")["nt_type"])
    nt_types = nt_by_neuron.reindex(keep).fillna("ACH").to_numpy()

    adjacency = sp.coo_matrix(
        (pairs["syn_count"].to_numpy(float),
         (index[pairs["pre_root_id"]].to_numpy(), index[pairs["post_root_id"]].to_numpy())),
        shape=(len(keep), len(keep)),
    ).tocsr()
    adjacency = _sign_and_compress(adjacency, nt_types)
    return Connectome(adjacency, nt_types, f"FlyWire: {Path(path).name}", neuron_ids=keep,
                      meta={"min_syn": min_syn})


def load_adjacency_npz(path: str | Path, nt_types_path: str | Path | None = None) -> Connectome:
    """Load a pre-distilled signed adjacency matrix (rows = pre, cols = post)."""
    adjacency = sp.load_npz(path).tocsr().astype(np.float64)
    if nt_types_path:
        nt_types = np.load(nt_types_path, allow_pickle=True)
    else:  # infer from sign of each neuron's outputs
        row_sum = np.asarray(adjacency.sum(axis=1)).ravel()
        nt_types = np.where(row_sum < 0, "GABA", "ACH")
    return Connectome(adjacency, nt_types, f"pre-distilled: {Path(path).name}")


def save_adjacency_npz(connectome: Connectome, path: str | Path) -> None:
    sp.save_npz(path, connectome.adjacency)
    np.save(Path(path).with_suffix(".nt_types.npy"), connectome.nt_types)


def _sign_and_compress(syn_counts: sp.csr_matrix, nt_types: np.ndarray) -> sp.csr_matrix:
    """Synapse counts -> signed weights. log1p compresses the heavy tail so a
    single 500-synapse edge can't dominate the dynamics; the row sign comes
    from the presynaptic neuron's transmitter (Dale's law)."""
    signs = np.array([NT_SIGN.get(str(nt).upper(), 1.0) for nt in nt_types])
    w = syn_counts.tocsr().copy()
    w.data = np.log1p(w.data)
    w = sp.diags(signs) @ w
    w.setdiag(0)
    w.eliminate_zeros()
    return w.tocsr()


# --------------------------------------------------------------------------- #
# Reservoir
# --------------------------------------------------------------------------- #
def spectral_radius(w: sp.spmatrix) -> float:
    """Largest |eigenvalue|. Dense solve for small graphs, ARPACK otherwise."""
    n = w.shape[0]
    if n <= 2500:
        return float(np.max(np.abs(np.linalg.eigvals(w.toarray()))))
    try:
        vals = eigs(w.astype(np.float64), k=1, which="LM", return_eigenvectors=False,
                    maxiter=n * 100, tol=1e-6)
        return float(np.abs(vals[0]))
    except ArpackNoConvergence as exc:
        if len(exc.eigenvalues):
            return float(np.max(np.abs(exc.eigenvalues)))
        raise


class FlyBrainReservoir:
    """Leaky-integrator ESN with connectome recurrent weights.

    Inputs enter only through "sensory" neurons - the ones whose out-degree
    most exceeds their in-degree (afferent-like) - instead of being sprayed
    onto every unit as in a textbook ESN.
    """

    def __init__(
        self,
        connectome: Connectome,
        n_inputs: int,
        spectral_radius_target: float = 0.9,
        leak_rate: float = 0.3,
        input_scaling: float = 0.5,
        sensory_fraction: float = 0.15,
        bias_scaling: float = 0.1,
        seed: int = 0,
        backend: str = "numpy",
        device: str | None = None,
    ):
        if not 0 < leak_rate <= 1:
            raise ValueError("leak_rate must be in (0, 1]")
        self.connectome = connectome
        self.n_neurons = connectome.n_neurons
        self.n_inputs = n_inputs
        self.leak_rate = leak_rate
        self.backend = backend
        self.device = device
        rng = np.random.default_rng(seed)

        # Recurrent matrix: state update uses W @ x with W[post, pre].
        w = connectome.adjacency.T.tocsr().astype(np.float64)
        rho = spectral_radius(w)
        if rho == 0:
            raise ValueError("Connectome has spectral radius 0 (feed-forward graph?)")
        self.rho_raw = rho
        self.W = (w * (spectral_radius_target / rho)).tocsr()
        self.spectral_radius = spectral_radius_target

        out_deg = np.diff(connectome.adjacency.indptr)
        in_deg = np.diff(connectome.adjacency.tocsc().indptr)
        afferentness = (out_deg + 1) / (in_deg + 1)
        n_sensory = max(n_inputs, int(round(sensory_fraction * self.n_neurons)))
        self.sensory_idx = np.argsort(-afferentness, kind="stable")[:n_sensory]

        self.W_in = np.zeros((self.n_neurons, n_inputs))
        self.W_in[self.sensory_idx] = rng.uniform(-1, 1, (n_sensory, n_inputs)) * input_scaling
        self.bias = rng.uniform(-1, 1, self.n_neurons) * bias_scaling
        self.state = np.zeros(self.n_neurons)

    def reset(self) -> None:
        self.state = np.zeros(self.n_neurons)

    def run(self, inputs: np.ndarray, reset: bool = True) -> np.ndarray:
        """Drive the reservoir with ``inputs`` (T, n_inputs); return states (T, N).
        State t is the response *after* seeing input t."""
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != self.n_inputs:
            raise ValueError(f"inputs must be (T, {self.n_inputs}), got {inputs.shape}")
        if reset:
            self.reset()
        if self.backend == "torch":
            return self._run_torch(inputs)

        drive = inputs @ self.W_in.T + self.bias  # input drive for all steps at once
        a = self.leak_rate
        x = self.state
        states = np.empty((len(inputs), self.n_neurons))
        for t in range(len(inputs)):
            x = (1 - a) * x + a * np.tanh(self.W @ x + drive[t])
            states[t] = x
        self.state = x
        return states

    def _run_torch(self, inputs: np.ndarray) -> np.ndarray:
        import warnings

        import torch

        warnings.filterwarnings("ignore", message=".*[Ss]parse.*", category=UserWarning)
        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        w = self.W.tocsr()
        W = torch.sparse_csr_tensor(
            torch.from_numpy(w.indptr.astype(np.int64)),
            torch.from_numpy(w.indices.astype(np.int64)),
            torch.from_numpy(w.data.astype(np.float32)),
            size=w.shape, device=device,
        )
        drive = torch.from_numpy((inputs @ self.W_in.T + self.bias).astype(np.float32)).to(device)
        x = torch.from_numpy(self.state.astype(np.float32)).to(device)
        a = self.leak_rate
        states = torch.empty((len(inputs), self.n_neurons), device=device)
        with torch.no_grad():
            for t in range(len(inputs)):
                x = (1 - a) * x + a * torch.tanh(W @ x + drive[t])
                states[t] = x
        self.state = x.cpu().numpy().astype(np.float64)
        return states.cpu().numpy().astype(np.float64)
