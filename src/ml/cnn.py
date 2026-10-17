"""Round 6: 1D-CNN on the last 120 one-minute bars + round-4 features (docs/STRATEGIES_R6.md, frozen)."""
from __future__ import annotations

# NOTE: never import LightGBM in a process that uses this module. On macOS both bring their own OpenMP
# runtime and the process segfaults once either trains (seen in the test suite).
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.ml.intraday_dataset import FEATURES
from src.ml.sequencing import sequence

SEQ = 120
THETAS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
SEEDS = (0, 1, 2)
MIN_NS = 60_000_000_000


def build_sequences(df: pd.DataFrame, m1: pd.DataFrame) -> np.ndarray:
    """(N, 4, 120) float32: the 120 complete 1m bars before each decision time t, scaled by ATR15.
    Rows without 120 bars of history get NaN (dropped later)."""
    m1 = m1.sort_index()
    ts = pd.DatetimeIndex(m1.index).as_unit("ns").asi8
    o, h, lo, c = (m1[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    v = np.log1p(np.clip(m1["volume"].to_numpy(dtype=float), 0, None))
    out = np.full((len(df), 4, SEQ), np.nan, dtype=np.float32)
    t = pd.DatetimeIndex(df["t"]).as_unit("ns").asi8
    A = df["atr15"].to_numpy(dtype=float)
    for n, (tn, a) in enumerate(zip(t, A, strict=False)):
        i = int(np.searchsorted(ts, tn))          # bars [i-120, i) all START before t -> complete at t
        if i < SEQ or not a > 0:
            continue
        sl = slice(i - SEQ, i)
        ct = c[i - 1]
        vv = v[sl]
        out[n, 0] = (c[sl] - ct) / a
        out[n, 1] = (h[sl] - lo[sl]) / a
        out[n, 2] = (c[sl] - o[sl]) / a
        out[n, 3] = vv - vv.mean()
    return out


class Net(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(4, 16, 5), nn.ReLU(), nn.Conv1d(16, 32, 5, stride=2), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(32 + n_feat, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, seq, feat):
        z = self.conv(seq).mean(dim=2)
        return self.head(torch.cat([z, feat], dim=1))          # logits for (long hits TP, short hits TP)


class Scaler:
    def __init__(self, X: np.ndarray):
        self.mu = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0) + 1e-9

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.nan_to_num((X - self.mu) / self.sd, nan=0.0).astype(np.float32)


def _targets(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([(df["r_long"] >= 1.49), (df["r_short"] >= 1.49)]).astype(np.float32)


def _train_one(seq, feat, y, seed, epochs=30, val=None, patience=5) -> tuple[Net, int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = Net(feat.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    S, F, Y = torch.from_numpy(seq), torch.from_numpy(feat), torch.from_numpy(y)
    best, best_ep, wait, best_state = np.inf, epochs, 0, None
    g = torch.Generator().manual_seed(seed)
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(S), generator=g)
        for b in range(0, len(S), 256):
            k = perm[b:b + 256]
            opt.zero_grad()
            loss = lossf(net(S[k], F[k]), Y[k])
            loss.backward()
            opt.step()
        if val is not None:
            net.eval()
            with torch.no_grad():
                vl = float(lossf(net(torch.from_numpy(val[0]), torch.from_numpy(val[1])), torch.from_numpy(val[2])))
            if vl < best - 1e-5:
                best, best_ep, wait = vl, ep, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_ep


def _predict(nets: list[Net], seq, feat) -> np.ndarray:
    ps = []
    with torch.no_grad():
        for n in nets:
            n.eval()
            ps.append(torch.sigmoid(n(torch.from_numpy(seq), torch.from_numpy(feat))).numpy())
    return np.mean(ps, axis=0)


class Ensemble:
    """3-seed CNN ensemble. fit(train, val): early stopping on val; refit(full, epochs) for the final models."""

    def __init__(self):
        self.nets: list[Net] = []
        self.epochs: list[int] = []
        self.scaler: Scaler | None = None

    def fit(self, tr: pd.DataFrame, seq_tr, va: pd.DataFrame | None = None, seq_va=None,
            epochs: list[int] | None = None) -> Ensemble:
        self.scaler = Scaler(tr[FEATURES].to_numpy(dtype=float))
        ftr, ytr = self.scaler(tr[FEATURES].to_numpy(dtype=float)), _targets(tr)
        val = None
        if va is not None:
            val = (seq_va, self.scaler(va[FEATURES].to_numpy(dtype=float)), _targets(va))
        self.nets, self.epochs = [], []
        for k, seed in enumerate(SEEDS):
            n, ep = _train_one(seq_tr, ftr, ytr, seed, epochs=(epochs[k] if epochs else 30), val=val)
            self.nets.append(n)
            self.epochs.append(ep)
        return self

    def predict(self, df: pd.DataFrame, seq) -> pd.DataFrame:
        p = _predict(self.nets, seq, self.scaler(df[FEATURES].to_numpy(dtype=float)))
        return pd.DataFrame({"p_long": p[:, 0], "p_short": p[:, 1]}, index=df.index)


def _chooser(pred: pd.DataFrame, theta: float):
    def choose(r):
        p = pred.loc[r.name]
        side, val = ("long", p["p_long"]) if p["p_long"] >= p["p_short"] else ("short", p["p_short"])
        return side if val >= theta else None
    return choose


def run_fold(df: pd.DataFrame, seqs: np.ndarray, start: str, end: str) -> dict:
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    ok = ~np.isnan(seqs).any(axis=(1, 2)) & df["r_long"].notna().to_numpy() & df["r_short"].notna().to_numpy()
    tr_m = ok & (df["t"] < s).to_numpy()
    te_m = ok & ((df["t"] >= s) & (df["t"] < e)).to_numpy()
    inner_cut = s - pd.DateOffset(months=12)
    in_m = tr_m & (df["t"] < inner_cut).to_numpy()
    va_m = tr_m & (df["t"] >= inner_cut).to_numpy()
    tr, va, te = df[tr_m], df[va_m], df[te_m]
    inner = Ensemble().fit(df[in_m], seqs[in_m], va, seqs[va_m])
    pv = inner.predict(va, seqs[va_m])
    theta, best = THETAS[-1], -np.inf
    for th in THETAS:
        t = sequence(va, _chooser(pv, th))
        if len(t) >= 50 and t["r"].mean() > best:
            theta, best = th, t["r"].mean()
    full = Ensemble().fit(tr, seqs[tr_m], epochs=inner.epochs)
    pt = full.predict(te, seqs[te_m])
    return {"fold": f"{start[:7]}..{end[:10]}", "theta": theta, "epochs": inner.epochs,
            "trades": {"cnn": sequence(te, _chooser(pt, theta))}}
