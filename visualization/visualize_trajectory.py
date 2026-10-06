"""Build a weight-trajectory report for one federated run.

Reads what fl_server.py wrote into results/<trial-tag>/ and renders:

  * the global model's path through weight space, with each site's local
    update drawn as a pull from the global model it started from (PCA to 3D)
  * pairwise cosine alignment of the site update deltas, per round
  * the metrics the server already logs (metrics_global.csv and
    metrics_per_site.csv): the global model on each site's validation set,
    and each site's update on its own training data

The weight panels need the run to have been made with save-round-weights = true
(round_weights/round_NNN_server.pth and round_NNN_client_<site>.pth). Without
them the report still renders the metric panels.

Adapted from the CIFAR-10 example (fed_cifar10.zip). The differences that
matter:

  * No shared test set. The example scores every model on the same 10,000
    CIFAR images; here images never leave a site, so the metric panels use
    what each site reported. A site's numbers are on its own data and are not
    comparable across sites.
  * Memory. The example stacks every flattened checkpoint into one matrix,
    which for RETFound (ViT-L, 300M parameters) over 60 rounds x 3 files is
    ~200 GB. Here the checkpoints are memory-mapped and streamed in chunks
    into an N x N Gram matrix; PCA and every cosine come out of that matrix
    exactly, so peak memory is a few hundred MB regardless of backbone.

Outputs (written to --out-dir, default the run directory):
  <trial>_report.html     interactive page, self-contained if plotly is installed
  trajectory_3d.png       static 3D plot of the same trajectory
  trajectory_pca.csv      PCA coordinates of every checkpoint
  client_similarity.csv   pairwise cosine of the site update deltas
"""

import argparse
import csv
import html
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(__file__).resolve().parent / "assets"

# Same eight-slot categorical order the page uses, so the swatches in the
# table view and the PNG match the marks in the charts.
SERIES_LIGHT = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

# BatchNorm buffers are not learnable parameters, and they wreck weight-space
# geometry: running_var sits at O(1) while weights sit at O(1e-2), and
# num_batches_tracked is an int64 counter in the thousands. On ResNet-50 they
# account for nearly all the variance between checkpoints, so both the PCA and
# the cosine alignment would describe the buffers instead of the model.
# RETFound has no BatchNorm, so this is a no-op there.
BUFFER_KEYS = ("running_mean", "running_var", "num_batches_tracked")

# Elements per chunk summed over all N checkpoints: at float64 this is 512 MB
# of working matrix, whatever the backbone.
CHUNK_ELEMS = 1 << 26

SERVER_RE = re.compile(r"round_(\d+)_server\.pth$")
CLIENT_RE = re.compile(r"round_(\d+)_client_(.+)\.pth$")


# --------------------------------------------------------------- run loading


def find_run_dir(results_root: Path, trial: str | None) -> Path | None:
    """Resolve the run to report on, defaulting to the most recently modified."""
    if trial:
        for candidate in (Path(trial), results_root / trial):
            if candidate.is_dir():
                return candidate
        return None
    runs = [p for p in results_root.iterdir() if p.is_dir()] if results_root.is_dir() else []
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def load_site_names(pyproject: Path) -> dict[str, str]:
    """site id -> display-name from [tool.fl.sites.<id>]; empty if unavailable."""
    try:
        with open(pyproject, "rb") as fh:
            doc = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    sites = doc.get("tool", {}).get("fl", {}).get("sites", {})
    return {sid: str(entry.get("display-name", sid)) for sid, entry in sites.items()}


def read_backbone(run_dir: Path) -> str:
    """The server logs 'backbone=<name>' at startup; take the last one."""
    path = run_dir / "log.log"
    if not path.exists():
        return ""
    found = ""
    with open(path, errors="replace") as fh:
        for line in fh:
            m = re.search(r"\bbackbone=(\S+)", line)
            if m:
                found = m.group(1)
    return found


def index_checkpoints(run_dir: Path):
    """
    Find the per-round checkpoints. Returns (servers, clients) where servers is
    {round: path} and clients is {round: {site: path}}.

    A client update in round r is only placeable relative to the global model
    it started from (round r-1), and the round-r server file is what proves
    the round completed, so client files from a round with no server file are
    dropped (e.g. a run that aborted mid-save).
    """
    wdir = run_dir / "round_weights"
    servers, clients = {}, {}
    if not wdir.is_dir():
        return servers, clients
    for p in sorted(wdir.iterdir()):
        m = SERVER_RE.match(p.name)
        if m:
            servers[int(m.group(1))] = p
            continue
        m = CLIENT_RE.match(p.name)
        if m:
            clients.setdefault(int(m.group(1)), {})[m.group(2)] = p

    for r in sorted(set(clients) - set(servers)):
        print(f"  ! round {r}: client weights but no server weights, skipping round")
        del clients[r]
    if servers and 0 not in servers:
        print("  ! round_000_server.pth missing (run predates save_initial_weights): "
              "round 1 updates have no origin, so the path starts at round 1 and "
              "round 1 is left out of the alignment panels")
    return servers, clients


def load_metrics(run_dir: Path):
    """
    Read metrics_global.csv and metrics_per_site.csv into two blocks the page
    plots, one per split:

      val:   the round-r global model scored on each site's validation set,
             global = the n-weighted mean across sites
      train: each site's update scored on its own training data during local
             training, global = the n-weighted mean across sites

    If a round appears twice (a resumed run) the last row wins.
    """
    def num(value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    blocks = {split: {"global": [], "clients": {}} for split in ("val", "train")}
    site_rows = []

    gpath = run_dir / "metrics_global.csv"
    if gpath.exists():
        by_round = {}
        with open(gpath, newline="") as fh:
            for row in csv.DictReader(fh):
                by_round[int(row["round"])] = row
        for r in sorted(by_round):
            row = by_round[r]
            for split in ("val", "train"):
                blocks[split]["global"].append({
                    "round": r,
                    "loss": num(row.get(f"{split}_loss")),
                    "acc": num(row.get(f"{split}_acc")),
                    "auc": num(row.get(f"{split}_auc")),
                })

    spath = run_dir / "metrics_per_site.csv"
    if spath.exists():
        by_key = {}
        with open(spath, newline="") as fh:
            for row in csv.DictReader(fh):
                by_key[(int(row["round"]), row["site"], row["split"])] = row
        for (r, site, split), row in sorted(by_key.items()):
            if split not in blocks:
                continue
            n = num(row.get("n"))
            rec = {
                "round": r,
                "n": int(n) if n is not None else None,
                "loss": num(row.get("loss")),
                "acc": num(row.get("accuracy")),
                "auc": num(row.get("auc")),
            }
            blocks[split]["clients"].setdefault(site, []).append(rec)
            site_rows.append({"site": site, "split": split, **rec})

    return blocks, site_rows


# ------------------------------------------------------------ Gram matrix


def is_learnable(key: str) -> bool:
    return not key.endswith(BUFFER_KEYS)


def gram_matrix(paths: list[Path]) -> np.ndarray:
    """
    G[i, j] = <w_i - w_0, w_j - w_0> over the learnable parameters, in float64.

    Offsetting by the first checkpoint is what keeps float64 exact enough:
    raw weight vectors are nearly parallel (every checkpoint shares the same
    pretrained backbone), so their raw inner products are huge and the small
    differences that the PCA and cosines depend on would cancel away. Since
    every quantity downstream is built from differences of checkpoints, the
    offset drops out of all of them.

    Files are opened memory-mapped, so only the current chunk is ever in RAM.
    """
    n = len(paths)
    sds = [torch.load(p, map_location="cpu", mmap=True, weights_only=True) for p in paths]

    keys = [k for k in sds[0] if is_learnable(k)]
    for p, sd in zip(paths[1:], sds[1:]):
        other = [k for k in sd if is_learnable(k)]
        if set(other) != set(keys):
            missing = sorted(set(keys) - set(other))[:3]
            extra = sorted(set(other) - set(keys))[:3]
            raise SystemExit(f"{p.name}: parameter names differ from {paths[0].name} "
                             f"(missing {missing}, extra {extra}) — mixed backbones?")
        for k in keys:
            if sd[k].shape != sds[0][k].shape:
                raise SystemExit(f"{p.name}: '{k}' has shape {tuple(sd[k].shape)}, "
                                 f"expected {tuple(sds[0][k].shape)}")

    total = sum(sds[0][k].numel() for k in keys)
    chunk = max(1, CHUNK_ELEMS // n)
    print(f"  Gram matrix: {n} checkpoints x {total:,} learnable parameters "
          f"({len(keys)} tensors)")

    G = torch.zeros((n, n), dtype=torch.float64)
    done, t0, last_print = 0, time.time(), 0.0
    for k in keys:
        flats = [sd[k].reshape(-1) for sd in sds]
        numel = flats[0].numel()
        for s in range(0, numel, chunk):
            e = min(s + chunk, numel)
            X = torch.stack([f[s:e].to(torch.float64) for f in flats])
            X = X - X[0:1]
            G += X @ X.T
            done += e - s
            now = time.time()
            if now - last_print > 5 or done == total:
                last_print = now
                print(f"\r    {done / total:6.1%}  ({now - t0:.0f}s)", end="", flush=True)
    print()
    return G.numpy()


def pair_dot(G, a, a0, b, b0) -> float:
    """<w_a - w_a0, w_b - w_b0> from the offset Gram matrix."""
    return float(G[a, b] - G[a, b0] - G[a0, b] + G[a0, b0])


def kernel_pca(G: np.ndarray, anchor: int):
    """
    Exact PCA from the Gram matrix: double-centre it, eigendecompose, and scale
    the eigenvectors by sqrt(eigenvalue). Same coordinates as sklearn's PCA on
    the stacked vectors, up to the sign of each component.

    PCA signs are arbitrary, so each component is flipped to put `anchor` (the
    final global model) on its positive side; re-running on a longer run then
    doesn't mirror the plot.

    Returns (coords [N x 3], explained [<=3]). With fewer than 4 points the
    trailing components are zero-padded so every point still has x, y, z.
    """
    n = G.shape[0]
    H = np.eye(n) - np.full((n, n), 1.0 / n)
    K = H @ G @ H
    K = (K + K.T) / 2
    vals, vecs = np.linalg.eigh(K)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0.0, None), vecs[:, order]

    k = min(3, max(n - 1, 0))
    coords = np.zeros((n, 3))
    coords[:, :k] = vecs[:, :k] * np.sqrt(vals[:k])
    for c in range(k):
        if coords[anchor, c] < 0:
            coords[:, c] *= -1
    total = vals.sum()
    explained = (vals[:k] / total).tolist() if total > 0 else [0.0] * k
    return coords, explained


# ------------------------------------------------------------- update alignment


def build_similarity(G, index, servers, clients):
    """
    Per-round pairwise cosine similarity of the site update deltas
    (client_r - global_{r-1}), computed in the full weight space.

    index maps ("server", r) / ("client", site, r) to the Gram row.
    Returns (per_round {r: (sites, matrix)}, all_sites, mean_matrix).
    """
    per_round = {}
    for r in sorted(clients):
        prev = ("server", r - 1)
        sites = sorted(s for s in clients[r] if ("client", s, r) in index)
        if prev not in index or len(sites) < 2:
            continue
        g = index[prev]
        rows = [index[("client", s, r)] for s in sites]
        dots = np.array([[pair_dot(G, a, g, b, g) for b in rows] for a in rows])
        norms = np.sqrt(np.clip(np.diag(dots), 0.0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            sim = dots / np.outer(norms, norms)
        # Zero-norm updates carry no direction: leave them undefined.
        dead = norms == 0
        sim[dead, :] = np.nan
        sim[:, dead] = np.nan
        per_round[r] = (sites, np.clip(sim, -1.0, 1.0))

    all_sites = sorted({s for sites, _ in per_round.values() for s in sites})
    idx = {s: i for i, s in enumerate(all_sites)}
    n = len(all_sites)
    total, count = np.zeros((n, n)), np.zeros((n, n))
    for sites, mat in per_round.values():
        for a, sa in enumerate(sites):
            for b, sb in enumerate(sites):
                if not np.isnan(mat[a, b]):
                    total[idx[sa], idx[sb]] += mat[a, b]
                    count[idx[sa], idx[sb]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_mat = np.where(count > 0, total / np.where(count == 0, 1, count), np.nan)
    return per_round, all_sites, mean_mat


def offdiag_mean(mat: np.ndarray) -> float:
    n = mat.shape[0]
    if n < 2:
        return float("nan")
    vals = mat[~np.eye(n, dtype=bool)]
    vals = vals[~np.isnan(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


def fedavg_check(G, index, clients, train_n):
    """
    Sanity check that the saved files are what FedAvg produced: the round-r
    server model should equal the n-weighted mean of that round's client
    uploads. Reports ||server_r - sum(n_s/N * client_s)|| relative to the size
    of the round's global step ||server_r - server_{r-1}||.

    The combination's coefficients sum to zero, so the Gram offset cancels and
    this is a plain quadratic form a^T G a. The residual is float32 rounding
    (the server casts the float64 average back to float32), so it should sit
    many orders of magnitude below 1.
    """
    worst = None
    for r in sorted(clients):
        if ("server", r) not in index or ("server", r - 1) not in index:
            continue
        sites = [s for s in clients[r] if ("client", s, r) in index]
        ns = [train_n.get((r, s)) for s in sites]
        if not sites or any(n is None for n in ns):
            continue
        a = np.zeros(G.shape[0])
        a[index[("server", r)]] = 1.0
        for s, n in zip(sites, ns):
            a[index[("client", s, r)]] -= n / sum(ns)
        resid = math.sqrt(max(float(a @ G @ a), 0.0))
        g, gp = index[("server", r)], index[("server", r - 1)]
        step = math.sqrt(max(pair_dot(G, g, gp, g, gp), 0.0))
        if step == 0:
            continue
        rel = resid / step
        if worst is None or rel > worst[1]:
            worst = (r, rel)
    return worst


# --------------------------------------------------------------------- outputs


def jsonable(mat: np.ndarray) -> list:
    """NaN is not valid JSON; the page renders None as "n/a"."""
    return [[None if np.isnan(v) else round(float(v), 6) for v in row] for row in mat]


def write_similarity_csv(path, per_round, all_sites, mean_mat) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "site_a", "site_b", "cosine_similarity"])
        for r in sorted(per_round):
            sites, mat = per_round[r]
            for a, sa in enumerate(sites):
                for b, sb in enumerate(sites):
                    if b > a and not np.isnan(mat[a, b]):
                        w.writerow([r, sa, sb, f"{mat[a, b]:.6f}"])
        for a, sa in enumerate(all_sites):
            for b, sb in enumerate(all_sites):
                if b > a and not np.isnan(mean_mat[a, b]):
                    w.writerow(["mean", sa, sb, f"{mean_mat[a, b]:.6f}"])


def write_pca_csv(path, entries, coords) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "site", "round", "pc1", "pc2", "pc3"])
        for (key, _), xyz in zip(entries, coords):
            if key[0] == "server":
                w.writerow(["server", "", key[1], *(f"{v:.6f}" for v in xyz)])
            else:
                w.writerow(["client", key[1], key[2], *(f"{v:.6f}" for v in xyz)])


def print_matrix(sites, mat, title) -> None:
    width = max(8, *(len(s) + 2 for s in sites))
    print(f"\n{title}")
    print(" " * width + "".join(f"{s:>{width}}" for s in sites))
    for i, s in enumerate(sites):
        cells = "".join(f"{'n/a' if np.isnan(v) else format(v, '+.3f'):>{width}}"
                        for v in mat[i])
        print(f"{s:<{width}}{cells}")


def render_png(path, payload, title) -> None:
    """Static twin of the 3D trajectory panel, for slides."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pca = payload["pca"]
    g = pca["global"]
    by_round = {d["round"]: d["xyz"] for d in g}
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    gx, gy, gz = zip(*(d["xyz"] for d in g))
    ax.plot(gx, gy, gz, "-o", color="#0b0b0b", lw=2, ms=4, label="Global model", zorder=5)
    step = max(1, len(g) // 15)
    for i, d in enumerate(g):
        if i % step == 0 or i == len(g) - 1:
            ax.text(*d["xyz"], f"R{d['round']}", fontsize=8, color="#52514e")

    for i, cid in enumerate(payload["run"]["client_ids"]):
        color = SERIES_LIGHT[i % 8]
        name = payload["run"]["client_names"].get(cid, cid)
        labelled = False
        for r_str, cmap in pca["clients"].items():
            to = cmap.get(cid)
            frm = by_round.get(int(r_str) - 1)
            if to is None:
                continue
            ax.scatter(*to, color=color, s=18, label=None if labelled else name)
            labelled = True
            if frm is not None:
                d = np.subtract(to, frm)
                ax.quiver(*frm, *d, color=color, alpha=0.55, lw=0.8,
                          arrow_length_ratio=0.12)

    ex = pca["explained"] + [0.0] * (3 - len(pca["explained"]))
    ax.set_xlabel(f"PC 1 ({ex[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC 2 ({ex[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC 3 ({ex[2] * 100:.1f}%)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ HTML page


def plotly_source() -> tuple[str, bool]:
    """Return (script contents or src url, inlined?)."""
    try:
        from plotly.offline import get_plotlyjs
        return get_plotlyjs(), True
    except Exception:
        return "https://cdn.plot.ly/plotly-2.35.2.min.js", False


def tile(label, value, sub=None, hero=False, delta=None) -> str:
    classes = "tile hero" if hero else "tile"
    parts = [f'<div class="{classes}">', f'<div class="label">{html.escape(label)}</div>',
             f'<div class="value">{html.escape(value)}</div>']
    if delta is not None:
        direction = "up" if delta.startswith("+") else "down"
        parts.append(f'<div class="sub delta {direction}">{html.escape(delta)}</div>')
    if sub:
        parts.append(f'<div class="sub">{html.escape(sub)}</div>')
    parts.append("</div>")
    return "".join(parts)


def build_tiles(payload, mean_mat) -> str:
    names = payload["run"]["client_names"]
    val = payload["evals"]["val"]
    g = [r for r in val["global"] if r["auc"] is not None]
    tiles = []

    if g:
        first, last = g[0], g[-1]
        best = max(g, key=lambda r: r["auc"])
        tiles.append(tile(
            "Global val AUC, final round",
            f"{last['auc']:.3f}",
            sub=f"round {last['round']} · n-weighted over sites",
            hero=True,
            delta=f"{last['auc'] - first['auc']:+.3f} vs round {first['round']}",
        ))
        tiles.append(tile("Best global val AUC", f"{best['auc']:.3f}",
                          sub=f"reached in round {best['round']}"))

    last_round = max((r["round"] for s in val["clients"].values() for r in s), default=None)
    if last_round is not None:
        labels, values = [], []
        for cid in payload["run"]["client_ids"]:
            rec = next((r for r in val["clients"].get(cid, []) if r["round"] == last_round), None)
            if rec and rec["auc"] is not None:
                labels.append(names.get(cid, cid))
                values.append(f"{rec['auc']:.3f}")
        if values:
            tiles.append(tile("Val AUC by site, final round", " · ".join(values),
                              sub=" · ".join(labels) + f" (round {last_round})"))

    if mean_mat is not None and mean_mat.size:
        avg = offdiag_mean(mean_mat)
        if not np.isnan(avg):
            tiles.append(tile("Mean update alignment", f"{avg:+.3f}",
                              sub="pairwise cosine of site deltas, averaged over rounds"))

    return f'<div class="tiles">{"".join(tiles)}</div>' if tiles else ""


def build_filters(payload) -> str:
    rounds = payload["run"]["rounds"]
    if not rounds:
        return ""
    max_idx = len(rounds) - 1
    return f"""
    <div class="filters">
      <div class="field">
        <label for="sel-round">Round <strong id="round-val">{rounds[-1]}</strong></label>
        <input type="range" id="sel-round" min="0" max="{max_idx}" step="1" value="{max_idx}">
      </div>
      <div class="note">Marks the round on the charts and picks the alignment matrix.</div>
    </div>"""


def build_table(site_rows, payload) -> str:
    """Table-view twin of the metric charts (nothing is locked behind hover)."""
    if not site_rows:
        return ""
    names = payload["run"]["client_names"]
    order = {cid: i for i, cid in enumerate(payload["run"]["client_ids"])}

    def fmt(v, spec):
        return "—" if v is None else format(v, spec)

    body = []
    for row in sorted(site_rows, key=lambda r: (r["round"], r["split"] != "val",
                                                order.get(r["site"], 0))):
        swatch = SERIES_LIGHT[order.get(row["site"], 0) % 8]
        body.append(
            f'<tr><td><span class="swatch" style="background:{swatch}"></span>'
            f"{html.escape(names.get(row['site'], row['site']))}</td>"
            f"<td>{row['round']}</td><td>{row['split']}</td>"
            f"<td>{fmt(row['auc'], '.4f')}</td><td>{fmt(row['acc'], '.2%')}</td>"
            f"<td>{fmt(row['loss'], '.4f')}</td><td>{fmt(row['n'], ',')}</td></tr>"
        )
    return f"""
    <details class="table-view">
      <summary>Table view — every per-site metric row</summary>
      <div class="table-scroll">
        <table>
          <thead><tr>
            <th>Site</th><th>Round</th><th>Split</th><th>AUC</th>
            <th>Accuracy</th><th>Loss</th><th>Images</th>
          </tr></thead>
          <tbody>{"".join(body)}</tbody>
        </table>
      </div>
    </details>"""


def build_similarity_table(all_sites, names, mean_mat) -> str:
    if mean_mat is None or not mean_mat.size:
        return ""
    head = "".join(f"<th>{html.escape(names.get(s, s))}</th>" for s in all_sites)
    body = []
    for i, s in enumerate(all_sites):
        cells = "".join(f"<td>{'—' if np.isnan(v) else format(v, '+.3f')}</td>"
                        for v in mean_mat[i])
        body.append(f"<tr><td>{html.escape(names.get(s, s))}</td>{cells}</tr>")
    return f"""
    <details class="table-view" style="margin-top:14px">
      <summary>Table view — mean pairwise cosine similarity</summary>
      <div class="table-scroll">
        <table>
          <thead><tr><th></th>{head}</tr></thead>
          <tbody>{"".join(body)}</tbody>
        </table>
      </div>
    </details>"""


def metric_card(fig_id, title, sub) -> str:
    return f"""
        <div class="card">
          <h3>{title}</h3>
          <p class="sub">{sub}</p>
          <div class="plot" id="{fig_id}" style="height:320px"></div>
        </div>"""


def render_html(payload, site_rows, similarity, out_path) -> None:
    per_round, all_sites, mean_mat = similarity
    run = payload["run"]
    names = run["client_names"]

    chips = []
    if run["backbone"]:
        chips.append(f"backbone {run['backbone']}")
    chips.append(f"{len(run['client_ids'])} sites")
    chips.append(f"{len(run['rounds'])} rounds")
    if payload["pca"]["global"]:
        chips.append(f"{run['n_checkpoints']} checkpoints")
    chip_html = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)

    has_val = bool(payload["evals"]["val"]["global"] or payload["evals"]["val"]["clients"])
    has_train = bool(payload["evals"]["train"]["global"] or payload["evals"]["train"]["clients"])
    has_sim = payload["similarity"]["mean"] is not None
    has_pca = bool(payload["pca"]["global"])

    sections = []

    if has_pca:
        explained = payload["pca"]["explained"]
        share = f"{sum(explained) * 100:.1f}%" if explained else "—"
        sections.append(f"""
    <section>
      <h2>Trajectory through weight space</h2>
      <p class="lede">Every checkpoint — the global model after each round and
        each site's upload before aggregation — projected onto the same three
        principal components ({share} of the variance between them). Dotted
        lines run from the global model a site started from to where its local
        training took it; the next global point is their n-weighted average, so
        it sits closest to the site with more training images. Drag to rotate.</p>
      <div class="row">
        <div class="card">
          <h3>Global path and site pulls (PCA)</h3>
          <p class="sub">PC 1–3 explain
            {" · ".join(f"{v * 100:.1f}%" for v in explained) if explained else "—"}
            of the variance. BatchNorm running statistics are excluded.</p>
          <div class="plot" id="fig-trajectory" style="height:620px"></div>
        </div>
      </div>
    </section>""")

    if has_sim:
        h = max(300, 46 * len(all_sites) + 120)
        sections.append(f"""
    <section>
      <h2>Site update alignment</h2>
      <p class="lede">Cosine similarity between site update deltas
        (site<sub>r</sub> − global<sub>r−1</sub>), measured in the full weight
        space before any dimensionality reduction. Sites whose data agree pull
        the model the same way (positive); heterogeneous sites pull apart (near
        zero or negative). With two sites every matrix holds a single number,
        so the per-round trend is the main view.</p>
      <div class="row">
        <div class="card">
          <h3>Mean pairwise alignment per round</h3>
          <p class="sub">Off-diagonal average of the matrix below, round by round.</p>
          <div class="plot" id="fig-sim-trend" style="height:260px"></div>
        </div>
      </div>
      <div class="row sim" style="margin-top:18px">
        <div class="card">
          <h3>Averaged over all rounds</h3>
          <p class="sub">The run's overall alignment between each pair of sites.</p>
          <div class="plot" id="fig-sim-mean" style="height:{h}px"></div>
        </div>
        <div class="card">
          <h3>Selected round</h3>
          <p class="sub" id="sim-round-note">Round</p>
          <div class="plot" id="fig-sim-round" style="height:{h}px"></div>
        </div>
      </div>
      {build_similarity_table(all_sites, names, mean_mat)}
    </section>""")

    if has_val:
        sections.append(f"""
    <section>
      <h2>Global model on each site's validation set</h2>
      <p class="lede">After each round the aggregated model is sent back and
        every site scores it on its own validation images; the ink line is the
        n-weighted mean the server logs as val_*. Images never leave a site, so
        there is no shared test set: each site's line is on different data and
        the lines are not directly comparable to each other.</p>
      <div class="row three">
        {metric_card("fig-val-auc", "Validation AUC", "Global in ink; sites in their own colour.")}
        {metric_card("fig-val-acc", "Validation accuracy", "At the 0.5 threshold.")}
        {metric_card("fig-val-loss", "Validation loss", "Cross-entropy.")}
      </div>
    </section>""")

    if has_train:
        sections.append(f"""
    <section>
      <h2>Site updates on their own training data</h2>
      <p class="lede">What each site reported for its local training in the
        round — its own model on its own training images, before aggregation.
        This is the closest available stand-in for scoring each upload, but it
        is a training-set number: it measures fit, not generalisation, and a
        site pulling far ahead here while the validation panels stall is
        specialising to its own data.</p>
      <div class="row three">
        {metric_card("fig-train-auc", "Train AUC", "n-weighted mean in ink.")}
        {metric_card("fig-train-acc", "Train accuracy", "At the 0.5 threshold.")}
        {metric_card("fig-train-loss", "Train loss", "Cross-entropy.")}
      </div>
    </section>""")

    if not sections:
        sections.append('<section><p class="empty">Nothing to report: this run '
                        "directory has no round weights and no metrics CSVs.</p></section>")

    css = (ASSETS / "report.css").read_text()
    js = (ASSETS / "report.js").read_text()
    plotly, inlined = plotly_source()
    plotly_tag = (f"<script>{plotly}</script>" if inlined
                  else f'<script src="{plotly}" charset="utf-8"></script>')

    # `</script>` inside the payload would close the tag early. allow_nan=False
    # turns a stray NaN into an error here instead of a blank page.
    payload_json = json.dumps(payload, allow_nan=False).replace("</", "<\\/")

    document = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(run["name"])} — federated run report</title>
<style>{css}</style>
{plotly_tag}
</head>
<body>
<div class="wrap">
  <header class="page">
    <div>
      <h1>Federated run report — {html.escape(run["name"])}</h1>
      <div class="run-meta">{chip_html}</div>
    </div>
    <button type="button" class="theme-toggle" id="theme-toggle">Dark theme</button>
  </header>

  {build_tiles(payload, mean_mat if has_sim else None)}
  {build_filters(payload)}
  {"".join(sections)}

  {build_table(site_rows, payload)}

  <footer class="page">
    Built from <code>{html.escape(run["name"])}</code> ·
    metrics from <code>metrics_global.csv</code> and <code>metrics_per_site.csv</code> ·
    trajectory and alignment from <code>round_weights/*.pth</code>
    (also written to <code>trajectory_pca.csv</code> and <code>client_similarity.csv</code>).
  </footer>
</div>
<script>window.__FL_PAYLOAD__ = {payload_json};</script>
<script>{js}</script>
</body>
</html>
"""
    Path(out_path).write_text(document)


# ------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the weight-trajectory report for a federated run.")
    parser.add_argument("--trial", default=None,
                        help="trial tag or run directory path "
                             "(default: the most recently modified run under --results-root)")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results",
                        help="where fl_server.py writes runs (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where to write the outputs (default: the run directory)")
    parser.add_argument("--pyproject", type=Path, default=REPO_ROOT / "pyproject.toml",
                        help="read for the site display names (default: %(default)s)")
    parser.add_argument("--sim-range", choices=["full", "auto"], default="full",
                        help="colour range of the alignment matrices: 'full' fixes it at "
                             "[-1, 1] (comparable across runs), 'auto' scales to the "
                             "largest observed magnitude")
    args = parser.parse_args()

    run_dir = find_run_dir(args.results_root, args.trial)
    if run_dir is None:
        sys.exit(f"No run directory found for trial={args.trial!r} under {args.results_root}")
    out_dir = args.out_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run: {run_dir}")

    servers, clients = index_checkpoints(run_dir)
    evals, site_rows = load_metrics(run_dir)
    names = load_site_names(args.pyproject)
    n_client_files = sum(len(c) for c in clients.values())
    print(f"  checkpoints: {len(servers)} server, {n_client_files} client")
    print(f"  metrics: {len(evals['val']['global'])} round(s), {len(site_rows)} per-site row(s)")
    if not servers and not site_rows:
        sys.exit("Nothing to visualize: no round_weights/ and no metrics CSVs.")

    # Gram rows: servers in round order, then clients by round and site.
    entries = [(("server", r), servers[r]) for r in sorted(servers)]
    entries += [(("client", s, r), clients[r][s]) for r in sorted(clients) for s in sorted(clients[r])]
    index = {key: i for i, (key, _) in enumerate(entries)}

    pca_payload = {"explained": [], "global": [], "clients": {}}
    per_round, all_sites, mean_mat = {}, [], np.zeros((0, 0))
    if servers:
        G = gram_matrix([p for _, p in entries])

        coords, explained = kernel_pca(G, anchor=index[("server", max(servers))])
        pca_payload["explained"] = [round(float(v), 4) for v in explained]
        for (key, _), xyz in zip(entries, coords):
            xyz = [round(float(v), 6) for v in xyz]
            if key[0] == "server":
                pca_payload["global"].append({"round": key[1], "xyz": xyz})
            else:
                pca_payload["clients"].setdefault(str(key[2]), {})[key[1]] = xyz
        write_pca_csv(out_dir / "trajectory_pca.csv", entries, coords)
        print(f"  PCA: PC1-3 explain {' / '.join(f'{v:.1%}' for v in explained)}")

        per_round, all_sites, mean_mat = build_similarity(G, index, servers, clients)
        if per_round:
            for r in sorted(per_round):
                sites, mat = per_round[r]
                print_matrix(sites, mat, f"Round {r}  (mean pairwise = {offdiag_mean(mat):+.4f})")
            print_matrix(all_sites, mean_mat,
                         f"Mean over rounds  (mean pairwise = {offdiag_mean(mean_mat):+.4f})")
            write_similarity_csv(out_dir / "client_similarity.csv", per_round, all_sites, mean_mat)
        else:
            print("\n  Not enough sites per round (need >= 2) for pairwise alignment.")

        train_n = {(r["round"], r["site"]): r["n"] for r in site_rows if r["split"] == "train"}
        worst = fedavg_check(G, index, clients, train_n)
        if worst is None:
            print("\n  FedAvg check skipped (no round with train n for every site).")
        else:
            r, rel = worst
            verdict = "OK" if rel < 1e-2 else "MISMATCH — server file is not the n-weighted client mean"
            print(f"\n  FedAvg check: worst residual {rel:.2e} of the global step "
                  f"(round {r}) — {verdict}")

    client_ids = sorted({s for c in clients.values() for s in c}
                        | set(evals["val"]["clients"]) | set(evals["train"]["clients"]))
    rounds = sorted(set(servers) - {0}
                    | {r["round"] for r in evals["val"]["global"]}
                    | {r["round"] for s in evals["val"]["clients"].values() for r in s})

    if args.sim_range == "auto" and per_round:
        finite = np.concatenate([m[~np.isnan(m)].ravel() for _, m in per_round.values()])
        bound = float(np.max(np.abs(finite))) if finite.size else 1.0
        srange = [-bound, bound]
    else:
        srange = [-1.0, 1.0]

    payload = {
        "run": {
            "name": run_dir.resolve().name,
            "backbone": read_backbone(run_dir),
            "rounds": rounds,
            "client_ids": client_ids,
            "client_names": {cid: names.get(cid, cid) for cid in client_ids},
            "n_checkpoints": len(entries),
        },
        "evals": evals,
        "similarity": {
            "mean": ({"ids": all_sites, "matrix": jsonable(mean_mat)} if per_round else None),
            "per_round": {str(r): {"ids": ids, "matrix": jsonable(mat)}
                          for r, (ids, mat) in per_round.items()},
            "trend": [{"round": r, "avg": round(offdiag_mean(per_round[r][1]), 6)}
                      for r in sorted(per_round)
                      if not np.isnan(offdiag_mean(per_round[r][1]))],
            "range": srange,
        },
        "pca": pca_payload,
    }

    name = payload["run"]["name"]
    if pca_payload["global"]:
        png = out_dir / "trajectory_3d.png"
        render_png(png, payload, f"{name}: weight trajectory (PCA)")
        print(f"  PNG    -> {png}")
    report = out_dir / f"{name}_report.html"
    render_html(payload, site_rows, (per_round, all_sites, mean_mat), report)
    print(f"  Report -> {report}")
    if not plotly_source()[1]:
        print("  (plotly not installed: the report loads plotly.js from the CDN, so "
              "it needs internet to render. pip install plotly to inline it.)")


if __name__ == "__main__":
    main()
