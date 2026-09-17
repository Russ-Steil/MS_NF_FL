# MS_NF_FL — Federated OCT MS Classifier

Cross-institution federated learning for **Multiple Sclerosis (MS) vs. control** classification
from OCT B-scans, using a ResNet50 binary classifier. Each participating site trains locally on
its own data and shares only model weights and scalar metrics with a central server — raw images,
per-sample labels, and probabilities never leave the site.

The federation is implemented over **plain HTTP with the Python standard library** — there is no
Flower runtime, no gRPC, and nothing needs to listen on an inbound port at the client sites. This
was a deliberate rewrite of an earlier Flower-based version so that clients can run unmodified on a
SLURM compute node that can only make *outbound* connections.

## How it works

```
                 ┌─────────────────────────────┐
                 │      fl_server.py           │
                 │  single HTTP port (:9092)   │
                 │  FedAvg + per-site metrics  │
                 │  TensorBoard + checkpoints  │
                 └──────────────┬──────────────┘
             pull weights  ▲    │  push weights + scalars
                           │    ▼
         ┌─────────────────┴┐  ┌┴──────────────────┐
         │  fl_client.py    │  │  fl_client.py     │
         │  site = ucd      │  │  site = unipd     │
         │  (Univ. Denver)  │  │  (Univ. Padova)   │
         └──────────────────┘  └───────────────────┘
```

On joining, each client reports its per-class training image counts in the
`POST /join` metadata. Once every site is connected the server picks the
**per-class training cap**: the smallest minority-class count across sites. Every
client then draws exactly that many images *per class* on each epoch, so all
sites train on the same number of images and the same 50/50 class balance, and
no site dominates the `num_examples`-weighted FedAvg.

The subset is **redrawn every epoch**, so a large site still works through all of
its data over the course of the run — nothing is permanently discarded. The site
that sets the cap trains on everything it would have anyway. Validation is *not*
capped: every site always evaluates on its complete validation set.

If a client joins without reporting counts, the server logs a warning and leaves
the run uncapped; each site then falls back to balancing against its own minority
class.

Each federated round:

1. Clients poll `GET /status`, see the `fit` phase, and pull the current global weights
   from `GET /params`.
2. Each client trains locally for `local-epochs`, then pushes updated weights plus scalar
   train metrics to `POST /fit`.
3. The server aggregates the weights with weighted **FedAvg** (weighted by each site's
   `num_examples`) and moves to the `evaluate` phase.
4. Clients pull the new global weights, evaluate on their local validation set, and push scalar
   validation metrics to `POST /evaluate`.
5. The server aggregates validation scalars, logs to TensorBoard/CSV, and checkpoints the model.

The server aggregates loss/accuracy/AUC into weighted global scalars **and** keeps them broken out
per site. **All AUCs are computed locally by each client on its own validation set** — there is no
pooled/global ROC, because probabilities are never shared.

### Privacy & security notes

- The server never receives per-sample labels or probabilities — only scalar summaries
  (`loss`, `accuracy`, `auc`, `n`) per site.
- Weights are exchanged as a validated `.npz` wire format (see `fl_common.py`). Payloads are loaded
  with `allow_pickle=False` and checked against the receiver's own state-dict manifest (names,
  order, shapes), so a payload can never introduce an unexpected key or executable content.
- Every endpoint requires an `Authorization: Bearer <site token>` header. Tokens are configured in
  `pyproject.toml` under `[tool.fl.sites.<id>]`.

### Failure policy

The run is fail-fast. The first fit/evaluate failure, any client that drops for longer than the
heartbeat timeout, or any round returning fewer results than `min-nodes` aborts the whole run. The
reason names the site and is written to `failure.log` and `final_metrics.json` before the process
exits non-zero. A background heartbeat from each client keeps the server's connection state warm
during long local epochs.

## Files

| File | Role |
|---|---|
| `fl_server.py` | Federated server: HTTP endpoints, FedAvg aggregation, orchestration, TensorBoard/CSV logging, checkpointing |
| `fl_client.py` | Federated client: pulls weights, trains/evaluates locally, pushes results; background heartbeat |
| `fl_common.py` | Shared wire format (`.npz` serialize/deserialize with manifest validation), FedAvg, console helpers. **numpy + stdlib only, no torch** |
| `model.py` | `get_resnet50_binary()` — ResNet50 with dropout + 2-class FC head |
| `train.py` | Local `train` / `evaluate` loops, `safe_auc`, metric/stat helpers, plotting (loss/AUC/ROC/confusion) |
| `my_datasets.py` | `UniversalOCTDataset` (on-the-fly crop/resize) and `CachedOCTDataset` (reads pre-cached `.pt` tensors); train/val transforms |
| `cache_images.py` | One-time pre-caching of OCT images to uint8 tensors (bakes in crop + resize) to remove PIL decode from the training loop |
| `downsampler.py` | `_BalancedDownsampler` — draws an equal, capped number of training images per class each epoch (see below) |
| `run_server.py` | Thin entry point that calls `fl_server.main()` |
| `*.sh` | Convenience launch scripts (see below) |

### Sites and data layout

Two sites are configured out of the box: `ucd` (University of Denver) and `unipd` (University of
Padova). The UniPD images get a 500px left crop (`CROP_LEFT` in `cache_images.py` / a 400px crop in
the on-the-fly transform) to remove a side panel; UCD images are used as-is.

Data is expected in an ImageFolder layout, one class subdirectory per label:

```
<data-path>/train/<class>/*.png
<data-path>/val/<class>/*.png
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ (uses the stdlib `tomllib` on 3.11+, falling back to `tomli` below that — see
`requirements.txt`).

## Configuration

All run configuration lives in `pyproject.toml`:

```toml
[tool.fl.server]                 # server host/port and timeouts
host = "0.0.0.0"
port = 9092
round-timeout-s = 7200
heartbeat-timeout-s = 60

[tool.fl.sites.ucd]              # one section per site; token is the client's bearer token
display-name = "University of Denver"
token = "replace-me-ucd"         # <-- change these before running

[tool.fl.sites.unipd]
display-name = "University of Padova"
token = "replace-me-unipd"

[tool.fl.config]                 # training hyperparameters
num-server-rounds = 10
min-nodes = 2
lr = 0.0005
weight-decay = 0.005
local-epochs = 2
batch-size = 16
trial-tag = "lr_5e4"
```

**Set the site tokens before running.** Any `[tool.fl.config]` value can be overridden on the
command line via `--run-config` (same `key=value` syntax the Flower version used).

Server output paths are set at the top of `fl_server.py` (`RESULTS_ROOT`, `TB_ROOT`).

## Usage

### 1. (Optional) Pre-cache images at each site

Removes PIL decode + resize from the training loop. Run once per site. The cache records a
`manifest.json`; the client refuses to train against a cache whose crop/image-size disagrees with
the run.

```bash
python cache_images.py \
  --site ucd \
  --data-path /path/to/data \
  --cache-path /path/to/cache/ucd_512 \
  --image-size 512 \
  --workers 16
```

See `cache_images_ucd.sh` / `cache_images_unipd.sh` for examples.

### 2. Start the server

```bash
python run_server.py --run-config "trial-tag='my_trial' lr=0.001 batch-size=4"
```

The server binds the port, waits for `min-nodes` sites to connect, then drives the rounds.
See `start_server.sh` / `run_trial.sh` for examples.

### 3. Start each client

```bash
FL_TOKEN=replace-me-ucd python fl_client.py \
  --site ucd \
  --data-path /path/to/data \
  --cache-path /path/to/cache/ucd_512 \
  --server http://<server-host>:9092 \
  --trial-tag my_trial \
  --gpu-index 0
```

`--site` must match a `[tool.fl.sites.<id>]` in the server's `pyproject.toml`, and `--token`
(or `$FL_TOKEN`) must match that site's token. Omit `--cache-path` to decode images on the fly from
`--data-path` instead. See `run_ucd_client.sh` / `run_unipd_client.sh` for examples.

## Outputs

Per trial, under `RESULTS_ROOT/<trial-tag>/`:

- `model_federated_final.pth` — latest aggregated global weights
- `model_best_val_auc.pth` — best checkpoint by weighted validation AUC
- `metrics_global.csv`, `metrics_per_site.csv` — per-round metrics
- `loss_curves.png`, `auc_curves.png` — training curves
- `final_metrics.json` — full history, best round, and (on failure) the abort reason
- `log.log`, `failure.log`

TensorBoard scalars (global, per-site, and overlay comparisons) are written under `TB_ROOT/<trial-tag>/`.
