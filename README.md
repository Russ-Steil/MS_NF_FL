# MS_NF_FL — Federated OCT MS Classifier

Cross-institution federated learning for **Multiple Sclerosis (MS) vs. control** classification
from OCT B-scans. Each participating site trains locally on its own data and shares only model
weights and scalar metrics with a central server — raw images, per-sample labels, and probabilities
never leave the site.

Two backbones are available, selected by `backbone` in the run config
(see [Choosing a backbone](#choosing-a-backbone)):

| `backbone` | Model | Input | Optimizer | Per-round transfer |
|---|---|---|---|---|
| `resnet` (default) | ResNet50, ImageNet weights | 512x1024 | SGD + momentum | ~95 MB |
| `retfound` | RETFound ViT-L/16, MAE-pretrained on retinal images | 224x448 | AdamW + layer-wise LR decay | ~1.21 GB |

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

The backbone is chosen **once, on the server**. Clients do not build a model until
they have joined: the `POST /join` response carries the backbone, the input
resolution, the wire version and the server's reference state-dict manifest, and
the client builds to match and refuses to continue if its own layout differs.
This matters because FedAvg pairs tensors positionally by name — two sites on
different `timm` versions would otherwise disagree silently, which is why `timm`
is pinned in `requirements.txt`.

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
| `model.py` | `build_model()` — the single factory both backbones go through, plus the RETFound checkpoint remap/pos-embed interpolation |
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
heartbeat-timeout-s = 300

[tool.fl.sites.ucd]              # one section per site; token is the client's bearer token
display-name = "University of Denver"
token = "replace-me-ucd"         # <-- change these before running

[tool.fl.sites.unipd]
display-name = "University of Padova"
token = "replace-me-unipd"

[tool.fl.config]                 # training hyperparameters
num-server-rounds = 10
min-nodes = 2
lr = 0.0005                      # resnet only (SGD)
weight-decay = 0.005             # resnet only (SGD)
local-epochs = 2
batch-size = 16
trial-tag = "lr_5e4"

backbone = "resnet"              # "resnet" or "retfound"
retfound-weights = ""            # required when backbone = "retfound"
retfound-finetune = true         # false freezes the encoder (fc_norm + head only)
retfound-lr = 0.001              # AdamW
retfound-weight-decay = 0.05
retfound-layer-decay = 0.75
```

**Set the site tokens before running.** Any `[tool.fl.config]` value can be overridden on the
command line via `--run-config` (same `key=value` syntax the Flower version used).

### Choosing a backbone

```bash
# ResNet50 (default)
python run_server.py --run-config "backbone=resnet trial-tag='my_trial' lr=0.001"

# RETFound ViT-L
python run_server.py --run-config "backbone=retfound \
  retfound-weights='/path/to/retfound_oct.safetensors' \
  retfound-lr=0.001 trial-tag='my_retfound_trial'"
```

`retfound-weights` is **required** for a RETFound run. Clients are overwritten with
the server's global weights before their first gradient step, so the server's
checkpoint is the only one that seeds the federation — without it every site would
train a randomly initialised ViT-L and nothing would say so. The server logs an
encoder fingerprint at startup so a random init is visible in `log.log`.

The RETFound checkpoint ships in HuggingFace `ViTModel` key naming with a square
14x14 positional embedding; `model.py` renames the keys to timm's layout and
bicubically interpolates the grid to 14x28 for the 224x448 input. Loading raises
rather than warns if fewer than 24 transformer blocks match.

`retfound-lr` and `retfound-weight-decay` are deliberately separate from `lr` and
`weight-decay`: those are SGD-scale numbers, and layer decay divides the early
layers by `0.75^25` on top of whatever it is given. Reusing `lr = 5e-4` would leave
`patch_embed` at an effective `4e-7`.

Each site may pass its own `--weights-path` (UniPD's copy need not be where the
server's is). It does not seed training — the client fingerprints it and compares
against the server's, so a site holding a different checkpoint fails at startup
instead of quietly contributing mismatched updates.

**Operational notes for a RETFound run:**

- ~3.6 GB moves per client per round. Measure the link before committing to 60
  rounds — at 10 Mbit/s that is ~48 min/round and will exceed `round-timeout-s`.
- Clients need **≥24 GB of host RAM** (not GPU): each weight exchange transiently
  holds ~3.6 GB. A SLURM job submitted with `--mem=8G` OOMs mid-round and the
  server reports it as a heartbeat drop.
- `batch-size` is one global value, so the smallest GPU across the sites sets it.
  `bs=16` needs ~15 GB of VRAM at 224x448. A site with a smaller GPU can call
  `set_grad_checkpointing(True)` locally — it does not change the state dict, so
  it cannot break FedAvg.
- Frozen mode (`retfound-finetune=false`) still ships the full 1.21 GB state dict
  each round to update a few thousand trainable weights.

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

Nothing about the architecture is passed here — the client takes it from the
server's `/join` response. The optional `--weights-path` points at this site's own
RETFound checkpoint and is used to verify it matches the server's, not to
initialise training.

## Outputs

Per trial, under `RESULTS_ROOT/<trial-tag>/`:

- `model_federated_final.pth` — latest aggregated global weights
- `model_best_val_auc.pth` — best checkpoint by weighted validation AUC
- `metrics_global.csv`, `metrics_per_site.csv` — per-round metrics
- `loss_curves.png`, `auc_curves.png` — training curves
- `final_metrics.json` — full history, best round, and (on failure) the abort reason
- `log.log`, `failure.log`

TensorBoard scalars (global, per-site, and overlay comparisons) are written under `TB_ROOT/<trial-tag>/`.
