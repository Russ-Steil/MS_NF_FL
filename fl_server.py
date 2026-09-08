#!/usr/bin/env python3
"""
Federated server for the ResNet50 MS classifier. No flwr.

Architecture: a single HTTP port. Clients pull the global weights, train
locally, and push weights plus scalar metrics back. Nothing inbound is needed
at the client sites, which is what makes a SLURM compute node workable.

Design note (unchanged from the flwr version): the server never receives
per-sample labels or probabilities. Each client reports only scalar summaries
of its own data (loss, accuracy, AUC, n). The server aggregates those into
weighted global scalars and additionally keeps them broken out per site. All
AUCs are computed locally by each client on its own validation set — there is
no pooled/global ROC.

Failure policy: the first fit or evaluate failure, any client that drops for
longer than the heartbeat timeout, and any round that returns fewer results
than min-nodes, aborts the whole run. The reason names the site and is written
to failure.log and final_metrics.json before the process exits non-zero.

Endpoints (all require Authorization: Bearer <site token>):
    POST /join       register, returns run identity
    GET  /status     phase, round, config for this round, submitted flag
    GET  /params     current global weights as npz
    POST /fit        npz body + X-FL-Meta {round, num_examples, metrics}
    POST /evaluate   X-FL-Meta {round, loss, num_examples, metrics}
    POST /abort      X-FL-Meta {phase, round, message}
"""
import argparse
import csv
import json
import logging
import os
import platform
import shlex
import sys
import threading
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from torch.utils.tensorboard import SummaryWriter

try:
    import tomllib
except ModuleNotFoundError:  # python < 3.11
    import tomli as tomllib

from fl_common import (
    BOLD, CYAN, DIM, GREEN, MAX_BODY_BYTES, META_HEADER, RED, RESET, YELLOW,
    deserialize_params, fedavg, manifest_of, now_hms, now_iso, paint,
    serialize_params,
)
from model import get_resnet50_binary
from train import plot_auc_curves, plot_loss_curves

# ---- config ----
RESULTS_ROOT = Path("/data/giacomo/Hereditary_MS/noflower_FL/results")
TB_ROOT = Path("/data/giacomo/Hereditary_MS/noflower_FL/runs/fl_server")
POS_LABEL_NAME = "MS"
NEG_LABEL_NAME = "CTRL"
DEFAULT_ROUND_TIMEOUT_S = 7200.0
DEFAULT_HEARTBEAT_TIMEOUT_S = 60.0
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9092
# ----------------

_LOGGER = logging.getLogger("ms_server")


class RunAborted(RuntimeError):
    """Raised to terminate the run after a client-side or connectivity failure."""


def log(msg: str) -> None:
    _LOGGER.info(f"[server] {msg}")


def event(msg: str, color: str) -> None:
    """Colored to the terminal, plain to log.log."""
    print(paint(msg, color), flush=True)
    _LOGGER.info(msg)


def setup_logging(output_dir: Path) -> None:
    path = Path(output_dir) / "log.log"

    _LOGGER.setLevel(logging.INFO)
    _LOGGER.handlers.clear()
    _LOGGER.propagate = False

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(path, mode="a")
    fh.setFormatter(fmt)
    _LOGGER.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    _LOGGER.addHandler(sh)


def weighted_scalar(pairs, key):
    """pairs is [(num_examples, metrics_dict), ...]."""
    num = sum(n * float(m[key]) for n, m in pairs if key in m and m[key] == m[key])
    den = sum(n for n, m in pairs if key in m and m[key] == m[key])
    return num / den if den else float("nan")


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if x == x else "nan"


# ---------------------------------------------------------------- config

def parse_run_overrides(text):
    """Same string syntax flwr used: "lr=0.001 batch-size=8 trial-tag='x'"."""
    out = {}
    if not text:
        return out
    for tok in shlex.split(text):
        if "=" not in tok:
            raise ValueError(f"malformed --run-config entry: {tok!r}")
        k, v = tok.split("=", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


def load_config(pyproject_path, overrides_text):
    with open(pyproject_path, "rb") as f:
        doc = tomllib.load(f)

    try:
        rc = dict(doc["tool"]["fl"]["config"])
    except KeyError:
        raise SystemExit(
            f"{pyproject_path}: missing [tool.fl.config] section"
        )

    rc.update(parse_run_overrides(overrides_text))

    fl = doc.get("tool", {}).get("fl", {})
    server_cfg = dict(fl.get("server", {}))
    sites_cfg = dict(fl.get("sites", {}))

    if not sites_cfg:
        raise SystemExit(
            f"{pyproject_path}: missing [tool.fl.sites.<id>] sections. Each site "
            f"needs display-name and token."
        )
    for sid, entry in sites_cfg.items():
        for field in ("display-name", "token"):
            if field not in entry:
                raise SystemExit(
                    f"{pyproject_path}: [tool.fl.sites.{sid}] is missing '{field}'"
                )

    return rc, server_cfg, sites_cfg


# ---------------------------------------------------------------- registry

class SiteRegistry:
    """Connection state and liveness for the named sites."""

    def __init__(self, sites_cfg, heartbeat_timeout):
        self.heartbeat_timeout = float(heartbeat_timeout)
        self.lock = threading.RLock()
        self.display = {}
        self.token_to_site = {}
        self.state = {}
        for sid, entry in sites_cfg.items():
            self.display[sid] = str(entry["display-name"])
            self.token_to_site[str(entry["token"])] = sid
            self.state[sid] = {
                "connected": False,
                "last_seen": 0.0,
                "joined_at": None,
            }

    def site_ids(self):
        return list(self.display.keys())

    def name(self, site_id):
        return self.display.get(site_id, site_id)

    def authenticate(self, header_value):
        if not header_value:
            return None
        parts = header_value.split()
        token = parts[-1] if parts else ""
        return self.token_to_site.get(token)

    def touch(self, site_id):
        """Record liveness. Announces a (re)connection in green."""
        with self.lock:
            st = self.state[site_id]
            st["last_seen"] = time.time()
            if not st["connected"]:
                st["connected"] = True
                st["joined_at"] = now_hms()
                event(f"{self.name(site_id)} is connected as of {st['joined_at']}", GREEN)

    def mark_disconnected(self, site_id, reason):
        with self.lock:
            st = self.state[site_id]
            if not st["connected"]:
                return False
            st["connected"] = False
            event(f"{self.name(site_id)} disconnected as of {now_hms()}", RED)
            print(paint(f"    reason: {reason}", DIM), flush=True)
            _LOGGER.info(f"    reason: {reason}")
            return True

    def connected(self):
        with self.lock:
            return {s for s, st in self.state.items() if st["connected"]}

    def sweep(self):
        """Returns [(site_id, reason), ...] for sites that just went silent."""
        dropped = []
        now = time.time()
        with self.lock:
            for sid, st in self.state.items():
                if not st["connected"]:
                    continue
                silence = now - st["last_seen"]
                if silence > self.heartbeat_timeout:
                    reason = (
                        f"no heartbeat for {silence:.0f}s "
                        f"(timeout {self.heartbeat_timeout:.0f}s)"
                    )
                    self.mark_disconnected(sid, reason)
                    dropped.append((sid, reason))
        return dropped


# ---------------------------------------------------------------- run state

class Run:
    """Shared state between the HTTP handlers and the orchestrator thread."""

    def __init__(self, registry, reference_params, num_rounds):
        self.registry = registry
        self.reference = reference_params
        self.manifest = manifest_of(reference_params)
        self.num_rounds = int(num_rounds)

        self.cv = threading.Condition()
        self.phase = "waiting"          # waiting | fit | evaluate | done | aborted
        self.round = 0
        self.config = {}
        self.params_blob = serialize_params(reference_params)
        self.global_params = reference_params

        self.fit_results = {}           # site -> (n, params, metrics)
        self.eval_results = {}          # site -> (loss, n, metrics)
        self.participants = set()
        self.abort_reasons = None

    # -- mutation helpers, all under the condition --

    def set_params(self, params):
        with self.cv:
            self.global_params = params
            self.params_blob = serialize_params(params)

    def begin_phase(self, phase, server_round, config):
        with self.cv:
            self.phase = phase
            self.round = int(server_round)
            self.config = dict(config)
            if phase == "fit":
                self.fit_results = {}
            elif phase == "evaluate":
                self.eval_results = {}
            self.cv.notify_all()

    def finish(self, phase):
        with self.cv:
            self.phase = phase
            self.cv.notify_all()

    def abort(self, reasons):
        with self.cv:
            if self.abort_reasons is None:
                self.abort_reasons = list(reasons)
                self.phase = "aborted"
            self.cv.notify_all()

    def submitted(self, site_id):
        with self.cv:
            if self.phase == "fit":
                return site_id in self.fit_results
            if self.phase == "evaluate":
                return site_id in self.eval_results
            return False

    def snapshot(self, site_id):
        with self.cv:
            return {
                "phase": self.phase,
                "round": self.round,
                "num_rounds": self.num_rounds,
                "config": dict(self.config),
                "submitted": self.submitted(site_id),
            }


# ---------------------------------------------------------------- http

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    RUN = None  # set in main()

    # quiet the default per-request stderr spam
    def log_message(self, fmt_str, *args):
        _LOGGER.debug("[http] " + (fmt_str % args))

    # ---- helpers ----

    def _auth(self):
        run = self.RUN
        site = run.registry.authenticate(self.headers.get("Authorization"))
        if site is None:
            self._json(401, {"error": "unknown or missing site token"})
            return None
        run.registry.touch(site)
        return site

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise ValueError(f"body of {length} bytes exceeds cap {MAX_BODY_BYTES}")
        remaining = length
        chunks = []
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _meta(self):
        raw = self.headers.get(META_HEADER)
        if not raw:
            raise ValueError(f"missing {META_HEADER} header")
        return json.loads(raw)

    def _json(self, code, payload):
        blob = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _binary(self, code, blob):
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    # ---- routes ----

    def do_GET(self):
        run = self.RUN
        path = urlparse(self.path).path
        site = self._auth()
        if site is None:
            return

        if path == "/status":
            self._json(200, run.snapshot(site))
        elif path == "/params":
            with run.cv:
                blob = run.params_blob
            self._binary(200, blob)
        else:
            self._json(404, {"error": f"no such endpoint: {path}"})

    def do_POST(self):
        run = self.RUN
        path = urlparse(self.path).path
        site = self._auth()
        if site is None:
            return

        try:
            if path == "/join":
                self._json(200, {
                    "site_id": site,
                    "display_name": run.registry.name(site),
                    "num_rounds": run.num_rounds,
                })

            elif path == "/fit":
                meta = self._meta()
                blob = self._body()
                params = deserialize_params(blob, expect=run.manifest)
                with run.cv:
                    if int(meta["round"]) != run.round or run.phase != "fit":
                        self._json(409, {"error": "stale round or phase"})
                        return
                    run.fit_results[site] = (
                        int(meta["num_examples"]), params, dict(meta.get("metrics", {}))
                    )
                    run.cv.notify_all()
                self._json(200, {"ok": True})

            elif path == "/evaluate":
                meta = self._meta()
                self._body()
                with run.cv:
                    if int(meta["round"]) != run.round or run.phase != "evaluate":
                        self._json(409, {"error": "stale round or phase"})
                        return
                    run.eval_results[site] = (
                        float(meta["loss"]), int(meta["num_examples"]),
                        dict(meta.get("metrics", {})),
                    )
                    run.cv.notify_all()
                self._json(200, {"ok": True})

            elif path == "/abort":
                meta = self._meta()
                self._body()
                name = run.registry.name(site)
                reason = (
                    f"{name} reported a failure in {meta.get('phase')} at round "
                    f"{meta.get('round')}: {meta.get('message')}"
                )
                run.registry.mark_disconnected(site, "client reported a fatal error")
                run.abort([reason])
                self._json(200, {"ok": True})

            else:
                self._json(404, {"error": f"no such endpoint: {path}"})

        except Exception as exc:
            name = run.registry.name(site)
            detail = f"{name} sent a malformed request to {path}: {type(exc).__name__}: {exc}"
            log(detail)
            self._json(400, {"error": detail})


# ---------------------------------------------------------------- aggregator

class Aggregator:
    """FedAvg plus timestamps, per-site metric tracking, TensorBoard, checkpointing."""

    def __init__(self, output_dir, tb_dir, run_config, expected_nodes, registry):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tb_dir = Path(tb_dir)
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

        self.global_csv = self.output_dir / "metrics_global.csv"
        self.site_csv = self.output_dir / "metrics_per_site.csv"
        self.failure_log = self.output_dir / "failure.log"

        self.registry = registry
        self.expected_nodes = int(expected_nodes)
        self.best_val_auc = -1.0
        self.best_round = -1
        self.t_start = time.time()
        self.round_t0 = {}
        self.fit_t0 = {}

        self.history = {
            "round": [], "timestamp": [], "elapsed_s": [], "round_duration_s": [],
            "train_loss": [], "train_auc": [], "train_acc": [],
            "val_loss": [], "val_auc": [], "val_acc": [],
        }
        self.per_site_history = {}
        self.pending_train = {"global": {}, "sites": {}, "duration_s": float("nan")}
        self._last_state_dict = None

        self._init_csvs()
        self._log_hparams(run_config)

    # ---------- setup ----------

    def _init_csvs(self):
        with open(self.global_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "round", "timestamp", "elapsed_s", "round_duration_s",
                "train_loss", "val_loss", "train_auc", "val_auc",
                "train_acc", "val_acc",
            ])
        with open(self.site_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "round", "timestamp", "elapsed_s", "site", "split",
                "n", "loss", "accuracy", "auc",
            ])

    def _log_hparams(self, rc):
        text = "\n".join(f"- **{k}**: {v}" for k, v in rc.items())
        self.writer.add_text("run/config", text, 0)
        self.writer.add_text("run/started_at", now_iso(), 0)

    def _append_global_csv(self, row):
        with open(self.global_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _append_site_csv(self, rows):
        with open(self.site_csv, "a", newline="") as f:
            csv.writer(f).writerows(rows)

    def _site_slot(self, site):
        if site not in self.per_site_history:
            self.per_site_history[site] = {
                "train": {"round": [], "n": [], "loss": [], "acc": [], "auc": []},
                "val": {"round": [], "n": [], "loss": [], "acc": [], "auc": []},
            }
        return self.per_site_history[site]

    # ---------- failure handling ----------

    def abort(self, server_round, phase, reasons):
        ts = now_iso()
        elapsed = time.time() - self.t_start
        body = "\n".join(f"  [{i}] {r}" for i, r in enumerate(reasons))
        header = (f"RUN ABORTED at {ts} — round {server_round} {phase} "
                  f"(elapsed {elapsed/60:.1f} min)")
        detail = f"{header}\n{body}\n"

        event(f"round {server_round} {phase} FAILED — aborting run", RED)
        for r in reasons:
            print(paint(f"    reason: {r}", RED), flush=True)
            _LOGGER.info(f"    reason: {r}")

        with open(self.failure_log, "w") as f:
            f.write(detail)

        self.writer.add_text("run/failure", detail.replace("\n", "  \n"), server_round)
        self.writer.flush()

        self.write_outputs(server_round, status="failed",
                           failed_round=server_round, failed_phase=phase,
                           failure_reasons=reasons)
        self.writer.close()
        raise RunAborted(f"{header}\n{body}")

    # ---------- fit ----------

    def start_round(self, server_round):
        self.round_t0[server_round] = time.time()
        self.fit_t0[server_round] = time.time()
        log(f"round {server_round} — starting fit")

    def start_evaluate(self, server_round):
        log(f"round {server_round} — starting evaluate")

    def aggregate_fit(self, server_round, results, reference):
        """results: [(site_id, n, params, metrics), ...]. Returns new global params."""
        agg = fedavg(results, reference)
        fit_duration = time.time() - self.fit_t0.get(server_round, time.time())

        pairs, site_rows = [], {}
        for site, n, _params, m in results:
            pairs.append((n, m))
            site_rows[site] = {
                "n": n,
                "loss": float(m.get("loss", float("nan"))),
                "acc": float(m.get("accuracy", float("nan"))),
                "auc": float(m.get("auc", float("nan"))),
            }

        g = {
            "loss": weighted_scalar(pairs, "loss"),
            "acc": weighted_scalar(pairs, "accuracy"),
            "auc": weighted_scalar(pairs, "auc"),
        }
        self.pending_train = {"global": g, "sites": site_rows, "duration_s": fit_duration}

        log(f"round {server_round} TRAIN  loss={fmt(g['loss'])} acc={fmt(g['acc'])} "
            f"auc={fmt(g['auc'])}  ({fit_duration:.1f}s)")
        for site, r in site_rows.items():
            log(f"    [{site:<8}] n={r['n']:<6} loss={fmt(r['loss'])} "
                f"acc={fmt(r['acc'])} auc={fmt(r['auc'])}")

        state_dict = OrderedDict(
            (k, torch.as_tensor(np.array(v))) for k, v in agg.items()
        )
        model = get_resnet50_binary(pretrained=False)
        model.load_state_dict(state_dict, strict=True)
        torch.save(model.state_dict(), self.output_dir / "model_federated_final.pth")
        self._last_state_dict = model.state_dict()

        return agg

    # ---------- evaluate ----------

    def aggregate_evaluate(self, server_round, results):
        """results: [(site_id, loss, n, metrics), ...]."""
        ts = now_iso()
        elapsed = time.time() - self.t_start
        round_duration = time.time() - self.round_t0.get(server_round, time.time())

        pairs, val_sites = [], {}
        loss_num, loss_den = 0.0, 0
        for site, loss, n, m in results:
            pairs.append((n, m))
            loss_num += n * float(loss)
            loss_den += n
            val_sites[site] = {
                "n": n,
                "loss": float(m.get("loss", loss)),
                "acc": float(m.get("accuracy", float("nan"))),
                "auc": float(m.get("auc", float("nan"))),
            }

        if loss_den == 0:
            self.abort(server_round, "evaluate",
                       ["evaluate results carried no examples"])

        val_loss = loss_num / loss_den
        val_acc = weighted_scalar(pairs, "accuracy")
        val_auc = weighted_scalar(pairs, "auc")

        train_sites = self.pending_train.get("sites", {})
        gt = self.pending_train.get("global", {})

        h = self.history
        h["round"].append(server_round)
        h["timestamp"].append(ts)
        h["elapsed_s"].append(round(elapsed, 2))
        h["round_duration_s"].append(round(round_duration, 2))
        h["val_loss"].append(val_loss)
        h["val_auc"].append(val_auc)
        h["val_acc"].append(val_acc)
        h["train_loss"].append(gt.get("loss", float("nan")))
        h["train_auc"].append(gt.get("auc", float("nan")))
        h["train_acc"].append(gt.get("acc", float("nan")))

        for split, rows in (("train", train_sites), ("val", val_sites)):
            for site, r in rows.items():
                slot = self._site_slot(site)[split]
                slot["round"].append(server_round)
                slot["n"].append(r["n"])
                slot["loss"].append(r["loss"])
                slot["acc"].append(r["acc"])
                slot["auc"].append(r["auc"])

        self._append_global_csv([
            server_round, ts, f"{elapsed:.2f}", f"{round_duration:.2f}",
            f"{h['train_loss'][-1]:.6f}", f"{h['val_loss'][-1]:.6f}",
            f"{h['train_auc'][-1]:.6f}", f"{h['val_auc'][-1]:.6f}",
            f"{h['train_acc'][-1]:.6f}", f"{h['val_acc'][-1]:.6f}",
        ])
        site_rows_csv = []
        for split, rows in (("train", train_sites), ("val", val_sites)):
            for site, r in rows.items():
                site_rows_csv.append([
                    server_round, ts, f"{elapsed:.2f}", site, split,
                    r["n"], f"{r['loss']:.6f}", f"{r['acc']:.6f}", f"{r['auc']:.6f}",
                ])
        self._append_site_csv(site_rows_csv)

        self._write_tensorboard(server_round, gt, val_loss, val_acc, val_auc,
                                train_sites, val_sites, round_duration)

        log(f"round {server_round} VAL    loss={fmt(val_loss)} acc={fmt(val_acc)} "
            f"auc={fmt(val_auc)}  (round took {round_duration:.1f}s, "
            f"elapsed {elapsed/60:.1f} min)")
        for site, r in val_sites.items():
            log(f"    [{site:<8}] n={r['n']:<6} loss={fmt(r['loss'])} "
                f"acc={fmt(r['acc'])} auc={fmt(r['auc'])}")

        if val_auc == val_auc and val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.best_round = server_round
            if self._last_state_dict is not None:
                torch.save(self._last_state_dict,
                           self.output_dir / "model_best_val_auc.pth")
                log(f"new best val AUC {fmt(val_auc)} at round {server_round} "
                    f"— checkpoint saved")

        self.write_outputs(server_round)

    # ---------- outputs ----------

    def _write_tensorboard(self, rnd, gt, val_loss, val_acc, val_auc,
                           train_sites, val_sites, round_duration):
        w = self.writer

        w.add_scalar("Global/Train/Loss", gt.get("loss", float("nan")), rnd)
        w.add_scalar("Global/Train/Accuracy", gt.get("acc", float("nan")), rnd)
        w.add_scalar("Global/Train/AUC", gt.get("auc", float("nan")), rnd)
        w.add_scalar("Global/Val/Loss", val_loss, rnd)
        w.add_scalar("Global/Val/Accuracy", val_acc, rnd)
        w.add_scalar("Global/Val/AUC", val_auc, rnd)

        for site, r in train_sites.items():
            w.add_scalar(f"Site_{site}/Train/Loss", r["loss"], rnd)
            w.add_scalar(f"Site_{site}/Train/Accuracy", r["acc"], rnd)
            w.add_scalar(f"Site_{site}/Train/AUC", r["auc"], rnd)
        for site, r in val_sites.items():
            w.add_scalar(f"Site_{site}/Val/Loss", r["loss"], rnd)
            w.add_scalar(f"Site_{site}/Val/Accuracy", r["acc"], rnd)
            w.add_scalar(f"Site_{site}/Val/AUC", r["auc"], rnd)
            w.add_scalar(f"Site_{site}/Val/n", r["n"], rnd)

        def overlay(tag, rows, key, global_val):
            d = {s: r[key] for s, r in rows.items() if r[key] == r[key]}
            if global_val == global_val:
                d["global"] = global_val
            if d:
                w.add_scalars(tag, d, rnd)

        overlay("Compare/Train/Loss", train_sites, "loss", gt.get("loss", float("nan")))
        overlay("Compare/Train/Accuracy", train_sites, "acc", gt.get("acc", float("nan")))
        overlay("Compare/Train/AUC", train_sites, "auc", gt.get("auc", float("nan")))
        overlay("Compare/Val/Loss", val_sites, "loss", val_loss)
        overlay("Compare/Val/Accuracy", val_sites, "acc", val_acc)
        overlay("Compare/Val/AUC", val_sites, "auc", val_auc)

        w.add_scalar("Timing/round_duration_s", round_duration, rnd)
        w.add_scalar("Timing/fit_duration_s",
                     self.pending_train.get("duration_s", float("nan")), rnd)
        w.add_scalar("Timing/elapsed_min", (time.time() - self.t_start) / 60.0, rnd)
        w.flush()

    def write_outputs(self, server_round, status="running", failed_round=None,
                      failed_phase=None, failure_reasons=None):
        plot_loss_curves(self.history["train_loss"], self.history["val_loss"],
                         self.output_dir / "loss_curves.png")
        plot_auc_curves(self.history["train_auc"], self.history["val_auc"],
                        self.output_dir / "auc_curves.png")

        payload = {
            "status": status,
            "rounds_completed": server_round,
            "started_at": datetime.fromtimestamp(self.t_start).strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": now_iso(),
            "elapsed_s": round(time.time() - self.t_start, 2),
            "history": self.history,
            "per_site_history": self.per_site_history,
            "best_val_auc": self.best_val_auc if self.best_val_auc >= 0 else None,
            "best_round": self.best_round if self.best_round > 0 else None,
            "final_val_auc": self.history["val_auc"][-1] if self.history["val_auc"] else None,
            "final_val_loss": self.history["val_loss"][-1] if self.history["val_loss"] else None,
            "failed_round": failed_round,
            "failed_phase": failed_phase,
            "failure_reasons": failure_reasons,
        }
        with open(self.output_dir / "final_metrics.json", "w") as f:
            json.dump(payload, f, indent=2)


# ---------------------------------------------------------------- orchestrator

class Orchestrator:
    def __init__(self, run, agg, registry, min_nodes, round_timeout):
        self.run = run
        self.agg = agg
        self.registry = registry
        self.min_nodes = int(min_nodes)
        self.round_timeout = float(round_timeout)

    def _drop_check(self, server_round, phase):
        """Any participant that has gone silent aborts the run."""
        dropped = self.registry.sweep()
        relevant = [(s, r) for s, r in dropped if s in self.run.participants]
        if relevant:
            reasons = [
                f"{self.registry.name(s)} dropped during round {server_round} {phase}: {r}"
                for s, r in relevant
            ]
            self.run.abort(reasons)
            self.agg.abort(server_round, phase, reasons)

    def _client_reported_abort(self, server_round, phase):
        with self.run.cv:
            reasons = self.run.abort_reasons
        if reasons:
            self.agg.abort(server_round, phase, reasons)

    def wait_for_participants(self):
        log(f"waiting for {self.min_nodes} site(s) to connect...")
        deadline = None
        while True:
            self.registry.sweep()
            connected = self.registry.connected()
            if len(connected) >= self.min_nodes:
                self.run.participants = set(connected)
                names = ", ".join(sorted(self.registry.name(s) for s in connected))
                event(f"all {len(connected)} site(s) connected: {names} — starting run",
                      CYAN)
                return
            self._client_reported_abort(0, "join")
            time.sleep(1.0)

    def collect(self, server_round, phase, store_getter):
        deadline = time.time() + self.round_timeout
        while True:
            with self.run.cv:
                have = set(store_getter().keys())
                if self.run.participants.issubset(have):
                    return dict(store_getter())
                self.run.cv.wait(1.0)

            self._client_reported_abort(server_round, phase)
            self._drop_check(server_round, phase)

            if time.time() > deadline:
                with self.run.cv:
                    missing = self.run.participants - set(store_getter().keys())
                reasons = [
                    f"round timeout of {self.round_timeout:.0f}s expired waiting for "
                    f"{self.registry.name(s)} in {phase}" for s in sorted(missing)
                ]
                self.run.abort(reasons)
                self.agg.abort(server_round, phase, reasons)

    def run_rounds(self, fit_config_fn, eval_config_fn):
        for rnd in range(1, self.run.num_rounds + 1):
            self.agg.start_round(rnd)
            self.run.begin_phase("fit", rnd, fit_config_fn(rnd))
            raw = self.collect(rnd, "fit", lambda: self.run.fit_results)

            results = [(s, n, p, m) for s, (n, p, m) in raw.items()]
            if len(results) < self.min_nodes:
                self.agg.abort(rnd, "fit", [
                    f"only {len(results)} of {self.min_nodes} expected client(s) "
                    f"returned a result for round {rnd} fit"
                ])
            new_params = self.agg.aggregate_fit(rnd, results, self.run.reference)
            self.run.set_params(new_params)

            self.agg.start_evaluate(rnd)
            self.run.begin_phase("evaluate", rnd, eval_config_fn(rnd))
            raw = self.collect(rnd, "evaluate", lambda: self.run.eval_results)

            ev = [(s, loss, n, m) for s, (loss, n, m) in raw.items()]
            if len(ev) < self.min_nodes:
                self.agg.abort(rnd, "evaluate", [
                    f"only {len(ev)} of {self.min_nodes} expected client(s) "
                    f"returned a result for round {rnd} evaluate"
                ])
            self.agg.aggregate_evaluate(rnd, ev)

        self.run.finish("done")
        self.agg.write_outputs(self.run.num_rounds, status="finished")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Federated server for the MS classifier")
    ap.add_argument("--pyproject", default="pyproject.toml")
    ap.add_argument("--run-config", default="",
                    help="overrides, e.g. \"lr=0.001 batch-size=8 trial-tag='x'\"")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    rc, server_cfg, sites_cfg = load_config(args.pyproject, args.run_config)

    num_rounds   = int(rc["num-server-rounds"])
    min_nodes    = int(rc["min-nodes"])
    lr           = float(rc["lr"])
    weight_decay = float(rc["weight-decay"])
    local_epochs = int(rc["local-epochs"])
    batch_size   = int(rc["batch-size"])
    trial_tag    = str(rc["trial-tag"])

    host = args.host or server_cfg.get("host", DEFAULT_HOST)
    port = int(args.port or server_cfg.get("port", DEFAULT_PORT))
    round_timeout = float(server_cfg.get("round-timeout-s", DEFAULT_ROUND_TIMEOUT_S))
    hb_timeout = float(server_cfg.get("heartbeat-timeout-s", DEFAULT_HEARTBEAT_TIMEOUT_S))

    output_dir = RESULTS_ROOT / trial_tag
    tb_dir = TB_ROOT / trial_tag
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    setup_logging(output_dir)

    log("=" * 60)
    log(f"torch  {torch.__version__}")
    log(f"numpy  {np.__version__}")
    log(f"python {platform.python_version()}")
    log(f"host   {platform.node()}")
    log("=" * 60)
    log(f"trial={trial_tag} lr={lr} wd={weight_decay} epochs={local_epochs} "
        f"bs={batch_size} rounds={num_rounds} min_nodes={min_nodes}")
    log(f"results  -> {output_dir}")
    log(f"tensorboard -> {tb_dir}")
    log(f"round_timeout={round_timeout}s heartbeat_timeout={hb_timeout}s, "
        f"abort-on-first-failure enabled")

    registry = SiteRegistry(sites_cfg, hb_timeout)
    for sid in registry.site_ids():
        log(f"configured site '{sid}' -> {registry.name(sid)}")

    reference = OrderedDict(
        (k, v.detach().cpu().numpy().copy())
        for k, v in get_resnet50_binary().state_dict().items()
    )
    log(f"global model initialised from ImageNet weights "
        f"({len(reference)} tensors)")

    run = Run(registry, reference, num_rounds)
    agg = Aggregator(output_dir, tb_dir, dict(rc), min_nodes, registry)

    Handler.RUN = run
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    event(f"federated server listening on {host}:{port}", CYAN)

    orch = Orchestrator(run, agg, registry, min_nodes, round_timeout)

    def fit_config(server_round: int):
        return {
            "server_round": server_round,
            "lr": lr,
            "weight_decay": weight_decay,
            "epochs": local_epochs,
            "batch_size": batch_size,
        }

    def eval_config(server_round: int):
        return {"server_round": server_round, "batch_size": batch_size}

    try:
        orch.wait_for_participants()
        orch.run_rounds(fit_config, eval_config)
    except RunAborted as exc:
        run.finish("aborted")
        time.sleep(2.0)  # let clients observe the aborted phase and exit cleanly
        print(paint(str(exc), RED), file=sys.stderr, flush=True)
        httpd.shutdown()
        sys.exit(1)
    except KeyboardInterrupt:
        reasons = ["interrupted from the terminal (SIGINT)"]
        run.abort(reasons)
        try:
            agg.abort(run.round, run.phase, reasons)
        except RunAborted:
            pass
        httpd.shutdown()
        sys.exit(130)
    except BaseException as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        reasons = [f"server-side {type(exc).__name__}: {exc}\n{tb}"]
        run.abort(reasons)
        try:
            agg.abort(run.round, run.phase, reasons)
        except RunAborted:
            pass
        httpd.shutdown()
        sys.exit(1)

    event(f"run {trial_tag} finished — best val AUC {fmt(agg.best_val_auc)} "
          f"at round {agg.best_round}", GREEN)
    agg.writer.close()
    httpd.shutdown()


if __name__ == "__main__":
    main()