#!/usr/bin/env python3
"""
Federated client for the ResNet50 MS classifier. No flwr.

Pulls the global weights from the server, trains locally, pushes weights and
scalar metrics back. Nothing inbound is needed here, so this runs unchanged on
a SLURM compute node.

A background heartbeat keeps the server's connection state current during long
local epochs. Any fatal error is written to runs/fl_clients/<site>/<trial>/
failure.log, reported to the server so the run aborts with a named reason, and
then the process exits non-zero.
"""
import argparse
import gc
import json
import os
import platform
import sys
import threading
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from downsampler import build_downsampler
from fl_common import (GREEN, META_HEADER, RED, deserialize_params, manifest_of,
                       now_hms, now_iso, paint, serialize_params)
from model import get_resnet50_binary
from my_datasets import (UCD_Dataset, UniPD_Dataset, CachedOCTDataset,
                         get_cached_train_transform, get_cached_val_transform,
                         get_train_transform, get_val_transform,
                         read_cache_manifest)
from train import evaluate, safe_auc, train

DATASETS = {"unipd": UniPD_Dataset, "ucd": UCD_Dataset}

DEFAULT_GPU_INDEX = 0
FAILURE_LOG_ROOT = Path("runs") / "fl_clients"
HEARTBEAT_INTERVAL_S = 10.0
POLL_INTERVAL_S = 5.0
HTTP_TIMEOUT_S = 1800.0
HTTP_RETRIES = 3
HTTP_BACKOFF_S = 3.0


def write_failure(dataset_name, trial_tag, phase, server_round, exc):
    """Persist the traceback before the process exits."""
    out_dir = FAILURE_LOG_ROOT / str(dataset_name) / str(trial_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "failure.log"

    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    detail = (
        f"CLIENT ABORTED at {now_iso()}\n"
        f"  site={dataset_name} trial={trial_tag} phase={phase} round={server_round}\n"
        f"  pid={os.getpid()} ppid={os.getppid()}\n"
        f"  {type(exc).__name__}: {exc}\n{tb}"
    )
    with open(path, "w") as f:
        f.write(detail)

    print(detail, file=sys.stderr, flush=True)
    print(f"[client:{dataset_name}] failure written to {path.resolve()}",
          file=sys.stderr, flush=True)

    if torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            print(f"[client:{dataset_name}] gpu at failure "
                  f"free={free_b / 1024**3:.2f}GiB total={total_b / 1024**3:.2f}GiB",
                  file=sys.stderr, flush=True)
        except Exception:
            pass
    return detail


def get_dataloaders(dataset_name, data_path, batch_size=4, image_size=512,
                    cache_path=None):
    if cache_path:
        manifest = read_cache_manifest(cache_path)

        expected_crop = dataset_name.lower() == "unipd"
        if bool(manifest["crop"]) != expected_crop:
            raise ValueError(
                f"cache at {cache_path} was built with crop={manifest['crop']} "
                f"but site '{dataset_name}' needs crop={expected_crop}"
            )
        if int(manifest["image_size"]) != int(image_size):
            raise ValueError(
                f"cache at {cache_path} was built at image_size="
                f"{manifest['image_size']} but this run wants {image_size}"
            )

        train_dir = os.path.join(cache_path, "train")
        val_dir = os.path.join(cache_path, "val")
        print(f"[client:{dataset_name}] cached train={train_dir} val={val_dir} "
              f"bs={batch_size} (built {manifest['created_at']})")

        train_ds = CachedOCTDataset(train_dir, transform=get_cached_train_transform())
        val_ds = CachedOCTDataset(val_dir, transform=get_cached_val_transform())
        num_workers = 8
    else:
        cls = DATASETS.get(dataset_name.lower())
        if cls is None:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Expected one of {list(DATASETS)}"
            )

        crop = dataset_name.lower() == "unipd"
        train_dir = os.path.join(data_path, "train")
        val_dir = os.path.join(data_path, "val")
        print(
            f"[client:{dataset_name}] train={train_dir} val={val_dir} bs={batch_size} crop={crop}"
        )

        train_ds = cls(
            img_dir=train_dir,
            transform=get_train_transform(image_size, crop_img=crop),
        )
        val_ds = cls(
            img_dir=val_dir, transform=get_val_transform(image_size, crop_img=crop)
        )
        num_workers = 4

    sampler = build_downsampler(np.array(train_ds.targets))
    trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0
    )
    testloader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0
    )

    num_examples = {"trainset": len(sampler), "testset": len(val_ds)}
    return trainloader, testloader, num_examples


# ---------------------------------------------------------------- transport

class Transport:
    """Thin stdlib HTTP client. Retries transient network errors, not HTTP 4xx."""

    def __init__(self, base_url, token, site):
        self.base = base_url.rstrip("/")
        self.token = token
        self.site = site

    def _request(self, method, path, body=None, meta=None, binary_response=False):
        url = f"{self.base}{path}"
        headers = {"Authorization": f"Bearer {self.token}"}
        if meta is not None:
            headers[META_HEADER] = json.dumps(meta)
        if body is not None:
            headers["Content-Type"] = "application/octet-stream"

        last = None
        for attempt in range(1, HTTP_RETRIES + 1):
            req = Request(url, data=body, headers=headers, method=method)
            try:
                with urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
                    payload = resp.read()
                    if binary_response:
                        return payload
                    return json.loads(payload.decode("utf-8")) if payload else {}
            except HTTPError as e:
                detail = e.read().decode("utf-8", "replace")
                raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {detail}")
            except (URLError, TimeoutError, ConnectionError, OSError) as e:
                last = e
                if attempt < HTTP_RETRIES:
                    time.sleep(HTTP_BACKOFF_S * attempt)
        raise RuntimeError(f"{method} {path} failed after {HTTP_RETRIES} attempts: {last}")

    def join(self):
        return self._request("POST", "/join", body=b"")

    def status(self):
        return self._request("GET", "/status")

    def params(self):
        return self._request("GET", "/params", binary_response=True)

    def post_fit(self, server_round, num_examples, metrics, blob):
        meta = {"round": server_round, "num_examples": num_examples, "metrics": metrics}
        return self._request("POST", "/fit", body=blob, meta=meta)

    def post_evaluate(self, server_round, loss, num_examples, metrics):
        meta = {"round": server_round, "loss": loss,
                "num_examples": num_examples, "metrics": metrics}
        return self._request("POST", "/evaluate", body=b"", meta=meta)

    def post_abort(self, phase, server_round, message):
        meta = {"phase": phase, "round": server_round, "message": message}
        return self._request("POST", "/abort", body=b"", meta=meta)


# ---------------------------------------------------------------- client

class OCTClient:

    def __init__(self, model, dataset_name, data_path, trial_tag, device,
                 cache_path=None):
        self.model = model
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.trial_tag = trial_tag
        self.device = device
        self.cache_path = cache_path
        self._cache = {}
        self.manifest = manifest_of(OrderedDict(
            (k, v.detach().cpu().numpy()) for k, v in model.state_dict().items()
        ))

        log_dir = os.path.join("runs", "fl_clients", self.dataset_name, trial_tag)
        self.writer = SummaryWriter(log_dir=log_dir)

    def _loaders(self, batch_size):
        if batch_size not in self._cache:
            self._cache[batch_size] = get_dataloaders(
                self.dataset_name, self.data_path, batch_size=batch_size,
                cache_path=self.cache_path
            )
        return self._cache[batch_size]

    def _release(self):
        """Drop dead references and return cached blocks to the CUDA allocator."""
        gc.collect()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def _log_gpu(self, tag):
        if self.device.type != "cuda":
            return
        alloc = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        peak = torch.cuda.max_memory_allocated(self.device) / 1024**3
        print(
            f"[client:{self.dataset_name}] gpu {tag} "
            f"alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB"
        )

    def get_parameters(self):
        return OrderedDict(
            (k, v.detach().to("cpu", copy=True).numpy())
            for k, v in self.model.state_dict().items()
        )

    def set_parameters(self, blob):
        params = deserialize_params(blob, expect=self.manifest)
        state_dict = OrderedDict(
            (k, torch.as_tensor(v)) for k, v in params.items()
        )
        self.model.load_state_dict(state_dict, strict=True)
        del state_dict, params
        self._release()

    def fit(self, rnd, config):
        lr = float(config["lr"])
        weight_decay = float(config["weight_decay"])
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])

        print(
            f"[client:{self.dataset_name}] round {rnd} "
            f"lr={lr} wd={weight_decay} epochs={epochs} bs={batch_size}"
        )

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._log_gpu(f"round {rnd} fit-start")

        trainloader, _, num_examples = self._loaders(batch_size)
        try:
            train(
                self.model,
                trainloader,
                epochs=epochs,
                device=self.device,
                lr=lr,
                weight_decay=weight_decay,
            )
        finally:
            self._release()

        try:
            with torch.no_grad():
                loss_tr, acc_tr, y_true, y_score = evaluate(
                    self.model, trainloader, device=self.device
                )
        finally:
            self._release()

        auc_tr = safe_auc(y_true, y_score)
        auc_tr_clean = float(auc_tr) if auc_tr == auc_tr else 0.0

        print(
            f"[client:{self.dataset_name}] train  loss={loss_tr:.4f} "
            f"acc={acc_tr:.4f} auc={auc_tr_clean:.4f}"
        )
        self._log_gpu(f"round {rnd} fit-end")

        if rnd > 0:
            self.writer.add_scalar("Train/Loss", loss_tr, rnd)
            self.writer.add_scalar("Train/Accuracy", acc_tr, rnd)
            self.writer.add_scalar("Train/AUC", auc_tr_clean, rnd)
            self.writer.flush()

        params = self.get_parameters()
        del y_true, y_score
        self._release()

        metrics = {
            "site": self.dataset_name,
            "accuracy": float(acc_tr),
            "loss": float(loss_tr),
            "auc": auc_tr_clean,
        }
        return params, num_examples["trainset"], metrics

    def evaluate(self, rnd, config):
        batch_size = int(config.get("batch_size", 4))
        _, testloader, num_examples = self._loaders(batch_size)

        self._log_gpu(f"round {rnd} eval-start")

        try:
            with torch.no_grad():
                loss, acc, y_true, y_score = evaluate(
                    self.model, testloader, device=self.device
                )
        finally:
            self._release()

        auc = safe_auc(y_true, y_score)
        auc_clean = float(auc) if auc == auc else 0.0

        print(
            f"[client:{self.dataset_name}] round {rnd} val  loss={loss:.4f} "
            f"acc={acc:.4f} auc={auc_clean:.4f}"
        )
        self._log_gpu(f"round {rnd} eval-end")

        if rnd > 0:
            self.writer.add_scalar("Validation/Loss", loss, rnd)
            self.writer.add_scalar("Validation/Accuracy", acc, rnd)
            self.writer.add_scalar("Validation/AUC", auc_clean, rnd)
            self.writer.flush()

        del y_true, y_score
        self._release()

        metrics = {
            "site": self.dataset_name,
            "accuracy": float(acc),
            "loss": float(loss),
            "auc": auc_clean,
        }
        return float(loss), num_examples["testset"], metrics


# ---------------------------------------------------------------- heartbeat

class Heartbeat(threading.Thread):
    """Keeps the server's connection state warm during long local epochs."""

    def __init__(self, transport, dataset_name):
        super().__init__(daemon=True)
        self.transport = transport
        self.dataset_name = dataset_name
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.wait(HEARTBEAT_INTERVAL_S):
            try:
                self.transport.status()
            except Exception as e:
                print(f"[client:{self.dataset_name}] heartbeat failed: {e}",
                      file=sys.stderr, flush=True)

    def stop(self):
        self.stop_event.set()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Federated client for the MS classifier")
    ap.add_argument("--site", required=True, choices=sorted(DATASETS.keys()),
                    help="site id, must match a [tool.fl.sites.<id>] on the server")
    ap.add_argument("--data-path", required=True,
                    help="directory holding train/ and val/")
    ap.add_argument("--server", required=True,
                    help="e.g. http://140.226.4.75:9092")
    ap.add_argument("--token", default=os.environ.get("FL_TOKEN", ""),
                    help="site token; defaults to $FL_TOKEN")
    ap.add_argument("--cache-path", default=None,
                    help="directory of .pt tensors from cache_images.py; "
                         "falls back to decoding from --data-path when omitted")
    ap.add_argument("--trial-tag", default="unknown")
    ap.add_argument("--gpu-index", type=int, default=DEFAULT_GPU_INDEX)
    args = ap.parse_args()

    dataset_name = args.site
    trial_tag = args.trial_tag

    if not args.token:
        print("no token supplied; pass --token or set FL_TOKEN", file=sys.stderr)
        sys.exit(2)

    print(f"[client:{dataset_name}] torch={torch.__version__} "
          f"numpy={np.__version__} python={platform.python_version()} "
          f"host={platform.node()}", flush=True)

    transport = Transport(args.server, args.token, dataset_name)
    client = None
    hb = None
    phase, rnd = "startup", 0

    try:
        info = transport.join()
        display = info.get("display_name", dataset_name)
        print(paint(f"[client:{dataset_name}] joined as {display} at {now_hms()} "
                    f"({info.get('num_rounds')} rounds)", GREEN), flush=True)

        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            if args.gpu_index >= n_gpus:
                raise ValueError(
                    f"gpu_index={args.gpu_index} but only {n_gpus} CUDA device(s) visible"
                )
            device = torch.device(f"cuda:{args.gpu_index}")
            torch.cuda.set_device(device)
            free_b, total_b = torch.cuda.mem_get_info(device)
            print(
                f"[client:{dataset_name}] using {device} "
                f"({torch.cuda.get_device_name(device)}) "
                f"free={free_b / 1024**3:.2f}GiB total={total_b / 1024**3:.2f}GiB"
            )
        else:
            device = torch.device("cpu")
            print(f"[client:{dataset_name}] no CUDA available, using cpu")

        model = get_resnet50_binary().to(device)
        client = OCTClient(model, dataset_name, args.data_path, trial_tag, device,
                           cache_path=args.cache_path)

        hb = Heartbeat(transport, dataset_name)
        hb.start()

        last_fit, last_eval = 0, 0
        waiting_on = None

        while True:
            st = transport.status()
            phase, rnd = st["phase"], int(st["round"])

            if phase == "done":
                print(paint(f"[client:{dataset_name}] run finished at {now_hms()}",
                            GREEN), flush=True)
                break

            if phase == "aborted":
                print(paint(f"[client:{dataset_name}] server aborted the run at "
                            f"{now_hms()}", RED), file=sys.stderr, flush=True)
                if hb:
                    hb.stop()
                sys.exit(1)

            if phase == "fit" and not st["submitted"] and rnd > last_fit:
                client.set_parameters(transport.params())
                params, n, metrics = client.fit(rnd, st["config"])
                transport.post_fit(rnd, n, metrics, serialize_params(params))
                del params
                client._release()
                last_fit = rnd
                continue

            if phase == "evaluate" and not st["submitted"] and rnd > last_eval:
                client.set_parameters(transport.params())
                loss, n, metrics = client.evaluate(rnd, st["config"])
                transport.post_evaluate(rnd, loss, n, metrics)
                last_eval = rnd
                continue

            if st["submitted"]:
                key = (phase, rnd)
                if waiting_on != key:
                    waiting_on = key
                    print(f"[client:{dataset_name}] submitted round {rnd} {phase} "
                          f"at {now_hms()} — waiting for other sites", flush=True)

            time.sleep(POLL_INTERVAL_S)

    except SystemExit:
        raise
    except BaseException as exc:
        detail = write_failure(dataset_name, trial_tag, phase, rnd, exc)
        try:
            transport.post_abort(phase, rnd, f"{type(exc).__name__}: {exc}")
        except Exception as e:
            print(f"[client:{dataset_name}] could not notify server: {e}",
                  file=sys.stderr, flush=True)
        if hb:
            hb.stop()
        if client is not None:
            try:
                client.writer.flush()
                client.writer.close()
            except Exception:
                pass
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)

    if hb:
        hb.stop()
    if client is not None:
        try:
            client.writer.flush()
            client.writer.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()