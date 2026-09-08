"""
Shared wire format and console helpers for the federated MS classifier.

Dependencies: numpy + stdlib only. Deliberately no torch here, so the
serialization format is independent of the torch version at either site.

Wire format: a .npz archive holding one array per state_dict entry, keyed
positionally (a0, a1, ...), plus a "__manifest__" entry that is the JSON
[[name, shape, dtype], ...] listing in state_dict order. Arrays are stored as
at least 1-d and restored to their declared shape on read, because 0-d entries
(BatchNorm's num_batches_tracked, 53 of them in ResNet50) do not round-trip
through npz as 0-d. Loading is done with allow_pickle=False and validated
against the receiver's own manifest, so a payload can never introduce an
unexpected key, an unexpected element count, or executable content.
"""
import io
import json
import sys
from collections import OrderedDict
from datetime import datetime

import numpy as np

WIRE_VERSION = 1
META_HEADER = "X-FL-Meta"
MAX_BODY_BYTES = 512 * 1024 * 1024

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_hms() -> str:
    return datetime.now().strftime("%H:%M:%S")


def paint(text: str, color: str, stream=None) -> str:
    """Wrap text in an ANSI color, but only when writing to a real terminal."""
    stream = stream or sys.stdout
    try:
        if not stream.isatty():
            return text
    except Exception:
        return text
    return f"{color}{text}{RESET}"


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def manifest_of(params) -> list:
    """params: mapping name -> ndarray-like. Returns [[name, shape, dtype], ...]."""
    out = []
    for k, v in params.items():
        a = np.asarray(v)
        out.append([str(k), [int(x) for x in a.shape], str(a.dtype)])
    return out


def serialize_params(params) -> bytes:
    """
    Serialize an ordered mapping of name -> ndarray into npz bytes.

    Arrays go on the wire as at least 1-d; the manifest carries the true shape
    so deserialize_params can restore 0-d entries exactly.
    """
    order = list(params.keys())
    payload = {}
    for i, k in enumerate(order):
        arr = np.asarray(params[k])
        payload[f"a{i}"] = np.ascontiguousarray(np.atleast_1d(arr))
    manifest = manifest_of(params)
    payload["__manifest__"] = np.frombuffer(
        json.dumps(manifest).encode("utf-8"), dtype=np.uint8
    )
    buf = io.BytesIO()
    np.savez(buf, **payload)
    return buf.getvalue()


def deserialize_params(blob: bytes, expect=None) -> "OrderedDict":
    """
    Inverse of serialize_params.

    Element counts are validated against the manifest and each array is
    reshaped to its declared shape. When expect is supplied (a manifest from
    manifest_of), the payload must match it exactly in names, order and shape,
    and arrays are cast to the expected dtype. Raises ValueError on any
    mismatch; nothing is ever unpickled.
    """
    if len(blob) > MAX_BODY_BYTES:
        raise ValueError(f"payload of {len(blob)} bytes exceeds cap {MAX_BODY_BYTES}")

    out = OrderedDict()
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        if "__manifest__" not in z.files:
            raise ValueError("payload has no __manifest__ entry")
        manifest = json.loads(z["__manifest__"].tobytes().decode("utf-8"))

        if len(z.files) != len(manifest) + 1:
            raise ValueError(
                f"payload holds {len(z.files) - 1} arrays but manifest lists {len(manifest)}"
            )

        for i, entry in enumerate(manifest):
            name, shape, dtype = entry[0], entry[1], entry[2]
            key = f"a{i}"
            if key not in z.files:
                raise ValueError(f"payload is missing array {key} for '{name}'")
            arr = z[key]
            if arr.size != _numel(shape):
                raise ValueError(
                    f"'{name}': payload holds {arr.size} elements, "
                    f"declared shape {list(shape)} needs {_numel(shape)}"
                )
            out[name] = arr.reshape(tuple(int(d) for d in shape))

    if expect is not None:
        if len(expect) != len(out):
            raise ValueError(
                f"payload has {len(out)} tensors, expected {len(expect)}"
            )
        rebuilt = OrderedDict()
        for (exp_name, exp_shape, exp_dtype), (got_name, got_arr) in zip(
            expect, out.items()
        ):
            if exp_name != got_name:
                raise ValueError(
                    f"tensor order mismatch: expected '{exp_name}', got '{got_name}'"
                )
            if list(got_arr.shape) != list(exp_shape):
                raise ValueError(
                    f"'{exp_name}': shape {list(got_arr.shape)} != expected {list(exp_shape)}"
                )
            rebuilt[exp_name] = got_arr.astype(np.dtype(exp_dtype), copy=False)
        return rebuilt

    return out


def fedavg(results, reference):
    """
    Weighted FedAvg over every entry of the state dict, BatchNorm buffers
    included, matching the previous flwr behaviour.

    results: [(site_id, num_examples, params_mapping, metrics), ...]
    reference: ordered mapping name -> ndarray, supplying key order and dtypes.
    Accumulation is in float64, then cast back to the reference dtype, so
    num_batches_tracked lands back on int64 exactly as load_state_dict used to
    coerce it.
    """
    total = sum(int(n) for _, n, _, _ in results)
    if total <= 0:
        raise ValueError("total num_examples across clients is zero")

    out = OrderedDict()
    for name, ref in reference.items():
        acc = None
        for _, n, params, _ in results:
            if name not in params:
                raise ValueError(f"client update is missing tensor '{name}'")
            contrib = np.asarray(params[name], dtype=np.float64) * (int(n) / total)
            acc = contrib if acc is None else acc + contrib
        out[name] = acc.astype(np.asarray(ref).dtype, copy=False)
    return out