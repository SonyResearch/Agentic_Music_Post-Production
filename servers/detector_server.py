#!/usr/bin/env python
"""HTTP server for the anomaly-fx detector (detector/afxdet), used by run_param_tuning.py.

    python servers/detector_server.py --port 8200 --device cuda:0

Loading MusicFM + RLAT takes about a minute, so the model is loaded once here and
every /predict call only computes embeddings and runs the small detector transformer.

Routes (JSON over HTTP):
    GET  /health   -> {"ok": true, ...model info...}
    POST /blobs    {"blobs": [{"sha256", "b64"}]} -> {"stored": [sha, ...]}
                   (audio files are uploaded once and referenced by sha256)
    POST /predict  {"scenario_kind": "subgroup"|"full", "threshold": float,
                    "mix": <blob ref>,
                    "units": [{"name", "owner_id", "chain": [tool, ...],
                               "dry": <blob ref>, "wet": <blob ref>}, ...]}
                   -> the detector's result dict, plus owner_id echoed onto each unit;
                      409 {"missing": [sha, ...]} if a referenced blob was never uploaded.
A blob ref is {"sha256": hex} or {"sha256": hex, "b64": inline-data}.

The detector runs with latent_mode="sample" and seed=0 (training-consistent and
deterministic). Audio is decoded like afxdet.detector.load_audio: float32 (T, C) at the
file's native rate. One lock serialises GPU work.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETECTOR_ROOT = os.path.join(REPO_ROOT, "detector")

SCENARIO_KINDS = ("subgroup", "full")
# Mirrors afxdet/model.py; used only for --fake validation. In real mode the values
# imported from afxdet win, and a drift between the two is reported at startup.
FALLBACK_TOOL_TYPES = ("eq", "compressor", "noisegate", "distortion", "reverb", "delay",
                       "stereo_imager", "gain_panning", "limiter", "multiband_compressor")
FALLBACK_MAX_CHAIN_LEN = 12
FALLBACK_MAX_UNITS = 32


# ======================================================================================
# Blob protocol (same as servers/critic_server.py)
# ======================================================================================

class HttpError(Exception):
    """An error with a chosen HTTP status and JSON body."""

    def __init__(self, status, payload):
        super().__init__(json.dumps(payload))
        self.status = status
        self.payload = payload


class BlobStore:
    """sha256 -> bytes, LRU-capped. Audio is uploaded once and referenced by hash."""

    def __init__(self, cap_bytes=2 * 1024 ** 3):
        self.cap_bytes = cap_bytes
        self._d = OrderedDict()
        self._n = 0
        self._lock = threading.Lock()

    def put(self, sha, data):
        sha = str(sha).lower()
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha:
            raise HttpError(400, {"error": f"sha256 mismatch: declared {sha}, got {actual}"})
        with self._lock:
            if sha in self._d:
                self._d.move_to_end(sha)
                return sha
            self._d[sha] = data
            self._n += len(data)
            while self._n > self.cap_bytes and len(self._d) > 1:
                _, evicted = self._d.popitem(last=False)
                self._n -= len(evicted)
        return sha

    def get(self, sha):
        with self._lock:
            data = self._d.get(str(sha).lower())
            if data is not None:
                self._d.move_to_end(str(sha).lower())
            return data

    def stats(self):
        with self._lock:
            return {"count": len(self._d), "bytes": self._n, "cap_bytes": self.cap_bytes}

    # -- reference resolution ----------------------------------------------------------
    def resolve(self, ref, field, stats):
        """{"sha256": ..., "b64": <optional>} -> bytes, or None (recorded as missing)."""
        if isinstance(ref, str):
            ref = {"sha256": ref}
        if not isinstance(ref, dict):
            raise HttpError(400, {"error": f"{field}: blob ref must be an object with 'sha256'"})
        sha = ref.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise HttpError(400, {"error": f"{field}: 'sha256' must be a 64-char hex digest"})
        sha = sha.lower()
        try:
            int(sha, 16)
        except ValueError:
            raise HttpError(400, {"error": f"{field}: 'sha256' is not hex"})
        data = self.get(sha)
        if data is not None:
            stats["hits"] += 1
            return data
        inline = ref.get("b64")
        if isinstance(inline, str):
            data = _b64_decode(inline, field)
            self.put(sha, data)
            stats["inline"] += 1
            return data
        stats["missing"].append(sha)
        return None


def _b64_decode(text, field):
    try:
        return base64.b64decode(text, validate=True)
    except Exception as exc:                                    # noqa: BLE001
        raise HttpError(400, {"error": f"{field}: bad base64 ({exc!r})"})


def store_blobs(store, payload):
    """POST /blobs body -> {"stored": [sha, ...]}."""
    blobs = payload.get("blobs")
    if not isinstance(blobs, list):
        raise HttpError(400, {"error": "'blobs' must be a list of {sha256, b64}"})
    stored = []
    for i, entry in enumerate(blobs):
        if not isinstance(entry, dict):
            raise HttpError(400, {"error": f"blobs[{i}] must be an object"})
        sha = entry.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise HttpError(400, {"error": f"blobs[{i}]: 'sha256' must be a 64-char hex digest"})
        b64 = entry.get("b64")
        if not isinstance(b64, str):
            raise HttpError(400, {"error": f"blobs[{i}]: 'b64' must be a base64 string"})
        stored.append(store.put(sha.lower(), _b64_decode(b64, f"blobs[{i}]")))
    return {"stored": stored}


def new_blob_stats():
    return {"hits": 0, "inline": 0, "missing": []}


def raise_if_missing(stats):
    if stats["missing"]:
        seen, ordered = set(), []
        for sha in stats["missing"]:
            if sha not in seen:
                seen.add(sha)
                ordered.append(sha)
        raise HttpError(409, {"missing": ordered})


def blob_note(stats):
    return (f"blobs hit={stats['hits']} inline={stats['inline']} "
            f"missing={len(stats['missing'])}")


# ======================================================================================
# Audio decode — mirrors afxdet.detector.load_audio (float32, (T, C), native sr)
# ======================================================================================

def decode_audio(raw, field):
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception as exc:                                                # noqa: BLE001
        raise HttpError(400, {"error": f"{field}: could not decode audio ({exc!r})"})
    return np.ascontiguousarray(audio), int(sr)


# ======================================================================================
# Detector
# ======================================================================================

class DetectorService:
    def __init__(self, args):
        self.args = args
        self.fake = bool(args.fake)
        self.lock = threading.Lock()          # serialises the GPU; one predict at a time
        self.detector = None
        self.Unit = None
        self.tool_types = list(FALLBACK_TOOL_TYPES)
        self.max_chain_len = FALLBACK_MAX_CHAIN_LEN
        self.max_units = FALLBACK_MAX_UNITS
        self.info = {"fake": True} if self.fake else {}
        if self.fake:
            return

        root = DETECTOR_ROOT
        if root not in sys.path:
            sys.path.insert(0, root)          # exactly what detector/infer.py does
        print(f"[init] importing afxdet from {root}", flush=True)
        from afxdet import AnomalyFxDetector, Unit                          # noqa: N806
        from afxdet.detector import DEFAULT_CHECKPOINT
        from afxdet.model import MAX_CHAIN_LEN, TOOL_TYPES

        self.Unit = Unit
        self.tool_types = list(TOOL_TYPES)
        self.max_chain_len = int(MAX_CHAIN_LEN)
        if set(self.tool_types) != set(FALLBACK_TOOL_TYPES):
            print(f"[init] NOTE afxdet TOOL_TYPES differ from this file's fallback list: "
                  f"{sorted(self.tool_types)}", flush=True)

        checkpoint = args.checkpoint or str(DEFAULT_CHECKPOINT)
        if not os.path.isfile(checkpoint):
            raise SystemExit(f"detector checkpoint not found: {checkpoint}")
        print(f"[init] loading detector ckpt={checkpoint} device={args.device} "
              f"latent_mode={args.latent_mode} seed={args.seed}", flush=True)
        t0 = time.time()
        self.detector = AnomalyFxDetector(
            checkpoint=checkpoint,
            device=args.device,
            latent_mode=args.latent_mode,
            seed=None if args.seed < 0 else args.seed,
        )
        self.max_units = int(getattr(self.detector.config, "max_units", FALLBACK_MAX_UNITS))
        self.info = {
            "checkpoint": checkpoint,
            "load_seconds": round(time.time() - t0, 2),
            "max_units": self.max_units,
            "max_chain_len": self.max_chain_len,
            "tool_types": self.tool_types,
            "embed_dim": int(getattr(self.detector.config, "embed_dim", -1)),
            "training_metrics": _jsonable(getattr(self.detector, "training_metrics", {})),
        }
        print(f"[init] ready in {self.info['load_seconds']}s", flush=True)

    # ----------------------------------------------------------------------------------
    def validate_chain(self, name, chain):
        unknown = [t for t in chain if t not in self.tool_types]
        if unknown:
            raise HttpError(400, {"error": f"unit {name!r}: unknown fx tool(s) {unknown}; "
                                           f"known: {sorted(self.tool_types)}"})
        if len(chain) > self.max_chain_len:
            raise HttpError(400, {"error": f"unit {name!r}: chain of {len(chain)} exceeds "
                                           f"MAX_CHAIN_LEN={self.max_chain_len}"})

    def predict(self, specs, mix, mix_sr, scenario_kind, threshold):
        """specs = [{"name","owner_id","chain","dry":(arr,sr),"wet":(arr,sr)}, ...]."""
        if len(specs) > self.max_units:
            raise HttpError(400, {"error": f"{len(specs)} units exceeds max_units="
                                           f"{self.max_units}"})
        if self.fake:
            with self.lock:
                time.sleep(0.01)
            result = _fake_result(specs, scenario_kind, threshold)
        else:
            units = []
            for s in specs:
                try:
                    units.append(self.Unit(name=s["name"], dry=s["dry"][0], wet=s["wet"][0],
                                           sr=s["dry"][1], chain=list(s["chain"])))
                except ValueError as exc:
                    raise HttpError(400, {"error": repr(exc)})
            with self.lock:
                try:
                    result = self.detector.predict(units, mix, mix_sr,
                                                   scenario_kind=scenario_kind,
                                                   threshold=threshold)
                except ValueError as exc:
                    raise HttpError(400, {"error": repr(exc)})

        # -- echo owner_id onto each unit, by position, name-checked ---------------------
        owners = [s["owner_id"] for s in specs]
        by_name = {}
        for s, owner in zip(specs, owners):
            by_name.setdefault(s["name"], owner)
        for i, entry in enumerate(result.get("units", [])):
            if i < len(specs) and entry.get("unit") == specs[i]["name"]:
                entry["owner_id"] = owners[i]
            else:
                entry["owner_id"] = by_name.get(entry.get("unit"))
        result["threshold"] = threshold
        return result

    def health(self):
        health = {
            "ok": True,
            "device": self.args.device,
            "threshold_default": self.args.threshold_default,
            "latent_mode": self.args.latent_mode,
            "seed": self.args.seed,
            "fake": self.fake,
            "scenario_kinds": list(SCENARIO_KINDS),
        }
        health.update({k: v for k, v in self.info.items() if k != "fake"})
        return health


def _jsonable(obj):
    """Training metrics come out of a torch checkpoint; keep only JSON-safe leaves."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, bool, int, float)) or obj is None:
        return obj
    try:
        return float(obj)
    except Exception:                                                       # noqa: BLE001
        return repr(obj)


def _fake_result(specs, scenario_kind, threshold):
    """Canned verdict with the real result's shape: unit 0 corrupted, module 0 flagged."""
    units = []
    for i, s in enumerate(specs):
        corrupted = (i == 0)
        prob = 0.91 if corrupted else 0.07
        modules = []
        for j, tool in enumerate(s["chain"]):
            mp = 0.88 if (corrupted and j == 0) else 0.05
            modules.append({"position": j, "tool": tool,
                            "corrupted": bool(mp > threshold), "probability": round(mp, 4)})
        units.append({"unit": s["name"], "corrupted": bool(prob > threshold),
                      "probability": round(prob, 4), "modules": modules})
    return {
        "scenario_kind": scenario_kind,
        "threshold": threshold,
        "n_units": len(specs),
        "corrupted_units": [u["unit"] for u in units if u["corrupted"]],
        "units": units,
    }


# ======================================================================================
# HTTP
# ======================================================================================

def make_handler(service, store):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "afxdet_server/1.0"

        # --- plumbing -----------------------------------------------------------------
        def log_message(self, fmt, *a):        # silence BaseHTTPRequestHandler's stderr
            return

        def _log(self, route, status, note="", seconds=None):
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            tail = "" if seconds is None else f" gen={seconds:.2f}s"
            print(f"[{stamp}] {route} -> {status} {note}{tail}", flush=True)

        def _send(self, status, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _read_body(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                raise HttpError(400, {"error": "bad Content-Length"})
            chunks, remaining = [], length
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

        def _read_json(self):
            raw = self._read_body()          # always drained, so keep-alive stays sane
            if not raw:
                raise HttpError(400, {"error": "empty body"})
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception as exc:                                        # noqa: BLE001
                raise HttpError(400, {"error": f"bad JSON ({exc!r})"})
            if not isinstance(payload, dict):
                raise HttpError(400, {"error": "body must be a JSON object"})
            return payload

        # --- routes -------------------------------------------------------------------
        def do_GET(self):                                                   # noqa: N802
            route = f"GET {self.path}"
            try:
                if self.path.split("?")[0] == "/health":
                    health = service.health()
                    health["blobs"] = store.stats()
                    self._send(200, health)
                    self._log(route, 200)
                else:
                    self._send(404, {"error": f"no route {self.path}"})
                    self._log(route, 404)
            except Exception as exc:                                        # noqa: BLE001
                self._send(500, {"error": repr(exc)})
                self._log(route, 500, repr(exc))

        def do_POST(self):                                                  # noqa: N802
            path = self.path.split("?")[0]
            route = f"POST {path}"
            note, seconds, status = "", None, 500
            try:
                payload = self._read_json()
                if path == "/blobs":
                    result = store_blobs(store, payload)
                    status, note = 200, f"stored={len(result['stored'])}"
                elif path == "/predict":
                    result, note, seconds = self._predict(payload)
                    status = 200
                else:
                    raise HttpError(404, {"error": f"no route {path}"})
                self._send(status, result)
            except HttpError as exc:
                status = exc.status
                note = note or json.dumps(exc.payload)[:200]
                self._send(status, exc.payload)
            except Exception as exc:                                        # noqa: BLE001
                status = 500
                note = repr(exc)
                self._send(500, {"error": repr(exc)})
            self._log(route, status, note, seconds)

        def _predict(self, payload):
            scenario_kind = payload.get("scenario_kind", "subgroup")
            if scenario_kind not in SCENARIO_KINDS:
                raise HttpError(400, {"error": f"'scenario_kind' must be one of "
                                               f"{list(SCENARIO_KINDS)}"})
            threshold = payload.get("threshold", service.args.threshold_default)
            try:
                threshold = float(threshold)
            except (TypeError, ValueError):
                raise HttpError(400, {"error": "'threshold' must be a number"})
            if not 0.0 <= threshold <= 1.0:
                raise HttpError(400, {"error": "'threshold' must be in [0, 1]"})

            units = payload.get("units")
            if not isinstance(units, list) or not units:
                raise HttpError(400, {"error": "'units' must be a non-empty list"})
            if "mix" not in payload:
                raise HttpError(400, {"error": "missing audio ref 'mix'"})

            # -- resolve every blob first, so one 409 lists everything the client must send
            stats = new_blob_stats()
            raw_mix = store.resolve(payload["mix"], "mix", stats)
            raw_units = []
            for i, u in enumerate(units):
                if not isinstance(u, dict):
                    raise HttpError(400, {"error": f"units[{i}] must be an object"})
                name = u.get("name")
                if not isinstance(name, str) or not name:
                    raise HttpError(400, {"error": f"units[{i}]: 'name' must be a string"})
                chain = u.get("chain", [])
                if not isinstance(chain, list) or not all(isinstance(t, str) for t in chain):
                    raise HttpError(400, {"error": f"units[{i}]: 'chain' must be a list of "
                                                   f"tool-type strings"})
                service.validate_chain(name, chain)
                owner_id = u.get("owner_id")
                if owner_id is not None:
                    try:
                        owner_id = int(owner_id)
                    except (TypeError, ValueError):
                        raise HttpError(400, {"error": f"units[{i}]: 'owner_id' must be an "
                                                       f"int or null"})
                for field in ("dry", "wet"):
                    if field not in u:
                        raise HttpError(400, {"error": f"units[{i}] ({name}): missing audio "
                                                       f"ref '{field}'"})
                raw_units.append({
                    "name": name,
                    "owner_id": owner_id,
                    "chain": list(chain),
                    "dry": store.resolve(u["dry"], f"units[{i}].dry", stats),
                    "wet": store.resolve(u["wet"], f"units[{i}].wet", stats),
                })
            raise_if_missing(stats)

            mix, mix_sr = decode_audio(raw_mix, "mix")
            specs, sr_seen = [], set()
            for i, u in enumerate(raw_units):
                dry = decode_audio(u["dry"], f"units[{i}].dry")
                wet = decode_audio(u["wet"], f"units[{i}].wet")
                if dry[1] != wet[1]:
                    raise HttpError(400, {"error": f"unit {u['name']!r}: dry {dry[1]} Hz vs "
                                                   f"wet {wet[1]} Hz"})
                sr_seen.add(dry[1])
                specs.append({**u, "dry": dry, "wet": wet})
            if len(sr_seen) > 1:
                raise HttpError(400, {"error": f"units have mixed sample rates: "
                                               f"{sorted(sr_seen)}"})

            t0 = time.time()
            result = service.predict(specs, mix, mix_sr, scenario_kind, threshold)
            seconds = time.time() - t0
            note = (f"{blob_note(stats)} kind={scenario_kind} thr={threshold} "
                    f"units={len(specs)} corrupted={len(result.get('corrupted_units', []))}")
            return result, note, seconds

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=None,
                    help="detector .pt (default: $MIXAGENT_WEIGHTS/detector/detector.pt)")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (keep it local; use an SSH tunnel for remote access)")
    ap.add_argument("--threshold-default", dest="threshold_default", type=float, default=0.5,
                    help="threshold used when a /predict request omits one")
    ap.add_argument("--latent-mode", dest="latent_mode", choices=["sample", "mode"],
                    default="sample", help="'sample' matches training (default)")
    ap.add_argument("--seed", type=int, default=0,
                    help="VAE draw seed (default 0); -1 leaves it unseeded")
    ap.add_argument("--blob-cap-gb", dest="blob_cap_gb", type=float, default=2.0,
                    help="LRU cap for the in-memory audio blob cache")
    ap.add_argument("--fake", action="store_true",
                    help="skip the model load and return a canned verdict (HTTP-layer tests)")
    args = ap.parse_args()

    service = DetectorService(args)
    store = BlobStore(cap_bytes=int(args.blob_cap_gb * 1024 ** 3))
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(service, store))
    httpd.daemon_threads = True
    print(f"[serve] afxdet fake={args.fake} on http://{args.host}:{args.port}"
          f"  (routes: /health /blobs /predict)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[serve] shutting down", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
