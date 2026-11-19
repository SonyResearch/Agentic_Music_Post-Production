#!/usr/bin/env python
"""HTTP inference server for a trained MixAgent critic (Qwen3.5-4B + RLAT + AL adapter + LoRA).

    python servers/critic_server.py --critic f3_dpo --port 8201 --device cuda:0

`--critic <id>` (f1_sft f1_dpo f2_sft f2_dpo f3_sft f3_dpo) loads the trainable tensors
from $MIXAGENT_WEIGHTS/critic/<id>.pt onto the base model described by
critic/configs/serve/<arm>.yaml ($MIXAGENT_MODELS/Qwen3.5-4B, the RLAT encoder in
$MIXAGENT_MODELS/rlat and $MIXAGENT_MODELS/stable-audio-open-1.0). The load is
verified: every adapter/LoRA tensor in the file must land and a probe tensor must change.

Routes (stdlib http.server, one global lock serialises the GPU):
  GET  /health    -> {"ok": true, "critic", "arm", "load_info", "use_dry_clip": false, ...}
  POST /blobs     {"blobs": [{"sha256", "b64"}]}  -> {"stored": [...]}   (audio uploaded once)
  POST /critique  {"instruction", "target", "mix", "ref", "max_new_tokens"}
                  each clip is {"sha256"} (optionally with inline "b64"); unknown hashes
                  -> 409 {"missing": [...]}. Reply: {"text", "cot", "answer", "gen_seconds"}.

Generation: chat template with thinking on, greedy decoding, stop at <|im_end|>, the
answer is the text after the last </think>. torch.manual_seed(0) precedes every
generation because the RLAT VAE draws a latent sample. Clips must be 44.1 kHz; they are
centre-cropped / zero-padded to 10 s like the validation data.
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
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SAMPLE_RATE = 44100
CLIP_SAMPLES = 441000          # 10 s, as in training
SPECIAL_LEFTOVERS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "<think>", "</think>",
                     "<|vision_start|>", "<|vision_end|>")
CRITIC_IDS = ("f1_sft", "f1_dpo", "f2_sft", "f2_dpo", "f3_sft", "f3_dpo")
AUDIO_FIELDS = ("target", "mix", "ref")

FAKE_COT = ("Comparing the target element against the mix and the reference: level and "
            "tone sit where the reference does. Nothing to change.")
FAKE_ANSWER = '{"tools": [], "keep": "all"}'


# ======================================================================================
# Blob protocol (same as the detector server)
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
# Audio decode — mirrors CriticismDataset._load_audio / _pick_offset / _crop (split=val)
# ======================================================================================

def decode_clip(raw, field):
    """FLAC/WAV bytes -> float32 (2, 441000) numpy array, val-style centre crop."""
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)   # (T, C)
    except Exception as exc:                                                    # noqa: BLE001
        raise HttpError(400, {"error": f"{field}: could not decode audio ({exc!r})"})
    if int(sr) != SAMPLE_RATE:
        # The training data was never resampled; resampling here would feed the encoder
        # something it never saw, so this is a client bug, not something to paper over.
        raise HttpError(400, {"error": f"{field}: sample rate {int(sr)} != {SAMPLE_RATE}"})
    a = np.ascontiguousarray(audio.T)                                           # (C, T)
    if a.shape[0] == 1:
        a = np.concatenate([a, a], axis=0)                                      # mono -> stereo
    elif a.shape[0] > 2:
        a = a[:2]
    c, t = a.shape
    if t < CLIP_SAMPLES:
        a = np.concatenate([a, np.zeros((c, CLIP_SAMPLES - t), dtype=a.dtype)], axis=1)
        offset = 0
    else:
        offset = (t - CLIP_SAMPLES) // 2
    return np.ascontiguousarray(a[:, offset:offset + CLIP_SAMPLES])


def clean_answer(answer):
    for marker in SPECIAL_LEFTOVERS:
        answer = answer.replace(marker, "")
    return answer.strip()


# ======================================================================================
# Model
# ======================================================================================

class CriticModel:
    def __init__(self, args):
        self.args = args
        self.fake = bool(args.fake)
        self.lock = threading.Lock()          # one generation at a time
        self.model = None
        self.load_info = {}
        self.weights = None
        self.config_path = str(REPO_ROOT / "critic" / "configs" / "serve" / f"{args.arm}.yaml")
        if self.fake:
            self.load_info = {"fake": True, "note": "no weights loaded"}
            return

        from critic.model.load import load_model
        from critic.paths import env_path, load_config

        self.weights = args.weights or str(env_path("MIXAGENT_WEIGHTS") / "critic" / f"{args.critic}.pt")
        if not os.path.isfile(self.weights):
            raise SystemExit(f"critic weights not found: {self.weights}")
        config = load_config(self.config_path)
        print(f"[init] loading {self.weights} with {self.config_path}", flush=True)
        t0 = time.time()
        self.model, info = load_model(self.weights, config, args.device)
        meta_id = (info.get("meta") or {}).get("id")
        if meta_id and meta_id != args.critic and not args.weights:
            raise SystemExit(f"{self.weights} declares id {meta_id!r}, expected {args.critic!r}")
        self.load_info = dict(info)
        self.load_info["load_seconds"] = round(time.time() - t0, 2)
        self.load_info["model_path"] = str(config.model.params.model_config.model_path)
        print(f"[init] ready in {self.load_info['load_seconds']}s", flush=True)

    def critique(self, instruction, clips, max_new_tokens):
        """clips = {"target", "mix", "ref"} of (2, 441000) arrays -> response dict."""
        from critic.model.load import split_thinking

        if self.fake:
            with self.lock:
                t0 = time.time()
                text = f"{FAKE_COT}\n</think>\n{FAKE_ANSWER}\n<|im_end|>"
                gen_seconds = time.time() - t0
            cot, answer = split_thinking(text, True)
            return {"text": text, "cot": cot, "answer": clean_answer(answer),
                    "gen_seconds": round(gen_seconds, 3)}

        import torch

        dev = self.args.device
        x_t = torch.from_numpy(clips["target"]).float()
        x_m = torch.from_numpy(clips["mix"]).float()
        x_r = torch.from_numpy(clips["ref"]).float()
        with self.lock:
            t0 = time.time()
            torch.manual_seed(0)  # pins the RLAT VAE latent draw; precedes EVERY call
            input_text = self.model.tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=True,
            )
            batch = {
                "input_text": [input_text],
                "audio_target": x_t[None].to(dev),
                "audio_mix": x_m[None].to(dev),
                "audio_ref": x_r[None].to(dev),
            }
            with torch.no_grad():
                texts = self.model.generation_step(
                    batch,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    skip_special_tokens=False,
                )
            gen_seconds = time.time() - t0
        text = texts[0]
        cot, answer = split_thinking(text, True)
        return {"text": text, "cot": cot, "answer": clean_answer(answer),
                "gen_seconds": round(gen_seconds, 3)}

    def health(self):
        return {
            "ok": True,
            "critic": self.args.critic,
            "arm": self.args.arm,
            "ckpt": self.weights,
            "config": self.config_path,
            "device": self.args.device,
            "load_info": self.load_info,
            "fake": self.fake,
            "sample_rate": SAMPLE_RATE,
            "clip_samples": CLIP_SAMPLES,
            "max_new_tokens_default": self.args.max_new_tokens,
            "use_dry_clip": False,
            "audio_fields": list(AUDIO_FIELDS),
        }


# ======================================================================================
# HTTP
# ======================================================================================

def make_handler(model, store):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "critic_server/1.0"

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
                    health = model.health()
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
                elif path == "/critique":
                    result, note, seconds = self._critique(payload)
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

        def _critique(self, payload):
            instruction = payload.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise HttpError(400, {"error": "'instruction' must be a non-empty string"})
            max_new_tokens = payload.get("max_new_tokens", model.args.max_new_tokens)
            try:
                max_new_tokens = int(max_new_tokens)
            except (TypeError, ValueError):
                raise HttpError(400, {"error": "'max_new_tokens' must be an int"})
            if max_new_tokens < 1:
                raise HttpError(400, {"error": "'max_new_tokens' must be >= 1"})

            stats = new_blob_stats()
            raw = {}
            fields = AUDIO_FIELDS
            for field in fields:
                if field not in payload:
                    raise HttpError(400, {"error": f"missing audio ref '{field}'"})
                raw[field] = store.resolve(payload[field], field, stats)
            raise_if_missing(stats)

            clips = {field: decode_clip(raw[field], field) for field in fields}
            result = model.critique(instruction, clips, max_new_tokens)
            note = (f"{blob_note(stats)} instr_chars={len(instruction)} "
                    f"max_new_tokens={max_new_tokens} answer_chars={len(result['answer'])}")
            return result, note, result["gen_seconds"]

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--critic", choices=CRITIC_IDS, default="f3_dpo",
                    help="released critic id; loads $MIXAGENT_WEIGHTS/critic/<id>.pt")
    ap.add_argument("--weights", default=None,
                    help="override the weight file (a released .pt or a Lightning .ckpt "
                         "of the same arm)")
    ap.add_argument("--port", type=int, default=8201)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=512,
                    help="default when a /critique request omits max_new_tokens")
    ap.add_argument("--blob-cap-gb", dest="blob_cap_gb", type=float, default=2.0,
                    help="LRU cap for the in-memory audio blob cache")
    ap.add_argument("--fake", action="store_true",
                    help="skip the model load and return a canned reply (HTTP-layer tests)")
    args = ap.parse_args()
    args.arm = args.critic.split("_")[0]

    model = CriticModel(args)
    store = BlobStore(cap_bytes=int(args.blob_cap_gb * 1024 ** 3))
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(model, store))
    httpd.daemon_threads = True
    print(f"[serve] critic {args.critic} fake={args.fake} on http://{args.host}:{args.port}"
          "  (routes: /health /blobs /critique)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[serve] shutting down", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
