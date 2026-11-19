"""Client for the Qwen3-Omni planner served by vLLM (OpenAI-compatible API).

Every call sends audio clips (inlined as base64 data URLs) plus one text prompt, with
thinking enabled. The context window is split per call: the exact prompt length is
counted locally with the Qwen3-Omni processor, ``answer_reserve`` tokens are kept for
the final answer and the rest is the thinking budget (honored by vLLM builds that
support ``thinking_token_budget``).

Sampling: temperature 0.6, top_p 0.95, top_k 20 (the paper's setting).
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

END_THINK = "</think>"
SPECIAL_TOKENS_TO_STRIP = ("<|im_end|>", "<|endoftext|>")
_AUDIO_MIME = {".wav": "wav", ".flac": "flac", ".mp3": "mpeg", ".ogg": "ogg",
               ".m4a": "mp4", ".webm": "webm"}


@dataclass
class SamplingConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20


@dataclass
class BudgetConfig:
    max_model_len: int = 65536
    answer_reserve: int = 4096
    safety_margin: int = 512


@dataclass
class GenerationResult:
    answer: str
    reasoning: str | None
    prompt_tokens: int
    max_tokens: int


def compute_budget(prompt_tokens: int, cfg: BudgetConfig) -> tuple[int, int]:
    """-> (thinking_token_budget, max_tokens) for one generation."""
    if cfg.max_model_len <= prompt_tokens:
        raise ValueError(f"prompt ({prompt_tokens} tokens) fills the context window "
                         f"({cfg.max_model_len})")
    available = cfg.max_model_len - prompt_tokens - cfg.safety_margin
    if available <= cfg.answer_reserve:
        raise ValueError(f"prompt too large: only {available} new tokens remain, "
                         f"<= answer_reserve {cfg.answer_reserve}")
    thinking = available - cfg.answer_reserve
    return thinking, min(thinking + cfg.answer_reserve, available)


def clean_model_output(content: str | None) -> str:
    """The final visible answer: text after the last ``</think>``, special tokens removed."""
    if not content:
        return ""
    text = content
    if END_THINK in text:
        text = text.rsplit(END_THINK, 1)[-1]
    for token in SPECIAL_TOKENS_TO_STRIP:
        text = text.replace(token, "")
    return text.strip()


def _audio_data_url(path: str) -> str:
    p = Path(path)
    subtype = _AUDIO_MIME.get(p.suffix.lower(), p.suffix.lower().lstrip(".") or "wav")
    return f"data:audio/{subtype};base64,{base64.b64encode(p.read_bytes()).decode('ascii')}"


def build_messages(conversation: list[dict]) -> list[dict]:
    messages = []
    for msg in conversation:
        parts = []
        for c in msg["content"]:
            if c.get("type") == "audio":
                parts.append({"type": "audio_url",
                              "audio_url": {"url": _audio_data_url(c["audio"])}})
            else:
                parts.append({"type": "text", "text": c.get("text", "")})
        messages.append({"role": msg.get("role", "user"), "content": parts})
    return messages


def conversation(audio_paths: list[Path], text: str) -> list[dict]:
    content = [{"type": "audio", "audio": str(Path(p).resolve())} for p in audio_paths]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


class PromptTokenCounter:
    """Exact prompt-token count on CPU with the Qwen3-Omni processor (no GPU model)."""

    def __init__(self, model_path: str):
        from transformers import Qwen3OmniMoeProcessor

        self.model_path = model_path
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        self._cache: dict[str, int] = {}

    def _key(self, conv: list[dict]) -> str:
        parts = [self.model_path]
        for msg in conv:
            for c in msg["content"]:
                if c.get("type") == "audio":
                    st = Path(c["audio"]).stat()
                    parts.append(f"a:{c['audio']}:{st.st_size}:{int(st.st_mtime)}")
                else:
                    parts.append("t:" + c.get("text", ""))
        return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()

    def count(self, conv: list[dict]) -> int:
        key = self._key(conv)
        if key in self._cache:
            return self._cache[key]
        from qwen_omni_utils import process_mm_info

        text = self.processor.apply_chat_template(conv, add_generation_prompt=True,
                                                  tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos,
                                return_tensors="pt", padding=True, use_audio_in_video=False)
        if "attention_mask" in inputs:
            n = int(inputs["attention_mask"][0].sum().item())
        else:
            n = int(inputs["input_ids"].shape[-1])
        self._cache[key] = n
        return n


class VLLMBackend:
    """One OpenAI-compatible endpoint. Not thread-safe (one per worker)."""

    def __init__(self, base_url: str, served_model: str, tokenizer_path: str,
                 sampling: SamplingConfig | None = None, budget: BudgetConfig | None = None):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.base_url = base_url
        self.model = served_model
        self.sampling = sampling or SamplingConfig()
        self.budget = budget or BudgetConfig()
        self.counter = PromptTokenCounter(tokenizer_path)

    def generate(self, conv: list[dict]) -> GenerationResult:
        prompt_tokens = self.counter.count(conv)
        thinking, max_tokens = compute_budget(prompt_tokens, self.budget)
        n_audio = sum(1 for m in conv for c in m["content"] if c.get("type") == "audio")
        print(f"[llm] {self.base_url} prompt_tokens={prompt_tokens} audio={n_audio} "
              f"thinking_budget={thinking} max_tokens={max_tokens}", file=sys.stderr)
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=build_messages(conv), max_tokens=max_tokens,
                temperature=self.sampling.temperature, top_p=self.sampling.top_p,
                modalities=["text"],
                extra_body={"top_k": self.sampling.top_k,
                            "thinking_token_budget": thinking,
                            "chat_template_kwargs": {"enable_thinking": True}})
        except Exception as exc:
            raise RuntimeError(f"vLLM request to {self.base_url} failed: {exc}") from exc
        msg = resp.choices[0].message
        answer = clean_model_output(getattr(msg, "content", None) or "")
        reasoning = getattr(msg, "reasoning_content", None)
        if not answer and reasoning:
            print("WARNING: model produced reasoning but no final answer", file=sys.stderr)
        return GenerationResult(answer, reasoning, prompt_tokens, max_tokens)


def write_trace(trace_dir: Path, tag: str, audio: list[Path], text: str,
                result: GenerationResult) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / f"{tag}.prompt.txt").write_text(
        "\n".join(f"[audio {i + 1}] {p}" for i, p in enumerate(audio))
        + ("\n\n" if audio else "") + text)
    (trace_dir / f"{tag}.answer.txt").write_text(result.answer or "")
    if result.reasoning:
        (trace_dir / f"{tag}.thinking.txt").write_text(result.reasoning)


# --------------------------------------------------------------------------- JSON replies

def _repair_truncated_json(text: str) -> str:
    """Append closers for unmatched braces/brackets (outside strings)."""
    stack: list[str] = []
    in_str = esc = False
    for ch in text:
        if esc:
            esc = False
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    if in_str:
        text += '"'
    return text + "".join("}" if c == "{" else "]" for c in reversed(stack))


def parse_json_reply(reply: str) -> dict:
    """Last ```json fenced block (or the whole reply) -> object, with light repair."""
    fences = re.findall(r"```json\s*(.*?)```", reply, flags=re.DOTALL)
    text = (fences[-1] if fences else reply).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("no JSON object found", text, 0)
    end = text.rfind("}")
    if end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return json.loads(_repair_truncated_json(text[start:]))


class AgentError(RuntimeError):
    pass


class QwenAgent:
    """One backend + tracing + retries (used by Stage 2)."""

    RETRY_WAITS = (0, 10, 30, 90, 180)

    def __init__(self, backend: VLLMBackend, log=print):
        self.backend = backend
        self.log = log

    def ask(self, sid: str, audio: list[Path], text: str, trace_dir: Path, tag: str) -> str:
        last = None
        for attempt, wait in enumerate(self.RETRY_WAITS):
            if wait:
                time.sleep(wait)
            try:
                result = self.backend.generate(conversation(audio, text))
                write_trace(trace_dir, tag, audio, text, result)
                return result.answer or ""
            except Exception as e:  # noqa: BLE001
                last = e
                self.log(sid, f"{tag}: call failed (attempt {attempt + 1}): "
                              f"{type(e).__name__}: {e}")
        raise AgentError(f"{tag}: all retries failed: {last}")

    def ask_json(self, sid: str, audio: list[Path], text: str, trace_dir: Path, tag: str,
                 validate) -> tuple[object | None, list[str]]:
        """One call + one repair retry. ``validate(parsed) -> (value, errors)``."""
        all_errs: list[str] = []
        prompt = text
        for round_no in (1, 2):
            rtag = tag if round_no == 1 else f"{tag}.retry"
            try:
                reply = self.ask(sid, audio, prompt, trace_dir, rtag)
            except AgentError as e:
                all_errs.append(str(e))
                return None, all_errs
            try:
                parsed = parse_json_reply(reply)
            except Exception as e:  # noqa: BLE001
                errs = [f"reply is not valid JSON: {e}"]
            else:
                value, errs = validate(parsed)
                if value is not None:
                    if errs:
                        all_errs.extend(errs)
                    return value, all_errs
            all_errs.extend(errs)
            prompt = (text + "\n\nYour previous reply could not be used. Problems:\n- "
                      + "\n- ".join(str(e)[:300] for e in errs[:8])
                      + "\nRe-output ONLY the corrected JSON in a single ```json``` "
                        "fenced block, same schema.")
        return None, all_errs
