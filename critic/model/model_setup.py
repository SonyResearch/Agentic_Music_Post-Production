"""Tokenizer / base-LM construction for the critic."""
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_tokenizer(model_path, cache_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    return tokenizer


def setup_llm(model_path, attn_implementation, cache_dir, dtype):
    tokenizer = setup_tokenizer(model_path, cache_dir)
    lm = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    return lm, tokenizer
