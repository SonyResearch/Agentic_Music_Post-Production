#!/usr/bin/env bash
# Download the third-party models MixAgent needs into $MIXAGENT_MODELS (default <repo>/models).
#
#   bash setup/download_models.sh            # everything (~83 GB)
#   bash setup/download_models.sh critic     # only what one component needs:
#                                            #   planner | critic | detector | eval | captioner
#
#   planner  : Qwen3-Omni-30B-A3B-Thinking (served by vLLM)              ~64 GB
#   critic   : Qwen3.5-4B + RLAT + Stable Audio Open 1.0 VAE (for RLAT)  ~10 GB
#   detector : RLAT + Stable Audio Open 1.0 VAE + MusicFM 25 Hz (MSD)     ~2.2 GB
#   captioner: Nemotron-Labs-Audex-30B-A3B, only to caption the stems of YOUR OWN songs
#              (caption_stems.py; the paper's clips ship their captions). NOT in "all". ~61 GB
#   eval     : scoring models for evaluation/encoders.py: Fx-Encoder++ + LAION-CLAP
#              (630k-audioset), TuneJury head + LAION-CLAP-Music, MERT-v1-330M  ~7 GB
#              (plus RLAT and the Stable Audio Open VAE above for RLAT metrics)
#              Licences: TuneJury head and MERT-v1-330M are CC BY-NC 4.0 (non-commercial
#              use only); LAION-CLAP checkpoints are CC0; Fx-Encoder++ has no licence
#              declared on its Hugging Face repo.
#              After this, export TUNEJURY_CKPT=$MIXAGENT_MODELS/tunejury/tunejury.pt and
#              TUNEJURY_ROOT=<clone of github.com/yonghyunk1m/TuneJury> (the CLAP-Music
#              checkpoint is picked up from next to tunejury.pt). MERT goes into the
#              standard Hugging Face cache (TuneJury loads it by name, m-a-p/MERT-v1-330M).
#
# Stable Audio Open 1.0 is gated: accept the Stability AI Community License at
# https://huggingface.co/stabilityai/stable-audio-open-1.0 and log in first
# (`hf auth login`, or `huggingface-cli login`, or export HF_TOKEN).
# Only its vae/ subfolder is downloaded; RLAT needs nothing else.
# RLAT (arXiv:2608.28127) comes from shlee-97/rlat: config.json + model.safetensors.
#
# MixAgent's own weights (critic LoRA/projector, detector) are
# distributed separately and go in $MIXAGENT_WEIGHTS (default <repo>/weights).
#
# Revisions are pinned to the snapshots the paper's results were produced with.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="${MIXAGENT_MODELS:-$REPO/models}"
WHAT="${1:-all}"

if command -v hf >/dev/null 2>&1; then
  HF=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=(huggingface-cli download)
else
  echo "Neither 'hf' nor 'huggingface-cli' found: pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

fetch() {  # fetch <repo_id> <revision> <dest_dir> [file or --include pattern ...]
  local repo="$1" rev="$2" dest="$3"; shift 3
  echo "==> $repo @ ${rev:0:8} -> $dest"
  mkdir -p "$dest"
  "${HF[@]}" "$repo" "$@" --revision "$rev" --local-dir "$dest"
}

planner() {
  fetch Qwen/Qwen3-Omni-30B-A3B-Thinking 2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b \
        "$MODELS/Qwen3-Omni-30B-A3B-Thinking"
}

qwen35() {
  fetch Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a "$MODELS/Qwen3.5-4B"
}

sao_vae() {
  fetch stabilityai/stable-audio-open-1.0 f21265c1e2710b3bd2386596943f0007f55f802e \
        "$MODELS/stable-audio-open-1.0" \
        vae/config.json vae/diffusion_pytorch_model.safetensors
}

rlat() {
  fetch shlee-97/rlat cbca1821a071d22e74fd8de44c6662bc27997dc7 "$MODELS/rlat" \
        config.json model.safetensors
}

musicfm() {
  fetch minzwon/MusicFM 4513b38bc25ad1d227b1980819b9691ba97f4d87 "$MODELS/musicfm" \
        pretrained_msd.pt msd_stats.json
}

fxencpp() {
  fetch yytung/fxencoder-plusplus d6ed786a5fa20126402121ba8e28e1a6427eea52 \
        "$MODELS/fxencoder_plusplus" fxenc_plusplus_default.pt
  # LAION-CLAP 630k-audioset (HTSAT-tiny + roberta), the audio tower inside Fx-Encoder++
  fetch lukewys/laion_clap b3708341862f581175dba5c356a4ebf74a9b6651 \
        "$MODELS/fxencoder_plusplus" 630k-audioset-best.pt
  # laion_clap names it 630k-audioset-best.pt; encoders.py expects clap-630k-audioset-best.pt
  mv -f "$MODELS/fxencoder_plusplus/630k-audioset-best.pt" \
        "$MODELS/fxencoder_plusplus/clap-630k-audioset-best.pt"
}

tunejury() {
  # head (CC BY-NC 4.0) and the LAION-CLAP-Music checkpoint TuneJury loads next to it (CC0)
  fetch TuneJury/tunejury a36a6adaad4a9786a12c3e5fd82501f2f2dc54ff "$MODELS/tunejury" tunejury.pt
  fetch lukewys/laion_clap b3708341862f581175dba5c356a4ebf74a9b6651 "$MODELS/tunejury" \
        music_audioset_epoch_15_esc_90.14.pt
  # MERT-v1-330M (CC BY-NC 4.0) -> default HF cache; skip the 4 GB fairseq checkpoint
  echo "==> m-a-p/MERT-v1-330M @ 5240c270 -> Hugging Face cache"
  "${HF[@]}" m-a-p/MERT-v1-330M --revision 5240c2708a5acaee1007f43fb9735c7dcd0b78c9 \
        --exclude "MERT-v1-330M_fairseq.pt" >/dev/null
}

captioner() {
  # only the HF inference helpers + the full checkpoint (the repo also holds TTS/audio-gen parts)
  fetch nvidia/Nemotron-Labs-Audex-30B-A3B 00f0afa02e8ec0a9afc88221e456a02591bfca4c \
        "$MODELS/Nemotron-Labs-Audex-30B-A3B" \
        --include "inference_scripts_hf/*" "checkpoint_folder_full/*"
}

verify() {  # sha256 of the files whose exact bytes the released weights were trained on
  echo "==> verifying checksums"
  local ok=1
  check() {
    [ -f "$1" ] || return 0
    local got; got="$(sha256sum "$1" | cut -d' ' -f1)"
    if [ "$got" = "$2" ]; then echo "  ok   $1"; else echo "  BAD  $1 ($got)"; ok=0; fi
  }
  check "$MODELS/stable-audio-open-1.0/vae/diffusion_pytorch_model.safetensors" \
        2131cdb52020b2473707465449d8bdb4f6cca61c93150a947baca02bc58ffd7b
  check "$MODELS/rlat/model.safetensors" \
        d0abdda9a4cbddaee1bd8d1f32bbc579fe3c18d455e738bb589b2a93bffe64a7
  check "$MODELS/musicfm/pretrained_msd.pt" \
        218b483a0256ddef736267425fabb166fd97008983696bb9270def464b47bded
  check "$MODELS/musicfm/msd_stats.json" \
        c36c61ab10ca4d2e7fdfefc3fcc15205316bec276a06a47baa3641a62c546f22
  check "$MODELS/fxencoder_plusplus/fxenc_plusplus_default.pt" \
        9291f51c4beb6a3bfd1c50aafeab3e3cbcc5d20f9a87234f2205f8156177c14c
  check "$MODELS/fxencoder_plusplus/clap-630k-audioset-best.pt" \
        8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037
  check "$MODELS/tunejury/tunejury.pt" \
        e95c058491ab61616989ecf19eece0734da6425636c3e76e1c352e9f237f5d44
  check "$MODELS/tunejury/music_audioset_epoch_15_esc_90.14.pt" \
        fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd
  [ "$ok" = 1 ] || { echo "checksum mismatch" >&2; exit 1; }
}

case "$WHAT" in
  all)      planner; qwen35; sao_vae; rlat; musicfm; fxencpp; tunejury ;;
  planner)  planner ;;
  critic)   qwen35; sao_vae; rlat ;;
  detector) sao_vae; rlat; musicfm ;;
  eval)     sao_vae; rlat; fxencpp; tunejury ;;
  captioner) captioner ;;
  *) echo "usage: $0 [all|planner|critic|detector|eval|captioner]" >&2; exit 2 ;;
esac
verify
echo "done: models in $MODELS"
