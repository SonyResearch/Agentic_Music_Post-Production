"""Train the critic: stage-1 / stage-2 SFT or DPO, selected by the config.

    python critic/train.py --config critic/configs/sft/stage1_f3.yaml --gpu 0,1
    python critic/train.py --config critic/configs/sft/stage2_f3.yaml --gpu 0,1,2,3
    python critic/train.py --config critic/configs/dpo/dpo_f3.yaml   --gpu 0,1,2,3
    python critic/train.py --config ... --gpu ... --resume <expdir>/last.ckpt

The config names the Lightning module (`model.target`: LMTrainingWrapper for SFT,
DPOTrainingWrapper for DPO) and the data module (`data.target`). DDP, bf16-mixed, seed
42 + rank; the best checkpoint by val_loss and last.ckpt go to model_config.expdir.
For DPO, the reference log-probs of the pairs are computed first (once) and written into
the pair files.
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # noqa: E402
from pytorch_lightning.strategies import DDPStrategy  # noqa: E402

from critic.model.dpo_trainer import DPOTrainingWrapper, fill_reference_logprobs  # noqa: E402
from critic.model.utils import instantiate_from_config, set_activation_checkpointing  # noqa: E402
from critic.paths import load_config  # noqa: E402

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_decoder_layer_cls(model_type):
    """Decoder-layer class, needed only for activation checkpointing."""
    if model_type in ("qwen3.5", "qwen3_5"):
        for cls_name in ("Qwen3_5DecoderLayer", "Qwen35DecoderLayer"):
            try:
                mod = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
                return getattr(mod, cls_name)
            except (ImportError, AttributeError):
                continue
    raise ValueError(f"No decoder-layer class known for model_type={model_type!r}")


def main(args):
    config = load_config(args.config)
    run_logdir = os.path.join(config.lightning.logdir, config.project, config.lightning.version_name)
    os.makedirs(run_logdir, exist_ok=True)
    os.makedirs(config.model.params.model_config.expdir, exist_ok=True)
    OmegaConf.save(config, os.path.join(run_logdir, "config.yaml"))  # resolved paths

    rank = int(os.environ.get("RANK", 0))
    precision = config.lightning.trainer.precision
    if precision in ("16", "16-mixed"):
        target_dtype = torch.float16
    elif precision in ("bf16", "bf16-mixed"):
        target_dtype = torch.bfloat16
    elif precision in ("32", "32-true"):
        target_dtype = torch.float32
    else:
        raise ValueError(f"Precision {precision} not supported")

    pl.seed_everything(42 + rank)
    data_module = instantiate_from_config(config.data)
    data_module.prepare_data()
    model = instantiate_from_config(config.model)
    model.to(target_dtype)
    if isinstance(model, DPOTrainingWrapper) and os.environ.get("LOCAL_RANK", "0") == "0":
        # the reference is the SFT checkpoint the policy starts from; DDP workers are
        # launched later and find the log-probs already in the pair files
        fill_reference_logprobs(model, config.data.params)

    trainer_config = dict(config.lightning.trainer)
    if trainer_config.pop("strategy") != "ddp":
        raise ValueError("only strategy: ddp is supported")
    strategy = DDPStrategy(find_unused_parameters=False)
    if config.activation_checkpointing:
        model.lm.config.use_cache = False
        set_activation_checkpointing(model, get_decoder_layer_cls(config.model_type))

    logger = pl.loggers.TensorBoardLogger(
        save_dir=config.lightning.logdir, name=config.project,
        version=config.lightning.version_name)
    print(f"Model dtype: {next(model.parameters()).dtype}")

    callbacks = [LearningRateMonitor(logging_interval="step")]
    if config.lightning.get("save_checkpoint", True):
        callbacks.append(ModelCheckpoint(
            dirpath=config.model.params.model_config.expdir,
            filename="step{step}-val{val_loss:.4f}",
            auto_insert_metric_name=False,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ))

    trainer = pl.Trainer(callbacks=callbacks, strategy=strategy, devices="auto",
                         logger=logger, **trainer_config)
    trainer.fit(model, data_module, ckpt_path=args.resume or None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="training YAML")
    parser.add_argument("--gpu", default=None, help="CUDA device ids, e.g. 0,1,2,3")
    parser.add_argument("--resume", default=None, help="Lightning checkpoint to resume from")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["NCCL_P2P_DISABLE"] = "1"
    main(args)
