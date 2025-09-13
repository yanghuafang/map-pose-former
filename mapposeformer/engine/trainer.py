"""The training loop. All of it, in one readable file.

There is no framework here. A learner reading this should be able to see every
step the optimizer takes without following a callback into a library: the loop
below is the whole of it, and the only non-obvious pieces -- warmup, gradient
clipping, bf16 autocast -- are each two lines and each carry the reason they
are there.

Single GPU. An RTX A6000 holds this model and this batch size several times
over, and distributed training would add a layer of indirection that teaches
nothing about localization.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mapposeformer.config import Config
from mapposeformer.data.synthetic import SyntheticDataset
from mapposeformer.engine.evaluator import evaluate, format_report
from mapposeformer.losses import compute_losses
from mapposeformer.model.model import MapPoseFormer


def _lr_scale(step: int, total: int, warmup: int) -> float:
    """Linear warmup, then cosine decay to a tenth of the peak."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def _autocast(device: str, mode: str):
    if mode == "off" or not device.startswith("cuda"):
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if mode == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


class Trainer:
    """Owns the run: data, model, optimizer, logging, checkpoints."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        torch.manual_seed(cfg.train.seed)
        self.device = cfg.train.device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("cuda requested but unavailable; falling back to cpu")
            self.device = "cpu"

        self.out = Path(cfg.train.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        # A run owns its directory. best.pt and last.pt are overwritten anyway,
        # so an appended metrics.jsonl was the one thing that survived a re-run
        # into the same out_dir -- and it survived as two interleaved histories
        # with no marker between them, which reads as one run that got worse.
        (self.out / "metrics.jsonl").unlink(missing_ok=True)

        self.train_set = SyntheticDataset(cfg.data, "train")
        self.val_set = SyntheticDataset(cfg.data, "val")
        self.train_loader = self._loader(self.train_set, shuffle=True)
        self.val_loader = self._loader(self.val_set, shuffle=False)

        self.model = MapPoseFormer(cfg.model).to(self.device)
        if cfg.train.compile:
            self.model = torch.compile(self.model)
        # No weight decay on norms, biases or embeddings: decaying a LayerNorm
        # gain pulls it towards zero, which is not a smaller model, only a
        # quieter one.
        decay, no_decay = [], []
        for name, prm in self.model.named_parameters():
            (no_decay if prm.ndim <= 1 or "emb" in name else decay).append(prm)
        self.opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": cfg.train.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg.train.lr,
            betas=(0.9, 0.95),
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(cfg.train.amp == "fp16" and self.device.startswith("cuda"))
        )

        steps_per_epoch = max(len(self.train_loader), 1)
        self.total_steps = cfg.train.max_steps or steps_per_epoch * cfg.train.epochs
        self.warmup = int(cfg.train.warmup_frac * self.total_steps)
        self.step = 0
        self.writer = self._writer()

    def _loader(self, dataset: SyntheticDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.train.num_workers,
            pin_memory=self.device.startswith("cuda"),
            drop_last=shuffle,
            persistent_workers=self.cfg.train.num_workers > 0,
        )

    def _writer(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("tensorboard not installed; logging to stdout and metrics.jsonl only")
            return None
        return SummaryWriter(str(self.out / "tb"))

    def _log(self, tag: str, values: dict[str, float]) -> None:
        with (self.out / "metrics.jsonl").open("a") as fh:
            fh.write(json.dumps({"step": self.step, "tag": tag, **values}) + "\n")
        if self.writer is not None:
            for k, v in values.items():
                self.writer.add_scalar(f"{tag}/{k}", v, self.step)

    def _grid(self):
        volume = getattr(self.model, "_orig_mod", self.model).volume
        return volume.cells, volume.pitch

    def train(self) -> None:
        cells, pitch = self._grid()
        cfg = self.cfg.train
        best = float("inf")
        print(
            f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M params, "
            f"{len(self.train_set)} train frames, {self.total_steps} steps, "
            f"device {self.device}"
        )
        for epoch in range(cfg.epochs):
            self.train_set.set_epoch(epoch)
            t0, seen = time.time(), 0
            for batch in self.train_loader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                with _autocast(self.device, cfg.amp):
                    out = self.model(batch)
                    total, scalars = compute_losses(out, batch, cells, pitch, self.cfg.loss)

                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(total).backward()
                self.scaler.unscale_(self.opt)
                grad = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip
                )
                scale = _lr_scale(self.step, self.total_steps, self.warmup)
                for group in self.opt.param_groups:
                    group["lr"] = cfg.lr * scale
                self.scaler.step(self.opt)
                self.scaler.update()

                self.step += 1
                seen += batch["delta"].shape[0]
                if self.step % cfg.log_every == 0:
                    scalars["grad_norm"] = float(grad)
                    scalars["lr"] = self.opt.param_groups[0]["lr"]
                    scalars["frames_per_s"] = seen / (time.time() - t0)
                    self._log("train", scalars)
                    print(
                        f"epoch {epoch} step {self.step}/{self.total_steps} "
                        + " ".join(f"{k} {v:.4f}" for k, v in scalars.items())
                    )
                if cfg.max_steps and self.step >= cfg.max_steps:
                    break

            if (epoch + 1) % cfg.eval_every == 0 or self.step >= self.total_steps:
                result = evaluate(self.model, self.val_loader, self.device)
                self._log("val", result["metrics"])  # type: ignore[arg-type]
                print(format_report(result))
                score = result["metrics"]["all/rmse_trans_m"]  # type: ignore[index]
                if score < best:
                    best = score
                    self.save("best.pt", epoch, result["metrics"])  # type: ignore[arg-type]
            self.save("last.pt", epoch, {})
            if cfg.max_steps and self.step >= cfg.max_steps:
                break

        if self.writer is not None:
            self.writer.close()
        print(f"done. best val translation RMSE {best:.4f} m -> {self.out}")

    def save(self, name: str, epoch: int, metrics: dict[str, float]) -> None:
        """Checkpoint the weights alongside the config that produced them.

        The config travels with the weights because a checkpoint whose input
        shape or grid extent is unknown cannot be loaded, only guessed at.
        """
        model = getattr(self.model, "_orig_mod", self.model)
        torch.save(
            {
                "model": model.state_dict(),
                "config": self.cfg,
                "epoch": epoch,
                "step": self.step,
                "metrics": metrics,
            },
            self.out / name,
        )
