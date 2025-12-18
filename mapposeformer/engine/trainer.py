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
from torch.utils.data import DataLoader, Dataset

from mapposeformer.config import Config, upgrade
from mapposeformer.data import build_dataset
from mapposeformer.distill import (
    check_shapes_agree,
    distill_losses,
    load_teacher,
)
from mapposeformer.engine.evaluator import evaluate, format_report
from mapposeformer.losses import compute_losses
from mapposeformer.model.attention import unpack_attention
from mapposeformer.model.model import MapPoseFormer


def _hms(seconds: float) -> str:
    """@brief Seconds as ``1h23m`` or ``4m12s``. @param seconds Duration.
    @return A short human-readable string."""
    seconds = max(0.0, seconds)
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{seconds:.0f}s"


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

        self.train_set = build_dataset(cfg.data, "train")
        self.val_set = build_dataset(cfg.data, "val")
        self.train_loader = self._loader(self.train_set, shuffle=True)
        self.val_loader = self._loader(self.val_set, shuffle=False)

        self.model = MapPoseFormer(cfg.model).to(self.device)
        if cfg.train.init_from:
            self._init_from(cfg.train.init_from)
        if cfg.train.compile:
            self.model = torch.compile(self.model)

        # The teacher runs beside the student, not before it: the synthetic
        # dataset redraws its noise every epoch, so a cached forward pass would
        # be answering about a frame the student never sees. Costs a forward
        # pass; docs/ROADMAP.md has what that is worth.
        self.teacher = None
        if cfg.distill.teacher:
            self.teacher = load_teacher(cfg.distill.teacher, self.device)
            check_shapes_agree(cfg.model, self.teacher.p)
            n = sum(p.numel() for p in self.teacher.parameters()) / 1e6
            print(f"distilling from {cfg.distill.teacher} ({n:.2f}M params)")
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
            "cuda",
            enabled=(
                cfg.train.amp == "fp16" and self.device.startswith("cuda")
            ),
        )

        steps_per_epoch = max(
            len(self.train_loader) // cfg.train.accum_steps, 1
        )
        self.total_steps = (
            cfg.train.max_steps or steps_per_epoch * cfg.train.epochs
        )
        self.warmup = int(cfg.train.warmup_frac * self.total_steps)
        self.step = 0
        self.writer = self._writer()

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
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
            print(
                "tensorboard not installed;"
                " logging to stdout and metrics.jsonl only"
            )
            return None
        return SummaryWriter(str(self.out / "tb"))

    def _log(self, tag: str, values: dict[str, float]) -> None:
        with (self.out / "metrics.jsonl").open("a") as fh:
            fh.write(
                json.dumps({"step": self.step, "tag": tag, **values}) + "\n"
            )
        if self.writer is not None:
            for k, v in values.items():
                self.writer.add_scalar(f"{tag}/{k}", v, self.step)

    def _grid(self):
        volume = getattr(self.model, "_orig_mod", self.model).volume
        return volume.cells, volume.pitch

    def _init_from(self, path: str) -> None:
        """@brief Start from an existing model's weights.

        A pruned checkpoint no longer matches the width its config implies, so
        its plan is replayed before the state dict is loaded. Without that the
        load fails on every pruned module, which is a confusing way to discover
        that a file was pruned.

        @param path Checkpoint to load.
        """
        from mapposeformer.prune import apply_plan, split_plan

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        upgrade(ckpt.get("config"))
        plan = ckpt.get("prune_plan")
        if plan:
            apply_plan(self.model, plan)
            self.prune_plan = plan
            ffn, attn = split_plan(self.model, plan)
            parts = []
            if ffn:
                parts.append(f"{len(ffn)} feed-forwards to {min(ffn.values())}")
            if attn:
                parts.append(
                    f"{len(attn)} attentions to {min(attn.values())} heads"
                )
            print(f"pruned init: {', '.join(parts)}")
        self.model.load_state_dict(unpack_attention(ckpt["model"]))
        self.model.to(self.device)
        print(f"initialised from {path}")

    def train(self) -> None:
        cells, pitch = self._grid()
        cfg = self.cfg.train
        best = float("inf")
        print(
            f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M"
            f" params, "
            f"{len(self.train_set)} train frames, {self.total_steps} steps, "
            f"device {self.device}"
        )
        started, start_step = time.time(), self.step
        for epoch in range(cfg.epochs):
            self.train_set.set_epoch(epoch)
            t0, seen = time.time(), 0
            for micro, batch in enumerate(self.train_loader):
                batch = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in batch.items()
                }
                with _autocast(self.device, cfg.amp):
                    out = self.model(batch)
                    total, scalars = compute_losses(
                        out, batch, cells, pitch, self.cfg.loss
                    )
                    if self.teacher is not None:
                        with torch.no_grad():
                            ref = self.teacher(batch)
                        kd, kd_scalars = distill_losses(
                            out, ref, self.cfg.distill
                        )
                        total = total + kd
                        scalars.update(kd_scalars)
                        scalars["loss"] = float(total.detach())

                # Divided by the accumulation count, because the gradients of
                # the micro-batches are summed and the objective is their mean.
                self.scaler.scale(total / cfg.accum_steps).backward()
                seen += batch["delta"].shape[0]
                if (micro + 1) % cfg.accum_steps:
                    continue  # keep accumulating; do not step or clip yet

                self.scaler.unscale_(self.opt)
                grad = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip
                )
                scale = _lr_scale(self.step, self.total_steps, self.warmup)
                for group in self.opt.param_groups:
                    group["lr"] = cfg.lr * scale
                self.scaler.step(self.opt)
                self.scaler.update()
                # Zeroed *after* the step, not before the backward, so the
                # accumulated gradients survive until they have been applied.
                self.opt.zero_grad(set_to_none=True)

                self.step += 1
                if self.step % cfg.log_every == 0:
                    scalars["grad_norm"] = float(grad)
                    scalars["lr"] = self.opt.param_groups[0]["lr"]
                    scalars["frames_per_s"] = seen / (time.time() - t0)
                    # How long is left, from the rate so far. A training run is
                    # the one thing here that takes hours, and a reader should
                    # not have to compute this from a step count and a
                    # benchmark they have to go and find.
                    done = self.step - start_step
                    rate = done / max(time.time() - started, 1e-9)
                    left = (self.total_steps - self.step) / max(rate, 1e-9)
                    scalars["eta_min"] = left / 60.0
                    self._log("train", scalars)
                    print(
                        f"epoch {epoch} step {self.step}/{self.total_steps} "
                        + " ".join(
                            f"{k} {v:.4f}"
                            for k, v in scalars.items()
                            if k != "eta_min"
                        )
                        + f" | {_hms(time.time() - started)} elapsed,"
                        f" {_hms(left)} left"
                    )
                if cfg.max_steps and self.step >= cfg.max_steps:
                    break

            if (
                epoch + 1
            ) % cfg.eval_every == 0 or self.step >= self.total_steps:
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
                "prune_plan": getattr(self, "prune_plan", None),
                "config": self.cfg,
                "epoch": epoch,
                "step": self.step,
                "metrics": metrics,
            },
            self.out / name,
        )
