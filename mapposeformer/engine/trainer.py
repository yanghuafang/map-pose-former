"""The training loop, and the two schedules it runs.

Learning rate warms up and decays, which is ordinary. The second schedule is
not: the robust residual scale has to *start wide and narrow*, because
Geman-McClure rejects whatever does not fit the current pose and at step zero
that is everything. Annealing it is graduated non-convexity -- solve an easy,
nearly-convex problem first and deform it into the hard one -- and without it
the model spends its early steps fitting whichever correspondences happened to
land close.

Validation reports every metric a comparison needs: long, lat and yaw error,
recall at the gate, the do-nothing baseline, and the calibration the covariance
is judged on. One translation number cannot settle the question a run was
launched to answer, and the omission is invisible until the run is over.
"""

from __future__ import annotations

import json
import math
import shlex
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from mapposeformer.config import Config, dump_config, upgrade
from mapposeformer.data import build_dataset
from mapposeformer.engine.evaluator import evaluate as evaluator_evaluate
from mapposeformer.losses import compute_losses
from mapposeformer.model import MapPoseFormer


def _loader(cfg: Config, split: str, shuffle: bool) -> DataLoader:
    # Augmentation and persistent workers cannot coexist. A persistent worker
    # holds its own copy of the dataset for the life of the run, so
    # `set_epoch` on the parent never reaches it and the reseeding silently
    # does nothing.
    augmenting = (
        split == "train" and cfg.data.augment and not cfg.data.cache_dir
    )
    return DataLoader(
        build_dataset(cfg.data, split),
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.device.startswith("cuda"),
        drop_last=shuffle,
        persistent_workers=cfg.train.num_workers > 0 and not augmenting,
    )


def _autocast(cfg: Config):
    kind = cfg.train.amp
    if kind == "off" or not cfg.train.device.startswith("cuda"):
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if kind == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: str
) -> dict[str, float]:
    """Every metric the evaluator computes, flattened for the metrics log.

    Logging translation RMSE alone wastes the run. Runs have to be compared on
    `long` and on recall separately: translation RMSE folds an along-track
    error the lane geometry barely constrains into a lateral one it constrains
    well, so a run that logs only `trans` cannot answer the question it was
    launched to settle, and the omission is invisible until the run is over.
    `evaluator.evaluate` computes all of it already for `tools/eval.py`, so the
    trainer calls it.

    The accumulators are streaming and the loader is the bottleneck, so the
    extra metrics cost one pass' arithmetic and no extra pass.

    @return Flat ``{name: float}``. ``trans_rmse`` and ``prior_rmse`` keep
        their names, because checkpoint selection and every existing log reads
        them.
    """
    result = evaluator_evaluate(model, loader, device)
    model.train()

    flat: dict[str, float] = {}
    for key, summary in result.items():
        if key == "calibration":
            flat.update({f"calib/{k}": v for k, v in summary.as_dict().items()})
        else:
            flat.update({f"{key}/{k}": v for k, v in summary.as_dict().items()})

    # The two names the rest of the code already depends on.
    flat["trans_rmse"] = flat["all/rmse_trans_m"]
    flat["prior_rmse"] = flat["nothing/rmse_trans_m"]
    return flat


class Trainer:
    """@param cfg The whole configuration. The run owns ``train.out_dir``."""

    def __init__(self, cfg: Config) -> None:
        torch.manual_seed(cfg.train.seed)
        self.cfg = cfg
        self.device = cfg.train.device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"

        self.out = Path(cfg.train.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        # best.pt and last.pt are overwritten, so an appended metrics.jsonl was
        # the one thing that survived a re-run and quietly interleaved two.
        # A resume is the one case where appending is right: it is the same
        # history continuing, not a second one.
        if not cfg.train.resume:
            (self.out / "metrics.jsonl").unlink(missing_ok=True)

        # Every flag this run was launched with, beside its checkpoints. A run
        # launched from an inline command line leaves no other record, and
        # a result whose configuration cannot be recovered is not a
        # measurement -- it happened once and cannot be argued with. Written
        # before the first step so it exists even for a run that crashes.
        (self.out / "config.yaml").write_text(dump_config(cfg))
        (self.out / "cmd.txt").write_text(
            " ".join(shlex.quote(a) for a in sys.argv) + "\n"
        )
        # Which silicon, because the answer changes the arithmetic. bf16 on
        # Turing (sm_75) is emulated -- `is_bf16_supported()` returns True and
        # a matmul runs, but at 4.0 TFLOP/s against fp32's 6.9, and the values
        # are rounded to 8 mantissa bits either way. A run on sm_75 and one on
        # sm_86 are comparable in kind and not bit-identical, so which
        # produced a checkpoint is recorded rather than inferred.
        if self.device.startswith("cuda") and torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(torch.cuda.current_device())
            (self.out / "device.txt").write_text(
                f"{prop.name}\nsm_{prop.major}{prop.minor}\n"
                f"bf16_supported={torch.cuda.is_bf16_supported()}\n"
                f"amp={cfg.train.amp}\n"
            )

        self.train_loader = _loader(cfg, "train", shuffle=True)
        self.val_loader = _loader(cfg, "val", shuffle=False)

        self.model = MapPoseFormer(cfg.model).to(self.device)
        #: Set by :meth:`_init_from` when the starting weights were pruned, and
        #: written into every checkpoint so the result is loadable on its own.
        self.prune_plan: dict[str, int] | None = None
        if cfg.train.init_from:
            self._init_from(cfg.train.init_from)

        # The teacher runs *beside* the student rather than being cached ahead
        # of it. With augmentation on, the dataset redraws its noise every
        # epoch, so a stored forward pass would be answering about a frame the
        # student never sees. It costs one extra forward and no gradient.
        self.teacher = None
        if cfg.distill.teacher:
            import torch as _t

            from mapposeformer.distill import check_shapes_agree, load_teacher

            _blob = _t.load(
                cfg.distill.teacher, map_location="cpu", weights_only=False
            )
            self._teacher_sample = upgrade(_blob["config"]).data.sample
            self.teacher = load_teacher(cfg.distill.teacher, self.device)
            # Sample geometry, not model shape: the assignment's dimensions
            # come from how many map and detection elements a sample packs,
            # and the teacher is *meant* to differ in dim and layers.
            check_shapes_agree(cfg.data.sample, self._teacher_sample)
            n = sum(p.numel() for p in self.teacher.parameters()) / 1e6
            print(
                f"distilling from {cfg.distill.teacher} ({n:.2f}M params)",
                flush=True,
            )
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        self.total_steps = cfg.train.max_steps or (
            cfg.train.epochs * len(self.train_loader)
        )
        self.step = 0

    def lr_at(self, step: int) -> float:
        """Linear warmup, then cosine decay to a tenth."""
        warmup = max(1, int(self.cfg.train.warmup_frac * self.total_steps))
        if step < warmup:
            return self.cfg.train.lr * (step + 1) / warmup
        t = (step - warmup) / max(1, self.total_steps - warmup)
        return self.cfg.train.lr * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))
        )

    def sigma_at(self, step: int) -> float:
        """The robust scale, annealed wide to narrow.

        Geometric rather than linear: the scale spans 6 m to 1 m and what
        matters is the ratio to the current residual, not the difference.
        """
        p = self.cfg.model
        frac = min(
            1.0, step / max(1, int(self.cfg.train.gnc_frac * self.total_steps))
        )
        ratio = p.robust_sigma_m / p.robust_sigma_start_m
        return p.robust_sigma_start_m * ratio**frac

    def log(self, tag: str, values: dict[str, float]) -> None:
        """One JSON line per record, appended.

        The console print is for watching; this is for plotting afterwards,
        and it is the only output that survives a re-run overwriting the
        checkpoints.
        """
        with (self.out / "metrics.jsonl").open("a") as fh:
            fh.write(
                json.dumps({"step": self.step, "tag": tag, **values}) + "\n"
            )

    def _init_from(self, path: str) -> None:
        """@brief Start from an existing model's weights, optimizer excluded.

        This is fine-tuning a model that has been *changed* -- pruned, almost
        always -- rather than resuming a run that stopped. Resuming wants the
        optimizer state and the schedule position; this wants neither, because
        the architecture underneath them is no longer the one they describe.

        A pruned checkpoint no longer matches the width its config implies, so
        its plan is replayed against the fresh model before the state dict is
        loaded. Without that the load fails on every pruned module, which is a
        confusing way to find out that a file was pruned.

        @param path Checkpoint to take weights from.
        """
        from mapposeformer.prune import apply_plan, split_plan

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
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
            print(f"pruned init: {', '.join(parts)}", flush=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        print(f"initialised from {path}", flush=True)

    def save(self, name: str, epoch: int, metrics: dict[str, float]) -> None:
        """Weights beside the config that produced them.

        A checkpoint whose configuration is unknown cannot be loaded, only
        guessed at.

        ``last.pt`` additionally carries everything needed to *continue*: the
        optimiser's moments, the step counter the schedules are functions of,
        the best score so far, and the RNG state. Without those a resumed run
        restarts Adam from zero moments and re-enters the learning-rate
        schedule at the wrong place, which is a different experiment wearing
        the same name. ``best.pt`` does not need them -- nothing resumes from
        it -- and they would double its size for no reader.
        """
        blob = {
            "model": self.model.state_dict(),
            "config": self.cfg,
            "epoch": epoch,
            "step": self.step,
            "metrics": metrics,
        }
        # Carried forward, or a fine-tuned pruned model is unloadable: its
        # config says one width and its tensors are another, and only the plan
        # reconciles them.
        if self.prune_plan:
            blob["prune_plan"] = self.prune_plan
        if name == "last.pt":
            blob.update(
                {
                    # `epoch` is the 0-based loop variable, so after two
                    # epochs it reads 1. Resuming from that would replay an
                    # epoch, which is why the count to continue from is stored
                    # separately and named for what it is.
                    "next_epoch": epoch + 1,
                    "optimizer": self.opt.state_dict(),
                    "best": self.best,
                    "reported": self.reported,
                    # Without these a resumed run restarts its early-stop
                    # counter, so it is stopped by a different rule than an
                    # uninterrupted one -- and a resume can happen unattended,
                    # so the difference would be invisible and unattributable.
                    "stale": self.stale,
                    "best_epoch": self.best_epoch,
                    "clipped": self.clipped,
                    "steps_seen": self.steps_seen,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                }
            )
        torch.save(blob, self.out / name)

    def resume(self) -> int:
        """Continue from ``last.pt`` if it is there.

        @return The epoch to start from; 0 when there is nothing to resume.

        Unattended runs are why this exists. Days of training on a shared
        machine will meet an OOM, a reboot or a stray kill, and without this
        each one costs every epoch it had already paid for.
        """
        path = self.out / "last.pt"
        if not self.cfg.train.resume or not path.exists():
            return 0
        blob = torch.load(path, map_location=self.device, weights_only=False)
        if "optimizer" not in blob:
            print(f"  {path} carries no optimiser state; starting from scratch")
            return 0
        self.model.load_state_dict(blob["model"])
        self.opt.load_state_dict(blob["optimizer"])
        self.step = int(blob["step"])
        self.best = float(blob.get("best", float("inf")))
        self.reported = float(blob.get("reported", float("inf")))
        self.clipped = int(blob.get("clipped", 0))
        self.steps_seen = int(blob.get("steps_seen", 0))
        self.stale = int(blob.get("stale", 0))
        self.best_epoch = int(blob.get("best_epoch", -1))
        if blob.get("rng") is not None:
            torch.set_rng_state(blob["rng"].cpu())
        if blob.get("cuda_rng") is not None and torch.cuda.is_available():
            # `.cpu()` for the same reason as the line above, and it is easy to
            # miss here because the state is a *list* of tensors rather than
            # one: `map_location` above put the whole blob on the device, and
            # `set_rng_state` wants a CPU ByteTensor. Without this, resuming
            # dies with "RNG state must be a torch.ByteTensor" -- after the
            # weights and the optimizer have already loaded, so the failure
            # looks like a corrupt checkpoint rather than a type error.
            torch.cuda.set_rng_state_all([s.cpu() for s in blob["cuda_rng"]])
        start = int(blob.get("next_epoch", int(blob["epoch"]) + 1))
        print(f"  resumed from {path}: epoch {start}, step {self.step}")
        return start

    def train(self) -> float:
        """@return The best validation score seen, in the units of
        ``train.select_on``.
        """
        self.best, self.reported = float("inf"), float("inf")
        self.clipped, self.steps_seen = 0, 0
        self.best_epoch, self.stale = -1, 0
        first_epoch = self.resume()
        started, seen = time.perf_counter(), 0
        stop = False
        for epoch in range(first_epoch, self.cfg.train.epochs):
            # Redraw the prior error and detection noise for this epoch. It is
            # a no-op on a cached split and on any dataset without the hook,
            # and it must happen *before* the iterator is made, because that
            # is when workers fork and take their copy.
            reseed = getattr(self.train_loader.dataset, "set_epoch", None)
            if reseed is not None and self.cfg.data.augment:
                reseed(epoch)
            for batch in self.train_loader:
                if self.step >= self.total_steps:
                    stop = True
                    break
                batch = {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in batch.items()
                }
                for g in self.opt.param_groups:
                    g["lr"] = self.lr_at(self.step)
                sigma = self.sigma_at(self.step)

                with _autocast(self.cfg):
                    out = self.model(batch, sigma_m=sigma)
                total, parts = compute_losses(out, batch, self.cfg.loss)
                if self.teacher is not None:
                    from mapposeformer.distill import distill_losses

                    with torch.no_grad():
                        ref = self.teacher(batch, sigma_m=sigma)
                    kd, kd_parts = distill_losses(out, ref, self.cfg.distill)
                    total = total + kd
                    # Logged beside the supervised parts, not folded into
                    # them, so `loss` stays comparable against a run that has
                    # no teacher.
                    parts.update(kd_parts)

                total.backward()
                grad = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.train.grad_clip
                )
                # `clip_grad_norm_` returns the norm *before* clipping, so this
                # is exactly the fraction of steps whose size the clip decided
                # rather than the schedule. A model with more parameters has a
                # larger gradient norm and is clipped more often, which turns a
                # structural comparison into an optimiser one -- so it is
                # reported rather than assumed.
                self.steps_seen += 1
                self.clipped += int(float(grad) > self.cfg.train.grad_clip)
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)

                self.step += 1
                seen += batch["delta"].shape[0]
                if self.step % self.cfg.train.log_every == 0:
                    elapsed = time.perf_counter() - started
                    left = elapsed / self.step * (self.total_steps - self.step)
                    scalars = {
                        "loss": total.item(),
                        **parts,
                        "lr": self.lr_at(self.step),
                        "sigma_m": sigma,
                        "grad_norm": float(grad),
                        "mass": out["mass"].mean().item(),
                        "frames_per_s": seen / elapsed,
                        "eta_min": left / 60.0,
                    }
                    self.log("train", scalars)
                    print(
                        f"  step {self.step}/{self.total_steps}"
                        f"  loss {total.item():.4f}"
                        f"  pose {parts['pose']:.3f}"
                        f"  match {parts['match']:.3f}"
                        f"  sigma {sigma:.2f}"
                        f"  eta {left / 60:.0f} min",
                        flush=True,
                    )
            if stop:
                break

            if (epoch + 1) % self.cfg.train.eval_every == 0:
                metrics = evaluate(self.model, self.val_loader, self.device)
                metrics["clip_binding"] = self.clipped / max(self.steps_seen, 1)
                self.log("val", metrics)
                print(
                    f"epoch {epoch + 1}: val trans "
                    f"{metrics['trans_rmse']:.4f} m"
                    f"  (prior {metrics['prior_rmse']:.4f})",
                    flush=True,
                )
                key = self.cfg.train.select_on
                if key not in metrics:
                    raise KeyError(
                        f"train.select_on={key!r} is not among the validation "
                        f"metrics: {sorted(metrics)}"
                    )
                # Minimised internally whichever way the metric points, so
                # one comparison serves both; `reported` keeps the real value.
                score = metrics[key]
                if self.cfg.train.select_higher_is_better:
                    score = -score
                if score < self.best:
                    self.best, self.reported = score, metrics[key]
                    self.best_epoch = epoch
                    self.stale = 0
                    self.save("best.pt", epoch, metrics)
                else:
                    self.stale += 1
            self.save("last.pt", epoch, {})

            p = self.cfg.train.patience
            if p and self.stale >= p:
                print(
                    f"early stop: {self.stale} evaluations without improving"
                    f" {self.cfg.train.select_on}; best was epoch"
                    f" {self.best_epoch + 1}",
                    flush=True,
                )
                break

        if self.best == float("inf"):  # stopped before the first evaluation
            metrics = evaluate(self.model, self.val_loader, self.device)
            self.best, self.reported = (
                metrics["trans_rmse"],
                metrics["trans_rmse"],
            )
            self.save("best.pt", 0, metrics)
        print(
            f"done. best val {self.cfg.train.select_on} {self.reported:.4f}"
            f" -> {self.out}"
        )
        return self.reported
