#!/usr/bin/env python3
"""Measure how often the covariance comes back non-positive-definite.

A covariance goes to a Kalman filter, which factorises it. If the smallest
eigenvalue is negative the factorisation raises -- and that is the *lucky*
outcome, because the unlucky one is a filter that silently corrupts its own
state through ``gain @ r @ gain.T``. So the question is not "is the formula
right" but "how close to the cliff does real data get", and only real data can
answer it: a synthetic road is too well conditioned to reach the edge.

This runs a trained checkpoint over validation frames and evaluates the same
Hessian two ways -- the naive arithmetic and the production path -- reporting
the fraction that comes back non-PD and the worst eigenvalue seen. It also
saves the worst offending ``(hessian, cost, mass)`` so a regression test can
use a case that actually failed rather than one invented to look like it.

    tools/probe_cov.py --config configs/nuscenes.yaml \
        --ckpt runs/base/best.pt --split val --limit 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.config import load_config, parse_overrides, upgrade
from mapposeformer.data import build_dataset
from mapposeformer.model import MapPoseFormer
from mapposeformer.solve import (
    MIN_MASS,
    MIN_RESIDUAL_M2,
    curvature_covariance,
)


def naive_covariance(hessian, cost, mass, prior_information=None):
    """The arithmetic each shortcut invites -- kept so the gap is measurable.

    Four differences from the production path, and the floor is the one that
    matters: a 1e-8 floor on the residual scale rather than a physical one,
    `mass` rather than `dof - 3` as the divisor, float32 rather than float64,
    and a plain inverse of a matrix nobody symmetrised.
    """
    scale = (cost / mass.clamp_min(MIN_MASS)).clamp_min(1e-8)
    information = hessian.float() / (2.0 * scale.float()).view(-1, 1, 1)
    if prior_information is not None:
        information = information + prior_information.float()
    return torch.linalg.inv(information)


def production_covariance(hessian, cost, dof, prior_information=None):
    """The production path itself, called rather than copied.

    A probe that duplicates the thing it probes measures its own copy. Inlining
    the formula here "so both are in one file" is how this tool would go on
    dividing by `mass` after `solve.py` moved to `dof - 3`, printing wrong
    numbers while claiming to measure the real one.
    """
    return curvature_covariance(hessian, cost, dof, prior_information)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=200, help="batches")
    ap.add_argument("--save", default="", help="write the worst case here")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.overrides))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The checkpoint carries the config that built it, and that is the one to
    # trust: `--config` here only says which data to read. A model trained at
    # four layers cannot be rebuilt from a file whose default is two.
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_params = upgrade(state["config"]).model
    model = MapPoseFormer(model_params).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    ds = build_dataset(cfg.data, args.split)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4
    )

    n = naive_bad = prod_bad = 0
    worst_naive = worst_prod = float("inf")
    worst_case = None
    scales: list[torch.Tensor] = []
    dofs: list[torch.Tensor] = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.limit:
                break
            batch = {
                k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)
            }
            out = model(batch)
            hess, cost = out["hessian"], out["cost"]
            mass, dof = out["mass"], out["dof"]
            prior = out.get("prior_information")
            scales.append((cost / (dof - 3.0).clamp_min(1.0)).cpu())
            dofs.append((dof / mass.clamp_min(MIN_MASS)).cpu())

            paths = (
                ("naive", naive_covariance, mass),
                ("production", production_covariance, dof),
            )
            for name, fn, divisor in paths:
                try:
                    cov = fn(
                        hess.cpu(),
                        cost.cpu(),
                        divisor.cpu(),
                        None if prior is None else prior.cpu(),
                    )
                    ev = torch.linalg.eigvalsh(cov.double()).min(-1).values
                except Exception:
                    ev = torch.full((hess.shape[0],), -1.0)
                bad = (ev <= 0) | ~torch.isfinite(ev)
                if name == "naive":
                    naive_bad += int(bad.sum())
                    n += ev.numel()
                    if float(ev.min()) < worst_naive:
                        worst_naive = float(ev.min())
                        j = int(ev.argmin())
                        worst_case = {
                            "hessian": hess[j].cpu().clone(),
                            "cost": cost[j].cpu().clone(),
                            "mass": mass[j].cpu().clone(),
                            "dof": dof[j].cpu().clone(),
                            "min_eig_naive": worst_naive,
                        }
                else:
                    prod_bad += int(bad.sum())
                    worst_prod = min(worst_prod, float(ev.min()))

    print(f"frames             {n}")
    print(f"naive non-PD       {naive_bad} ({naive_bad / max(n, 1):.2%})")
    print(f"solve.py non-PD    {prod_bad} ({prod_bad / max(n, 1):.2%})")
    print(f"worst eig naive    {worst_naive:+.4e}")
    print(f"worst eig solve.py {worst_prod:+.4e}")

    # Where the residual floor sits relative to the data it is meant to guard.
    # A floor that binds on ordinary frames is not a guard, it is a knob that
    # silently widened every covariance ever reported.
    # dof/mass separates the two residual kinds without needing the flag: a
    # polyline point contributes 1 and a pole 2, so the ratio is 1.0 on a road
    # of lane lines alone and rises toward 2.0 as point landmarks come in.
    ratio = torch.cat(dofs)
    print(
        f"dof/mass           median {float(ratio.median()):.4f}"
        f"   min {float(ratio.min()):.4f}   max {float(ratio.max()):.4f}"
    )

    sc = torch.cat(scales)
    q = torch.tensor([0.0, 0.001, 0.01, 0.5, 1.0])
    vals = torch.quantile(sc.double(), q.double())
    print("residual scale cost/(dof-3) (m^2):")
    names = ("min", "p0.1", "p1", "median", "max")
    for name, v in zip(names, vals, strict=True):
        print(f"  {name:>6} {float(v):.6e}")
    print(
        f"  floor  {MIN_RESIDUAL_M2:.6e}"
        f"   binds on {float((sc < MIN_RESIDUAL_M2).float().mean()):.2%}"
        f" of frames"
    )

    if args.save and worst_case is not None:
        worst_case["min_eig_production"] = worst_prod
        torch.save(worst_case, args.save)
        print(f"worst case written to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
