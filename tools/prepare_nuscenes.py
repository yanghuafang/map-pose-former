#!/usr/bin/env python3
"""Turn nuScenes into per-scene worlds this project can train on.

Runs once, offline, and writes what the training loop reads: one ``World`` per
scene plus a manifest naming the geographic split. Nothing in
``mapposeformer/engine`` or a dataloader worker parses a nuScenes JSON, which
is the rule that keeps the devkit -- and its startup cost -- out of the hot
path.

    tools/prepare_nuscenes.py --root <dataset> --out <cache>
    tools/prepare_nuscenes.py --root <dataset> --out <cache> --no-traffic-lights

The second form drops the only point landmark nuScenes has, which is the
control for whether 307 sparse traffic lights recover any of the along-track
observability the missing poles cost.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapposeformer.data.nuscenes import (
    LOCATIONS,
    NuScenesParams,
    geographic_split,
    load_map,
    load_scenes,
    scene_world,
)


def main() -> None:
    """@brief Build the cache. @return None."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="dataset root")
    ap.add_argument("--out", required=True, help="cache directory to write")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--radius", type=float, default=120.0, help="map radius, m")
    ap.add_argument("--buffer", type=float, default=0.0, help="split buffer, m")
    ap.add_argument("--no-traffic-lights", action="store_true")
    args = ap.parse_args()

    p = NuScenesParams(
        root=args.root,
        version=args.version,
        map_radius_m=args.radius,
        traffic_lights=not args.no_traffic_lights,
    )
    out = Path(args.out)
    (out / "scenes").mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    scenes = load_scenes(p)
    dt = time.perf_counter() - t0
    print(f"{len(scenes)} scenes with keyframe poses  ({dt:.1f}s)")

    splits = geographic_split(scenes, buffer_m=args.buffer)
    assigned = {n for names in splits.values() for n in names}
    print(
        "  ".join(f"{k} {len(v)}" for k, v in splits.items())
        + f"   kept {len(assigned)}/{len(scenes)}"
    )

    frames: dict[str, int] = {}
    # Recorded because every city's map has its own origin: two scenes in
    # different cities can sit at the same (x, y) and be 15 000 km apart, so
    # any spatial comparison has to be made within one location.
    where: dict[str, str] = {}
    for location in LOCATIONS:
        names = [n for n in assigned if scenes[n]["location"] == location]
        if not names:
            continue
        t0 = time.perf_counter()
        city = load_map(p.root, location, p)
        for name in names:
            world = scene_world(city, scenes[name]["trajectory"], p)
            torch.save(world, out / "scenes" / f"{name}.pt")
            frames[name] = int(world.trajectory.shape[0])
            where[name] = location
        dt = time.perf_counter() - t0
        print(
            f"  {location:26s} {len(names):4d} scenes, "
            f"{len(city.elements):5d} map elements  ({dt:.0f}s)"
        )

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "version": args.version,
                "traffic_lights": p.traffic_lights,
                "map_radius_m": p.map_radius_m,
                "split_buffer_m": args.buffer,
                "splits": splits,
                "frames": frames,
                "locations": where,
            },
            indent=2,
        )
    )
    total = sum(frames.values())
    print(f"\nwrote {len(frames)} scenes, {total} keyframes, to {out}")


if __name__ == "__main__":
    main()
