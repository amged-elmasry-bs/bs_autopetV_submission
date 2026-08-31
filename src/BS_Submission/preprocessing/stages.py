"""The three stages, and the orchestration around them.

    seg   organ masks from the raw CT                      (GPU, or supply cached masks)
    ref   per-scan PET reference + cohort-wide z statistics (CPU pool)
    prep  write every channel onto the reference grid       (CPU pool)

`seg` is separable on purpose: it is the only stage needing a GPU and a third-party segmentation
model, so a cohort that already has organ masks can skip straight to `ref`.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from .layout import Layout
from .pipeline import CaseJob, process_case
from .reference import ref_case


def run_segmentation(cases: list[str], layout: Layout, device: str) -> None:
    """Cache liver and aorta masks from the raw CT, skipping cases already done."""
    from totalsegmentator.python_api import totalsegmentator

    for i, case in enumerate(cases, 1):
        directory, liver, aorta = layout.case_seg_paths(case)
        if os.path.exists(liver) and os.path.exists(aorta):
            print(f"[seg {i}/{len(cases)}] skip {case[:48]}", flush=True)
            continue
        os.makedirs(directory, exist_ok=True)
        try:
            totalsegmentator(
                input=layout.raw_ct(case),
                output=directory,
                task="total",
                fast=True,
                roi_subset=["liver", "aorta"],
                device=device,
                quiet=True,
            )
            print(f"[seg {i}/{len(cases)}] done {case[:48]}", flush=True)
        except Exception as exc:
            print(f"[seg {i}/{len(cases)}] ERROR {case[:48]}: {type(exc).__name__}: {exc}", flush=True)


def run_reference(cases: list[str], layout: Layout, workers: int, order: list[str]) -> None:
    """Compute every case's PET reference, and one mean/std over the whole cohort.

    The statistics are accumulated as sums so the cohort value is exact rather than an average
    of per-case averages, which would weight small scans equally with large ones.
    """
    ref_map: dict[str, dict] = {}
    total = squared = 0.0
    count = 0
    sources: dict[str, int] = {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ref_case, c, order, layout): c for c in cases}
        for future in tqdm(as_completed(futures), total=len(futures), desc="ref"):
            try:
                case, ref, src, cv, frac, s, s2, n = future.result()
            except Exception as exc:
                tqdm.write(f"ERROR ref {futures[future][:40]}: {type(exc).__name__}: {exc}")
                continue
            ref_map[case] = {
                "ref": ref,
                "source": src,
                "cv": None if np.isnan(cv) else round(cv, 4),
                "frac_outlier": None if np.isnan(frac) else round(frac, 4),
            }
            sources[src] = sources.get(src, 0) + 1
            total += s
            squared += s2
            count += n

    if not count:
        raise SystemExit("no body voxels accumulated; check the dataset and organ masks")
    mean = total / count
    std = float(np.sqrt(max(squared / count - mean * mean, 1e-12)))
    # Always written into the output directory, never over an override: an override names a
    # released artifact that a run must not silently replace.
    ref_out = os.path.join(layout.out_dir, "ref.json")
    stats_out = os.path.join(layout.out_dir, "pet_norm_stats.npz")
    os.makedirs(layout.out_dir, exist_ok=True)
    with open(ref_out, "w") as handle:
        json.dump(ref_map, handle, indent=1)
    np.savez(stats_out, mean=np.float32(mean), std=np.float32(std))
    print(f"\nreference sources: {sources}")
    print(f"cohort PET global-z: mean={mean:.5f} std={std:.5f} -> {stats_out}", flush=True)


def run_prep(
    cases: list[str],
    layout: Layout,
    workers: int,
    pet_mean: float,
    pet_std: float,
    pet_only: bool = False,
) -> dict[str, int]:
    """Write every channel for every case, and record each case's original geometry."""
    os.makedirs(layout.pet_dir, exist_ok=True)
    if not pet_only:
        for directory in (layout.ct_dir, layout.labels_out_dir, layout.scribbles_dir):
            os.makedirs(directory, exist_ok=True)

    with open(layout.ref_json) as handle:
        ref_map = json.load(handle)
    print(
        f"PET global-z: mean={pet_mean:.5f} std={pet_std:.5f} | references for {len(ref_map)} cases",
        flush=True,
    )

    meta = {}
    if os.path.exists(layout.meta_json):
        with open(layout.meta_json) as handle:
            meta = json.load(handle)

    jobs = []
    for case in cases:
        if case not in ref_map:
            print(f"WARN no reference for {case[:50]} -- run the ref stage first; skipping")
            continue
        jobs.append(CaseJob(case, ref_map[case]["ref"], pet_mean, pet_std, layout, pet_only))

    counts = {"done": 0, "skip": 0, "error": 0}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_case, job): job.case for job in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="prep"):
            case, status, spacing, shape = future.result()
            meta[case] = {"spacing": spacing, "shape": shape}
            if status.startswith("error"):
                counts["error"] += 1
                tqdm.write(f"ERROR {case}: {status}")
            else:
                counts[status] += 1

    with open(layout.meta_json, "w") as handle:
        json.dump(meta, handle)
    print(counts, f"| meta entries: {len(meta)}", flush=True)
    return counts
