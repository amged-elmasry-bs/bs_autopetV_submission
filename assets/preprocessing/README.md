# Preprocessing artifacts

These are the values the released weights were trained against. They are **inputs** to
dataset preparation, not outputs of it.

## `ref.json` (1611 entries)

The per-scan PET reference SUV for every case in the cohort, plus the diagnostics from
computing it.

```json
"fdg_06d55e8295_04-17-2003-…": {
  "ref": 2.19987416267395,
  "source": "aorta",
  "cv": 0.2133,
  "frac_outlier": 0.008
}
```

| field | meaning |
|---|---|
| `ref` | the divisor in `arcsinh(SUV / ref)` — the only field the pipeline consumes |
| `source` | which rung of the fallback chain produced it (`aorta` for all 1611) |
| `cv` | robust coefficient of variation inside the region, `1.4826 × MAD / median` |
| `frac_outlier` | fraction of region voxels rejected as focal hotspots |

Distributions across the cohort: `ref` 0.32–4.65 (median 1.84), `cv` median 0.194,
`frac_outlier` median 1.0%. Reference SUVs are more than twice as variable for PSMA
(sd 0.73) as for FDG (sd 0.32), which is the case for per-scan self-calibration rather than
one cohort-wide divisor.

**Use it directly**, rather than copying it into each output directory:

```bash
bs-preprocess --stage prep --ref-json assets/preprocessing/ref.json …
```

`bs-preprocess --stage ref` regenerates the file from the raw cohort, but needs the CT and PET
of all 1611 cases. Shipping it means `prep` can be run on any subset without that.

A regenerated file will not match this one bit-for-bit. TotalSegmentator is not deterministic:
repeat runs of one case at the same version move the aorta mask by a few voxels (IoU ≈ 0.999),
shifting the reference by about 1e-4 relative — below the float16 quantisation of the arrays
`prep` writes, so it does not change a trained model, but enough that this file is the
authoritative record of what the released weights were fitted against.

## `pet_norm_stats.npz`

The cohort-wide `mean` and `std` of `arcsinh(SUV / ref)` over body voxels: the fixed global
z-score applied after per-scan calibration. The same two numbers are committed as literals in
[`preprocessing/constants.py`](../../src/BS_Submission/preprocessing/constants.py); a test asserts
the two copies never drift apart.

`bs-preprocess` resolves them in that order: `--pet-stats` if given, else a
`pet_norm_stats.npz` already sitting in `--out-dir`, else the committed literals. So a fresh run
needs neither file — the fallback *is* the released input space, and recomputing it for a different
cohort has to be asked for explicitly.

> Regenerating these invalidates every existing checkpoint: the weights expect this exact
> input space.
