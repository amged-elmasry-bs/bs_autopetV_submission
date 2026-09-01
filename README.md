# BS_Submission — whole-body PET/CT lesion segmentation with prompt channels

Interactive lesion segmentation in whole-body PET/CT. The network takes four input channels — CT,
PET, and two prompt heatmaps (foreground and background) — and predicts a lesion mask, alongside an
auxiliary 11-class organ head inherited from the pretrained backbone and carried through training
unused. One user scribble on the model's largest error lifts cohort Dice from 0.55 to 0.72; see
[Results](#results).

Licensed under Apache 2.0. Third-party components carry their own terms — notably the model
weights are **CC-BY-4.0** and the datasets **CC BY-NC 4.0**, neither of which is Apache 2.0. See
[NOTICE](NOTICE).

## Commands

Installing the package puts six commands on your `PATH`. Each has `--help` with its arguments
grouped by what they affect.

| command | what it does | needs |
|---|---|---|
| [`bs-segment`](#segment-one-scan) | one case's raw CT + PET → a lesion mask, whole chain | `[preprocess]`¹ |
| [`bs-score`](#score-a-mask) | one predicted mask vs. its annotation → Dice, F1, volumes | `[evaluate]` |
| [`bs-preprocess`](#preprocessing) | raw NIfTI cohort → model-ready arrays | `[preprocess]` |
| [`bs-train`](#train) | train one cross-validation fold | base |
| [`bs-predict`](#predict) | masks for a fold's validation split, from prepared arrays | base |
| [`bs-evaluate`](#evaluate) | the challenge's interactive correction protocol, scored | `[evaluate]` |

¹ only to derive the PET reference from the CT; `--ref <suv>` skips it and the base install suffices.

Two helper scripts and two maintenance scripts sit in [`scripts/`](scripts/): `train_fold.sh` and
`tensorboard.sh` for running, `prepare_pretrained.py` and `verify_fold_alignment.py` for weights.

## Install

```bash
git clone <repository-url> && cd BS_Submission
pip install -e .
```

Python ≥3.10. Optional extras, all installable together — `pip install -e ".[preprocess,evaluate]"`:

| extra | pulls in | for |
|---|---|---|
| `preprocess` | TotalSegmentator (pinned 2.15.0), tqdm | `bs-preprocess`, and `bs-segment` without `--ref` |
| `evaluate` | connected-components-3d, nibabel, networkx, matplotlib | `bs-score`, `bs-evaluate` |
| `dev` | pytest, ruff, black | linting |

**Every dependency is pinned exactly**, to the versions the reported results were produced on and
which are verified end to end. The torch pin in particular is load-bearing: releases past 2.3.x
ship no CUDA kernels below compute capability 7.5, so on a V100 (CC 7.0) a newer torch imports
cleanly and reports `torch.cuda.is_available() == True`, then fails at the first kernel launch with
`no kernel image is available for execution on the device`.

pip resolves torch to its default CUDA variant. To choose one explicitly:

```bash
pip install "torch==2.3.1" "torchvision==0.18.1" \
    --index-url https://download.pytorch.org/whl/cu121
```

To run against a **different** torch — newer hardware, or a CUDA build already in the environment
that you do not want replaced — install it yourself and add `--no-deps`, or supply a constraints
file.

For an exact, reproducible stack use [Docker](#docker) — the pinned image matters, see the note in
`docker/Dockerfile`.

## Quickstart

A mask for one scan — the shortest useful path. No dataset preparation and no training; you need
only the published weights and one case's CT and PET.

```bash
pip install -e ".[preprocess,evaluate]"

# 1. trained weights -> assets/model/fold<n>.pt   (see Weights for the download)
unzip "Models Weights.zip"
for n in 0 1 2 3 4; do cp "Models Weights/model_checkpoint_fold_${n}.pt" assets/model/fold${n}.pt; done

# 2. segment, ensembling all five folds as the submitted container does
bs-segment --ct ct.nii.gz --pet pet.nii.gz --fold 0 1 2 3 4 --out mask.nii.gz

# 3. score it, if you have the annotation
bs-score --prediction mask.nii.gz --reference labelsTr/case.nii.gz
```

`--clicks lesion-clicks.json` adds user prompts; without it the pass is unguided. PET must be in
SUV. The first run downloads TotalSegmentator's weights unless `--ref` supplies the reference SUV.

Everything past this point — [preprocessing](#preprocessing), [training](#train),
[cross-validation](#evaluate) — works from the full 1611-case cohort.

## Repository layout

```
src/BS_Submission/     the package
  preprocessing/       raw NIfTI -> prepared arrays; owns the input space
  inference/           sliding window, prompt encoding, bs-segment's whole chain
  evaluation/          scoring and the interactive protocol
    upstream/          vendored unmodified from the organisers -- do not edit
  models/              vendored residual-encoder U-Net
splits/                the five-fold assignment, version-controlled (10 CSVs)
assets/model/          where weights go; README only, nothing in git
assets/preprocessing/  the per-case PET references and cohort statistics
scripts/               training helpers and weight preparation
docker/                Dockerfile, build.sh, export.sh
```

## Method

| | |
|---|---|
| Input | `[ct, pet, prompt_foreground, prompt_background]`, 192³ patches |
| Architecture | Residual-encoder U-Net (`nnUNetResEncUNetLPlansMultiTalent` 3d_fullres preset), 6 stages, ~102M parameters |
| Objective | Cross-entropy + soft Tversky (α=0.3, β=0.7), foreground classes only, with nnU-Net-style deep supervision |
| Patch sampling | Foreground-weighted: 33% deliberately lesion-free windows (background), otherwise a window containing a lesion voxel |
| Optimiser | SGD, lr 1e-3, momentum 0.99 (Nesterov), weight decay 3e-5, gradient-norm clip 12 |
| Precision | 16-mixed via Lightning Fabric |
| Output | lesion probability, thresholded at **0.65** (tuned on validation, not 0.5) |
| Schedule | None — the learning rate is constant for the whole run |

Two design points worth stating explicitly, because both are easy to get wrong:

**Tversky excludes the background class.** With β > α the loss is meant to punish missed lesions
harder than spurious ones. Averaging the background class back in cancels that asymmetry exactly,
since a foreground false negative *is* a background false positive. Background stays supervised
through the cross-entropy term.

**Validation uses the same random cropping as training.** It is a patch-level estimate, not
whole-volume evaluation, so the epoch-to-epoch validation curve is genuinely noisy. Compare runs
by trend, not by single epochs.

## Data

Two separate downloads, neither redistributed here. **The heatmaps are not part of the image
release** — `imagesTr/` carries only channels 0 and 1 — so both are required.

**1. Images and annotations.** The combined FDG + PSMA release, 1611 cases:
[NIfTI, DOI 10.57754/FDAT.rdkqd-wdh87](https://doi.org/10.57754/FDAT.rdkqd-wdh87). It unzips into
exactly the layout `--dataset-dir` expects, so no conversion is needed:

```
PSMA-FDG-PET-CT-Lesions_v2/
  imagesTr/          {case}_0000.nii.gz   CT
                     {case}_0001.nii.gz   PET
  labelsTr/          {case}.nii.gz        lesion annotation
  splits_final.json  the official folds -- the same assignment shipped in splits/
```

**2. Prompt heatmaps.** Published by the organisers in
[lab-midas/autoPETV](https://github.com/lab-midas/autoPETV) as `nnunet-baseline/heatmaps.zip` —
3.6 MB holding 1611 `_0002.nii.gz` (foreground) and 1611 `_0003.nii.gz` (background). Unzip and
point `--heatmaps-dir` at the directory of `.nii.gz` files. They arrive on **per-case grids that
differ from the images**, which is what the `prep` stage resamples away.

### Licence and citation

The databases are **not this repository's Apache 2.0**. Cite them in any publication:

- **FDG** — Gatidis S, Kuestner T. *A whole-body FDG-PET/CT dataset with manually annotated tumor
  lesions (FDG-PET-CT-Lesions)* [Dataset]. The Cancer Imaging Archive, 2022.
  [DICOM](https://doi.org/10.7937/gkr0-xv29) ·
  [NIfTI](https://doi.org/10.57754/FDAT.8f14a-pf846) ·
  [paper](https://doi.org/10.1038/s41597-022-01718-3)
- **PSMA** — Jeblick K, et al. *A whole-body PSMA-PET/CT dataset with manually annotated tumor
  lesions (PSMA-PET-CT-Lesions)* (Version 1) [Dataset]. The Cancer Imaging Archive, 2024.
  [DICOM](https://doi.org/10.7937/r7ep-3x37) ·
  [NIfTI](https://doi.org/10.57754/FDAT.g27kx-86t35)

Both are dual-licensed by their creators: DICOM under the **TCIA Restricted License**, requiring a
signed agreement before access, and NIfTI under
**[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)** — non-commercial, with
attribution. That governs the data whatever you do with the code; for commercial use the
organisers ask to be contacted first. Each tracer is also published on its own — FDG
[NIfTI](https://doi.org/10.57754/FDAT.wf9fy-txq84), PSMA
[NIfTI](https://doi.org/10.57754/FDAT.6gjsg-zcg93) — but `splits/` references the combined
1611-case cohort. The splits and `assets/preprocessing/ref.json` are derived from these databases;
see [NOTICE](NOTICE).

### Splits

Version-controlled in [`splits/`](splits/) — results cannot be reproduced without the exact case
assignment. 1611 cases, each validated in exactly one fold: fold 0 is 1288 train / 323 val, folds
1–4 are 1289 / 322.

Each row holds **paths relative to `--data-root`**, so the files stay portable:

```csv
ct,pet,hm0,hm1,mask
ct/case_0000.npy,pet/case_0001.npy,scribbles/v1/case_0002.npy,scribbles/v1/case_0003.npy,labels/case.npy
```

- `ct`, `pet` — intensity volumes, pre-normalised, `.npy`
- `hm0` — **foreground** (lesion) prompt heatmap, `_0002`, uint8-encoded
- `hm1` — **background** prompt heatmap, `_0003`, uint8-encoded
- `mask` — lesion labels

The prompt channels are not interchangeable: swapping them inverts the correction a user is
asking for. Both are scaled to [0, 1] at load time by dividing by 255, and a prompt encoded at
inference must use that same quantisation so it lands on the scale the weights were trained on.
See [`constants.py`](src/BS_Submission/constants.py) for the contract.

Volumes are memory-mapped, so only the sampled window is read from disk. A CSV holding absolute
paths also works, and `--path-from` / `--path-to` rewrite their prefix if the data has moved
between machines.

## Weights

**Nothing here is stored in git.** Weights are downloaded and converted locally; the repository
carries only the instructions and the conversion script. Full detail, with per-file checksums, in
[assets/model/README.md](assets/model/README.md).

### Trained checkpoints — to run the model

The five trained folds: **[10.5281/zenodo.22176946](https://doi.org/10.5281/zenodo.22176946)**,
one archive `Models Weights.zip` (1.8 GB, md5 `5eb2aaa77d632577be08b53b12e86bd7`) holding
`model_checkpoint_fold_0.pt` … `_fold_4.pt`. **CC-BY-4.0**, not this repository's Apache 2.0.

Copy them to `assets/model/fold<n>.pt` and `--fold` resolves them; see the
[quickstart](#quickstart).

### Initialisation weights — to train

Each fold initialises from the **same** fold upstream, so a fold's starting point never trained on
the cases it later validates on.

**1. Download** [autoPET III LesionTracer](https://zenodo.org/records/14007247) (Rokuss, DKFZ),
`autoPET-3-LesionTracer.zip`, 3.8 GB, md5 `566016409b0bd14770c0b57c1f2873f1`. Also **CC-BY-4.0**.

```bash
unzip autoPET-3-LesionTracer.zip
# -> Dataset222_AutoPETIII_2024/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3/
#      fold_0/checkpoint_final.pth  ...  fold_4/checkpoint_final.pth
```

**2. Convert** from two input channels to four. The published network takes CT and PET; this one
also takes two prompt channels.

```bash
python scripts/prepare_pretrained.py \
    Dataset222_AutoPETIII_2024/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3 \
    assets/model
# -> assets/model/pretrained_fold0.pt ... pretrained_fold4.pt   (966 tensors, ~410 MB each)
```

Every trained weight is kept; only the stem convolution is widened, with the two prompt channels
**zero-initialised** — so at step zero the network computes exactly what the pretrained CT/PET
model computed and the prompts contribute nothing. CC-BY-4.0 requires modifications to be
indicated; this widening is the modification, recorded in [NOTICE](NOTICE).

**3. Train.** `--fold <n>` now finds `assets/model/pretrained_fold<n>.pt` automatically.

Optionally confirm the fold assignment matches upstream, so the initialisation really has not seen
the validation cases:

```bash
python scripts/verify_fold_alignment.py \
    Dataset222_AutoPETIII_2024/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3
```

## Preprocessing

Required before training, `bs-predict` or `bs-evaluate`: the released dataset ships as raw NIfTI
plus separately published heatmaps, not prepared arrays. Skip it only if you have already run
`bs-preprocess` and still have its `--out-dir`. Download both halves of the cohort first — see
[Data](#data).

```bash
pip install -e ".[preprocess]"
bs-preprocess --dataset-dir /raw/PSMA-FDG-PET-CT-Lesions_v2 \
              --heatmaps-dir /raw/heatmaps \
              --out-dir /data/prepared --stage all
```

| stage | what it does | cost |
|---|---|---|
| `seg` | liver and aorta masks from the raw CT | GPU; skippable if masks are cached |
| `ref` | each scan's PET reference region + the cohort-wide z statistics | CPU pool |
| `prep` | write all four channels and both masks onto the reference grid | CPU pool |

Run them separately with `--stage seg|ref|prep`, shard `seg` across GPUs with
`--num-shards`/`--shard-index`, and smoke-test with `--limit 5`. The result is the `--data-root`
every other command reads:

```
/data/prepared/
  ct/  pet/  scribbles/v1/     the four input channels, .npy
  labels/    {case}.npy        on the target grid, for training
             {case}_orig.npy   on the annotation grid, for scoring
  seg/                         cached organ masks
  meta.json  ref.json  pet_norm_stats.npz
```

**PET normalisation is the substantive part.** PET is normalised as
`global_z(arcsinh(SUV / reference))`, where the reference is a robust SUV from a CT-derived organ
mask **on that same scan** — so each scan self-calibrates and the axis stays comparable across
tracers without needing labels. Focal outliers above `median + 3 × 1.4826 × MAD` are dropped
first, so a tumour inside the reference organ cannot inflate it. The `global_z` afterwards only
re-centres the already-harmonised axis using fixed cohort values, and does **not** re-normalise
per scan.

The reference region is the **aortic blood pool** (`--ref-organ aorta`, the default), which is what
the released dataset and weights use — all 1611 cases. Liver is selectable and its masks were
available for every case, but liver uptake is strongly tracer-dependent: on sampled PSMA scans the
liver reference came out 5–6× the blood-pool value, which would put FDG and PSMA on different axes.
Switching organ changes the PET axis and invalidates the released checkpoints.

**Geometry is a real resample, not an alignment.** The published prompt heatmaps do not share one
grid: the FDG half arrives at the target spacing `(2.0364, 2.0364, 3.0)`, while PSMA cases come on
their own grids — `(4.0728, 4.0728, 2.0)` and `(2.7344, 2.7344, 3.27)` both occur — and dtypes are
mixed (`uint8` and `float32`) within the same release. Everything is resampled onto a CT-derived
reference grid: BSpline for intensities, Linear for heatmaps (clipped to [0, 1], stored `uint8×255`),
nearest-neighbour for labels. Masks are written twice — on the reference grid for training, and on
their original grid, since evaluation belongs in the space the annotation was made in.

**The per-case references are shipped too**, in
[`assets/preprocessing/ref.json`](assets/preprocessing/README.md) — 1611 entries, the reference SUV
each scan was normalised against plus the diagnostics from computing it. Pass
`--ref-json assets/preprocessing/ref.json` to prepare the cohort without re-running the `ref` stage,
which needs the raw PET of every case.

**Prefer the shipped file over re-deriving.** Re-running `ref` does not reproduce these values
bit-for-bit, because TotalSegmentator is not deterministic: on repeat runs of the same case, same
version, same GPU, the aorta mask moves by a few voxels (IoU ≈ 0.999) and the reference shifts by
around 1e-4 relative. That is far below the float16 quantisation of the stored arrays, so it does
not change a trained model — but it does mean the shipped file, not a fresh `ref` run, is what
reproduces the exact input space the released weights were fitted on.

**The constants live in the repository**, in
[`preprocessing/constants.py`](src/BS_Submission/preprocessing/constants.py): the CT clip window and
z-score, the target spacing, and the cohort PET mean/std. They define the input space every released
checkpoint expects, so they are committed as auditable values rather than binary files. `--pet-stats`
overrides them for a different cohort; changing them invalidates existing weights.

## Train

```bash
bs-train --fold 0 --data-root /path/to/prepared
```

That resolves the split pair from `splits/` and the initialisation weights from `assets/model/`,
and fails immediately with a clear message if either is missing. Per fold, with output laid out
under `runs/`:

```bash
scripts/train_fold.sh 0 /path/to/prepared
```

Anything can be overridden — `--train-csv`, `--val-csv`, `--checkpoint-path` (or
`--checkpoint-path none` to train from scratch), `--devices 0,1`, `--epochs`.

Each epoch runs one training pass then one validation pass. Checkpoints keep the **five lowest
validation losses** plus a rolling `last`, with the value in the filename. Watch a run with:

```bash
scripts/tensorboard.sh runs 6006
```

Scalars: `train_loss_train_loss`, `val_loss_eval_loss`, and `_on_epoch` aggregates of each.

## Predict

```bash
bs-predict --fold 0 --data-root /path/to/prepared --out-dir runs/pred0
```

`--fold` selects **both** halves of an evaluation: the fold's validation split *and* the model
trained on that fold. Pairing them by hand is how a model ends up scored on cases it was trained
on, so the correct pairing is the default. Weights resolve as `<--model-dir>/fold<n>.pt`;
`--val-csv` and `--checkpoint` override either half.

Prediction reuses the training normalisation rather than restating it — `inference/` imports grids,
resampling and the PET reference from `preprocessing/`, so the two cannot drift onto different
input spaces. What is inference-specific:

- **sliding window** over 192³ patches at 0.5 overlap, Gaussian-weighted so a voxel is dominated
  by the patches that saw it away from their edges
- **reverse resample** of the *continuous* probability back to the annotation grid, and only then
  the 0.65 threshold — thresholding first would put nearest-neighbour staircase edges on every
  lesion
- **prompt encoding**: coordinates on the original grid, marked, resampled Linear onto the target
  grid, clipped and quantised to 255 levels, the same scale the stored training prompts use

## Segment one scan

`bs-predict` works from a prepared cohort. `bs-segment` takes one case's raw scans instead — the
same inputs the challenge provides — and does the whole chain itself: PET reference, normalisation,
resampling, prediction, and a mask written on the CT's own geometry.

```bash
bs-segment --ct ct.nii.gz --pet pet.nii.gz \
           --clicks lesion-clicks.json \
           --fold 0 1 2 3 4 \
           --out mask.nii.gz
```

**Several folds means an ensemble.** Each patch is run through every fold and their softmax outputs
are averaged before stitching — a per-patch ensemble, which is what the submitted container does by
default. Averaging probabilities rather than logits is deliberate: softmax is not linear, so the two
are different combinations. One fold divides by exactly 1.0, leaving single-model results untouched.
Loaded parameters are checksummed and a warning is printed if every fold turns out identical, since
copying one checkpoint under five names ensembles happily and silently returns that one model.

**Clicks** are read in either shape the challenge uses — the platform's `lesion-clicks.json`
(`{"points": [{"point": [x, y, z], "name": "tumor"}, …]}`) or the plain
`{"tumor": [[x, y, z], …], "background": […]}` the simulator writes. Coordinates are on the CT grid.
Omit `--clicks` for an unguided pass.

**The PET reference** needs an aorta mask, so TotalSegmentator runs on the CT unless `--ref` supplies
the value. The rung that produced it is printed: a scan that quietly fell through to `body_p50` or
`global_median` is not on the axis the weights were trained against, even though it still yields a
mask. `--organ-dir` caches the masks so a second run skips the segmentation.

## Score a mask

```bash
bs-score --prediction mask.nii.gz --reference labelsTr/case.nii.gz
```

Dice, lesion-detection F1, lesion TP/FP/FN and unmatched volumes in ml, through the same scorer
`bs-evaluate` uses, so the two cannot disagree. NIfTI and `.npy` are both read; spacing comes from
the NIfTI header, or `--spacing` for `.npy`, since the volume metrics are in ml. `--json` writes the
full record.

Both masks must be on the same grid. `bs-segment` writes on the CT's original grid, which is where
the annotation lives — the prepared `labels/<case>.npy` is on the target grid, so compare against
`labels/<case>_orig.npy` instead.

## Evaluate

`bs-evaluate` drives the challenge's interactive protocol: predict, find the largest remaining
error, draw one scribble on it, predict again. Scores land in per-case JSON alongside a printed
per-round table.

```bash
pip install -e ".[evaluate]"
bs-evaluate --fold 0 --data-root /path/to/prepared --out-dir runs/eval0
```

`--rounds` (default 5) counts the scribbles **after** the unguided first pass, so the default is six
inference passes per case. This differs from the upstream script, whose `max_iters=5` yields only
four scribbles because its first iteration adds none.

Scoring is on each case's **original** grid: the probability is reverse-resampled before
thresholding and compared against `labels/<case>_orig.npy`, which is where the annotation lives.
Cases with no annotated lesions have an undefined Dice and F1 under the official scorer, which
returns NaN — they are excluded from the averages rather than counted as perfect or as zero, as the
official evaluation does.

Every round's mask is saved, bit-packed and compressed to roughly 60 kB. That is what makes
`--rescore` possible: it recomputes every metric from the stored masks with no model and no GPU, so
a changed metric definition costs CPU minutes instead of another pass over the cohort. Pass
`--no-save-masks` to score without keeping them.

```bash
bs-evaluate --data-root /path/to/prepared --out-dir runs/eval0 --rescore    # no GPU
bs-evaluate --data-root /path/to/prepared --out-dir runs/eval0 --aggregate  # table only
```

A run resumes: a case whose JSON already exists is skipped, so an interrupted sweep continues where
it stopped.

### The vendored scorer

Scoring and the interactive correction protocol come from the challenge organisers, vendored
**unmodified** under [`evaluation/upstream/`](src/BS_Submission/evaluation/upstream/) from
[lab-midas/autoPETV](https://github.com/lab-midas/autoPETV) (commit 231d9a8, Apache 2.0 — see
[NOTICE](NOTICE)).

They are copied rather than reimplemented deliberately. `metrics.py` is the official scorer, and a
reimplementation would quietly stop the numbers being comparable with other entries. The scribble
rule in `simulate_scribbles.py` *is* the protocol, and it is subtle — the choice between a
foreground and a background correction compares the **length of the drawn scribbles**, not the
volume of the errors — so reproducing it from a description would be guesswork. A test asserts both
files stay byte-identical to what was received: modifying them is permitted but would oblige us to
state the change, and an accidental formatter run should fail rather than create that duty silently.

## Docker

The image contains the **code, its dependencies and the repository's own small data files** —
both extras, so all six commands run in it, plus [`splits/`](splits/) and
[`assets/preprocessing/`](assets/preprocessing/) so that `--fold` and `--ref-json` resolve inside
the container. Datasets, model weights and outputs are mounted at run time, never baked in.

TotalSegmentator's *model weights* are deliberately **not** baked in: they are non-commercial, and
shipping them inside a permissively licensed image would redistribute them. They download on first
use, so the first `bs-segment` run needs network access — mount a cache at `TOTALSEG_HOME_DIR` to
keep later runs offline.

```bash
docker/build.sh          # -> image tagged bs-submission:latest
```

### Run a command

Each `-v host:container` maps a directory on your machine to a path inside the container. The image
has **no fixed entry point** — name the command you want, so any of the six is reachable:

```bash
docker run --rm --gpus all \
  -v /path/to/prepared:/data \
  -v $PWD/assets/model:/weights \
  -v $PWD/runs:/output \
  bs-submission:latest \
    bs-train \
    --fold 0 \
    --data-root /data \
    --checkpoint-path /weights/pretrained_fold0.pt \
    --checkpoint-save-path /output/checkpoints \
    --log-root-dir /output/logs
```

Any other command is named the same way:

```bash
docker run --rm --gpus all \
  -v /path/to/prepared:/data \
  -v $PWD/assets/model:/weights \
  -v $PWD/runs:/output \
  bs-submission:latest \
    bs-predict \
    --fold 0 \
    --data-root /data \
    --model-dir /weights \
    --out-dir /output/pred0
```

`--fold` finds the split CSVs in the image, but **the trained weights are not in it** — they are
mounted, so point `--model-dir` at the mount. `bs-evaluate` takes the same `--model-dir`; `bs-train`
above names `--checkpoint-path` directly for the same reason.

Paths in the arguments are the **container's** (`/data`, `/weights`, `/output`), not the host's.
`--rm` removes the container when it exits; the image and anything written to `/output` remain. Run
with no arguments to get `bs-train --help`.

The container runs as its own unprivileged user rather than yours, so whatever is mounted at
`/output` must be writable by it — `chmod 777 runs` once after creating it, or mount a directory the
container can already write. Otherwise the run fails on its first write with `PermissionError`.

### Moving it to a machine with no internet

```bash
docker/export.sh                          # -> bs-submission.tar.gz
# on the target machine:
docker load < bs-submission.tar.gz
```

`docker save` serialises the image; `docker load` restores it under the same tag. This is the form a
Grand Challenge submission takes.

## Results

Five-fold cross-validation over the full 1611-case cohort, under the interactive protocol above: one
unguided pass (r0) followed by five rounds, each adding one simulated scribble on the model's largest
remaining error. Every fold is scored **only on its own held-out split**, using the checkpoint
trained on that fold, so no case is ever scored by a model that trained on it. Scoring is the
organisers' `MetricEvaluator`.

**Dice**

| fold | r0 | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|---|
| 0 | 0.5708 | 0.7344 | 0.7536 | 0.7591 | 0.7637 | 0.7654 |
| 1 | 0.4583 | 0.7130 | 0.7424 | 0.7523 | 0.7598 | 0.7638 |
| 2 | 0.5374 | 0.7164 | 0.7242 | 0.7281 | 0.7300 | 0.7310 |
| 3 | 0.6312 | 0.7291 | 0.7422 | 0.7480 | 0.7528 | 0.7540 |
| 4 | 0.5719 | 0.7186 | 0.7328 | 0.7390 | 0.7414 | 0.7416 |
| **all** | **0.5525** | **0.7222** | **0.7390** | **0.7452** | **0.7495** | **0.7511** |

**F1** (lesion detection, IoU > 0.1, connectivity 18)

| fold | r0 | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|---|
| 0 | 0.5327 | 0.7160 | 0.7359 | 0.7388 | 0.7493 | 0.7506 |
| 1 | 0.4394 | 0.6908 | 0.7139 | 0.7232 | 0.7302 | 0.7395 |
| 2 | 0.5103 | 0.7010 | 0.7066 | 0.7093 | 0.7122 | 0.7135 |
| 3 | 0.6142 | 0.7062 | 0.7250 | 0.7272 | 0.7334 | 0.7347 |
| 4 | 0.5437 | 0.7039 | 0.7178 | 0.7230 | 0.7224 | 0.7246 |
| **all** | **0.5266** | **0.7034** | **0.7197** | **0.7242** | **0.7294** | **0.7325** |

**The first correction does most of the work.** Averaged over the cohort, r0 → r1 is worth +0.170
Dice; the four rounds after it add +0.029 between them. That is the expected shape for an
interactive system — a single scribble resolves the cases the model misses outright, and there is
progressively less left for later rounds to fix.

**Where the gain comes from.** It is concentrated almost entirely in small lesions, which the
unguided pass tends to miss. Bucketed by annotated volume on fold 0:

| lesion size (voxels) | r0 | r5 | gain |
|---|---|---|---|
| 0–500 | 0.220 | 0.743 | **+0.523** |
| 500–2 000 | 0.486 | 0.688 | +0.202 |
| 2 000–10 000 | 0.707 | 0.785 | +0.078 |
| 10 000+ | 0.798 | 0.809 | +0.011 |

Large lesions are already found without guidance and gain almost nothing; the whole effect lives in
the bottom two buckets.

**Which weights produced these.** Each fold's row comes from the checkpoint published in
[the release](https://doi.org/10.5281/zenodo.22176946), verified by md5:

| fold | epoch | val loss | md5 (first 8) |
|---|---|---|---|
| 0 | 394 | 0.199 | `088dd2a8` |
| 1 | 451 | 0.224 | `7a6bd366` |
| 2 | 271 | 0.238 | `af40f305` |
| 3 | 375 | 0.204 | `06de9900` |
| 4 | 486 | 0.194 | `3ff0151f` |

Note the validation loss does **not** rank the folds: fold 4 has the lowest loss and fold 2 the
highest, yet fold 0 scores best and fold 2 worst. The loss is measured on lesion-centred crops with
prompts present, so it is not comparable to whole-volume Dice on a full split.

Reproduce any row with:

```bash
bs-evaluate --fold <n> --data-root /path/to/prepared --out-dir runs/eval<n>
```
