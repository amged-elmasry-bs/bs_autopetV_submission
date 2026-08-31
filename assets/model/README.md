# Model weights

**Nothing in this directory is stored in git** — it holds instructions only. The weights are
downloaded and converted locally, and `.gitignore` excludes them.

Two different sets exist. Which you need depends on what you are doing:

| you want to | you need | goes to |
|---|---|---|
| run the model — `bs-segment`, `bs-predict`, `bs-evaluate` | [trained checkpoints](#1-trained-fold-checkpoints) | `assets/model/fold<n>.pt` |
| train it yourself — `bs-train` | [initialisation weights](#2-per-fold-initialisation-weights) | `assets/model/pretrained_fold<n>.pt` |

Both are released under **CC-BY-4.0**, which is not this repository's Apache 2.0. See
[NOTICE](../../NOTICE) for the required attribution.

## 1. Trained fold checkpoints

The five cross-validation folds of the submitted model, produced by this repository.

**Zenodo: [10.5281/zenodo.22176946](https://doi.org/10.5281/zenodo.22176946)** — cite this version
DOI for reproducibility. The concept DOI
[10.5281/zenodo.22176945](https://doi.org/10.5281/zenodo.22176945) always resolves to the newest
version instead.

One archive, `Models Weights.zip` — 1,900,950,213 bytes, md5 `5eb2aaa77d632577be08b53b12e86bd7`.
The name contains a space, so quote it everywhere:

```bash
unzip "Models Weights.zip"
# -> Models Weights/model_checkpoint_fold_0.pt ... model_checkpoint_fold_4.pt
```

| file | bytes | md5 |
|---|---|---|
| `model_checkpoint_fold_0.pt` | 409740326 | `088dd2a8b70dd904f9ec4c0b72cbbae4` |
| `model_checkpoint_fold_1.pt` | 409741994 | `7a6bd366804bf6615c64517138f872fe` |
| `model_checkpoint_fold_2.pt` | 409741994 | `af40f30523b10f72bb6abd44ff0a4c53` |
| `model_checkpoint_fold_3.pt` | 409741994 | `06de9900d0cefdf16d266a27ef1adfbb` |
| `model_checkpoint_fold_4.pt` | 409741438 | `3ff0151fa2b621d5b4271f6b7a48b9f1` |

Each carries 966 tensors and 102.4M parameters over four input channels, and all five were checked
to load through `BS_Submission.model.build_model`.

### Using them

The released filenames differ from the `fold<n>.pt` that `--fold` resolves. Copy them into place
once and `--fold` then pairs each checkpoint with the right validation split automatically:

```bash
for n in 0 1 2 3 4; do
    cp "Models Weights/model_checkpoint_fold_${n}.pt" assets/model/fold${n}.pt
done

bs-segment  --ct ct.nii.gz --pet pet.nii.gz --fold 0 1 2 3 4 --out mask.nii.gz
bs-predict  --fold 0 --data-root /data/prepared --out-dir runs/pred0
bs-evaluate --fold 0 --data-root /data/prepared --out-dir runs/eval0
```

Or point at a checkpoint directly, in which case you must name the split yourself:

```bash
bs-predict --checkpoint "Models Weights/model_checkpoint_fold_0.pt" \
           --val-csv splits/fold_0_val.csv \
           --data-root /data/prepared --out-dir runs/pred0
```

**Score each fold only on its own validation split.** A checkpoint trained on fold *n* has seen
every other fold's validation cases, so scoring it there is meaningless. `--fold` enforces the
pairing; `--checkpoint` does not, which is why the explicit form above also passes `--val-csv`.

## 2. Per-fold initialisation weights

Training a fold starts from the weights of the **same** fold, so the initialisation has never seen
that fold's validation cases. `--fold <n>` resolves them by convention:

```
assets/model/pretrained_fold<n>.pt
```

Override with `--checkpoint-path`, change the directory with `--pretrained-dir`, or pass
`--checkpoint-path none` to train from scratch.

These derive from the per-fold checkpoints of **autoPET III LesionTracer** (Rokuss, DKFZ):

- Download: https://zenodo.org/records/14007247
  (`autoPET-3-LesionTracer.zip`, 3.8 GB, md5 `566016409b0bd14770c0b57c1f2873f1`)
- Inside it: `Dataset222_AutoPETIII_2024/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3/fold_<n>/checkpoint_final.pth`
- Code: https://github.com/MIC-DKFZ/autopet-3-submission
- Paper: arXiv:2409.09478

The published checkpoints take **two** input channels (CT, PET); this network takes four. Convert
them with:

```bash
python scripts/prepare_pretrained.py /path/to/autoPET3_Trainer__.../ assets/model
```

That keeps every trained weight and widens the stem convolution from two input channels to four,
initialising the two prompt channels to **zero**, so at step zero the network computes exactly what
the pretrained CT/PET model computed. Each output is 966 tensors, ~410 MB, and loads with
`strict=True`.

CC-BY-4.0 requires modifications to be indicated. This widening is the modification, and it is
recorded in [NOTICE](../../NOTICE).

**Fold alignment is verified.** The splits here use the same fold assignment as the upstream
release, so fold *n*'s initialisation never trained on the cases fold *n* validates on. Confirm it
yourself — each upstream `fold_<n>/validation/summary.json` lists the cases that fold validated:

```bash
python scripts/verify_fold_alignment.py /path/to/autoPET3_Trainer__.../
```

```
fold    upstream   ours   identical
0            323    323        True
1            322    322        True
2            322    322        True
3            322    322        True
4            322    322        True
```

Worth re-running if the splits are ever regenerated: were the assignments to diverge, each fold
would initialise from weights already trained on its own validation cases, and the validation
scores would be quietly optimistic.
