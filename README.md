# DAGISASM: Mitigating Demographic Bias in Deepfake Image Detection

Code and results for the COS700 research report *Mitigating Demographic Bias in Deepfake Image Detection* (University of Pretoria, 2026).

The project audits a multimodal deepfake detector across three FairFace-predicted groups: Black, Indian and White. The detector combines a Swin-Tiny image stream with a Fourier-spectrum stream. It is tested in two ways:

- **In-distribution test:** held-out FFHQ real faces and StyleGAN fake faces from the 140k corpus.
- **Cross-source/cross-generator test:** IMDb-WIKI real faces and Stable Diffusion 1.5 fakes from AI-Face.

The baseline model is then compared with a fairness-aware training recipe: the DAW-FDD loss, a demographic adversary, and model selection on validation AUC minus the FPR gap.

## Repository layout

```
config/                    training settings (YAML)
  model140k_swin_aug.yaml  final models in the report
  model140k_{swin,vit,xception}.yaml   preliminary backbone comparison
scripts/                   one script per pipeline step (run from the repo root)
src/sadf/                  package: data loading, model, losses, trainer, metrics
figures.py                 chart figures of the report
figures/                   generated charts (PDF and PNG)
figures/tikz/              TikZ sources of report Figures 1, 4 and 6
runs/models140k/           saved results: training histories, evaluation reports, per-image scores
```

Datasets, FairFace weights and model checkpoints are not stored in the repository (see `.gitignore`).

## Setup

Python 3.10 or newer.

```
pip install -r requirements.txt
```

OpenCV must be version 4 (`opencv-python-headless<5`), because OpenCV 5 moved the Haar cascade into the contrib package.

A GPU is not needed. On the CPU used for the report, one training epoch of the final model took about 80 to 85 minutes, so each final model takes about 5.5 hours to train.

All commands below are run from the repository root. Forward slashes work on Windows too.

## Data and weights

Place the data as follows. The scripts expect exactly these paths.

| What | Where to get it | Where to put it |
| --- | --- | --- |
| 140k Real and Fake Faces (70 000 FFHQ real, 70 000 StyleGAN fake, 256 x 256) | Kaggle, `xhlulu/140k-real-and-fake-faces` | `data/raw/real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}/*.jpg` |
| AI-Face: IMDb-WIKI real faces | [AI-Face](https://github.com/Purdue-M2/AI-Face-FairnessBench) (Lin et al., CVPR 2025) | `data/raw/imdb_wiki/<NN>/nm..._face1.jpg` |
| AI-Face: Stable Diffusion 1.5 fakes (wiki subset) | same | `data/raw/StableDiffusion1.5_wiki_processed_process/<NN>/*.png` |
| FairFace 7-race model `res34_fair_align_multi_7_20190809.pt` | [FairFace](https://github.com/joojs/fairface) (Kärkkäinen and Joo, WACV 2021) | `models/fairface/` |

With the Kaggle command-line tool, the 140k corpus unpacks into the right place:

```
kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/raw --unzip
```

## Reproducing the results

### 1. FairFace race labels for the 140k corpus

```
python scripts/build_140k_race.py --phase label \
    --root data/raw/real_vs_fake/real-vs-fake \
    --weights models/fairface/res34_fair_align_multi_7_20190809.pt
```

The 140k images are already aligned, face-centred crops. FairFace therefore runs on them directly: no face detection is applied, and the images are resized to 224 x 224 with ImageNet normalisation.

This step writes:

- `data/faces140k/manifests/raw_race_labels.csv`, with one row for each of the 140 000 images. The seven FairFace groups are merged into Black, Indian, White and Other.
- A random sample of 30 faces per group (seed 0), copied to `data/faces140k/_race_check/` for the visual check.

Expected composition: White 88 496, Other 41 165, Black 5 975, Indian 4 364.

### 2. Label the AI-Face test sources and merge

```
python scripts/merge_new_sources.py \
    --pool     data/faces140k/manifests/raw_race_labels.csv \
    --sd-dir   data/raw/StableDiffusion1.5_wiki_processed_process \
    --imdb-dir data/raw/imdb_wiki \
    --weights  models/fairface/res34_fair_align_multi_7_20190809.pt \
    --out      data/faces140k/manifests/pool_merged.csv
```

### 3. Split first, then augment the training split

```
python scripts/build_augmented_140k.py --manifest data/faces140k/manifests/pool_merged.csv --dry-run
python scripts/build_augmented_140k.py --manifest data/faces140k/manifests/pool_merged.csv
```

The defaults are the settings used in the report:

- 400 test and 300 validation images per race x label cell.
- Training cells brought up to 4 000 images each.
- Real and fake matched within each race (`--aug-balance within-race`).
- Split and augmentation seed 1337.

This step writes `data/faces140k/manifests/manifest_aug.csv` and the augmented PNGs in `data/faces140k/augmented/`. Expected result: 28 200 rows from 18 112 distinct images.

| | Training | Validation | Test |
| --- | --- | --- | --- |
| Rows | 24 000 (13 912 originals and 10 088 augmented copies) | 1 800 | 2 400 |

The script stops if any source image ends up in more than one split.

### 4. Train the two models

```
python scripts/run_models140k.py --train swin --recipe baseline --config config/model140k_swin_aug.yaml
python scripts/run_models140k.py --train swin --recipe fair     --config config/model140k_swin_aug.yaml
```

Both models use seed 42 and train for at most 4 epochs, with early stopping after 2 epochs without improvement. The settings are in `config/model140k_swin_aug.yaml`. Checkpoints are saved as `runs/models140k/swin_aug/<recipe>/best.pt`, and the validation metrics of every epoch are saved in `history.json`.

To train with another seed, copy the config, change `seed`, **and change `paths.runs`**. Otherwise the new run overwrites the reported checkpoints.

Training is deterministic on the same machine and software. After the original baseline checkpoint was overwritten by an unfinished extra-seed run, the baseline was retrained with this config, and every epoch reproduced the saved `history.json` exactly.

### 5. In-distribution test

```
python scripts/run_models140k.py --eval swin --recipe baseline --config config/model140k_swin_aug.yaml
python scripts/run_models140k.py --eval swin --recipe fair     --config config/model140k_swin_aug.yaml
```

The output goes to `runs/models140k/swin_aug/<recipe>/eval/`.

### 6. Cross-source/cross-generator test

```
python scripts/eval_cross_generator.py \
    --checkpoint runs/models140k/swin_aug/baseline/best.pt \
    --manifest data/faces140k/manifests/pool_merged.csv \
    --out runs/models140k/swin_aug/baseline/eval_cross \
    --cap-per-identity 5
```

Repeat with `fair` in place of `baseline`.

- IMDb holds only real faces, so it gives the FPR of each group.
- Stable Diffusion holds only fakes, so it gives the FNR.
- `combined/` pools the two sources. Its AUC is not meaningful, because the source predicts the label, and its accuracy depends on the 6 499 real / 14 177 fake mix of this set.

### 7. Figures

```
python figures.py
```

This regenerates the chart figures in `figures/` from the counts in its DATA section. Report Figures 1, 4 and 6 are compiled from `figures/tikz/` with the report.

### Preliminary backbone comparison (optional)

Before the main experiments, ViT-Small, Xception and Swin-Tiny were compared on a smaller balanced set. The report mentions this comparison only as the reason for choosing Swin-Tiny, not as evidence.

```
python scripts/build_140k_race.py --phase balance --per-group 3000 --drop-other
python scripts/run_models140k.py --train vit
python scripts/run_models140k.py --train xception
python scripts/run_models140k.py --train swin
python scripts/run_models140k.py --eval-all
python scripts/run_models140k.py --compare
```

### ITA skin-tone check (optional)

ITA was the first labelling method and was dropped. This script measures ITA on a sample of 140k images and saves example crops per tone group for a visual check:

```
python scripts/check_140k_tone.py --root data/raw/real_vs_fake/real-vs-fake --sample 2000
```

## Where the reported numbers come from

Every evaluation folder keeps `scores.csv`, which holds the detector's score for each image. The metrics can therefore be recomputed without the checkpoints.

| Report item | File |
| --- | --- |
| In-distribution results (baseline), FPR gap and its interval | `runs/models140k/swin_aug/baseline/eval/per_group_skin_tone.csv`, `gaps.json` |
| In-distribution results (fairness-aware) | `runs/models140k/swin_aug/fair/eval/` |
| Cross-source/cross-generator rates and intervals per group, number of IMDb identities | `runs/models140k/swin_aug/<recipe>/eval_cross/{imdb,sd_wiki}/per_group_error.csv` |
| Pooled accuracy and equal-weight (balanced) accuracy | `runs/models140k/swin_aug/<recipe>/eval_cross/combined/overall.json` |
| Validation AUC, FPR gap and selection score per epoch | `runs/models140k/swin_aug/<recipe>/history.json` |
| Intervals for the gaps and the changes in gaps | `figures.py` (bootstrap from the counts, 2 000 resamples, seed 0) |
| Preliminary backbone comparison | `runs/models140k/{swin,vit,xception}/baseline/` |

## Notes

- **Group labels.** The labels are FairFace predictions of perceived race, not self-reported groups. For historical reasons, the column that holds them is called `skin_tone` throughout the code.
- **Random seeds.**

  | Seed | Value |
  | --- | --- |
  | Training | 42 |
  | Split and augmentation | 1337 |
  | Visual-check sample | 0 |
  | Identity cap and evaluation bootstrap | 42 |
  | `figures.py` bootstrap | 0 |

- **Load-time flip.** During training, each image is also flipped horizontally at random when it is loaded (`src/sadf/data/dataset.py`).
