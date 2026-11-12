# ProFuse-ST

**Prototype-enhanced fusion for histology-based spatial gene expression prediction.**

This compact implementation contains the full ProFuse-ST model, slide-level
training, expression prediction, and explicit dataset splits. It contains no
comparison methods, historical model versions, ablation runners, parameter
sweeps, or experiment outputs.

The model combines spatial and semantic graphs with a coordinate-aware
Transformer. A whole-section embedding is projected, broadcast, and used for
feature-wise affine modulation of the Transformer output. A learned gate fuses
local and global representations. Gene2vec embeddings attend to trainable
prototypes; spot-to-gene cross-attention then produces non-negative expression
predictions. Training uses MSE plus expression distribution alignment.

## Layout

```text
profuse-st/
├── configs/             # One compact starting configuration per dataset
├── docs/data.md         # Input schema, feature sources, preprocessing contract
├── profuse_st/
│   ├── model.py         # The complete ProFuse-ST architecture
│   ├── data.py          # Barcode alignment, gene embeddings, graph loading/building
│   ├── loss.py          # MSE + per-spot KL alignment
│   ├── metrics.py       # PCC, MSE, MAE, HVG/HEG subsets
│   ├── splits.py        # Generate one explicit evaluation fold
│   ├── engine.py        # Shared inference and output helpers
│   ├── train.py         # Training, validation selection, held-out evaluation
│   └── predict.py       # Expression-free inference
├── tests/               # Small synthetic correctness and workflow checks
└── pyproject.toml       # Package dependencies and command-line entry points
```

## Installation

Use Python 3.10 or later. Install the PyTorch build appropriate for your CPU/CUDA
environment, then install this project from its root:

```bash
pip install -e .
```

The prediction network only requires PyTorch, NumPy, h5py, and PyYAML. Foundation
model feature extraction runs separately; pretrained weights and datasets are
not distributed here. See [data preparation](docs/data.md).

## Data

The loader directly accepts the prepared files used by the original project:

```text
data/DLPFC/
└── 151507/
    ├── gene_expression_ground_truth.pkl
    ├── feature_uni.h5
    ├── slide_feature_gigapath.h5
    └── graph_faiss.pt                  # Optional; built in memory if absent
weights/gene2vec_dim_200_iter_9_w2v.txt
```

All sections in a run must use the same target genes, gene order, and expression
normalization. Data may stay outside this repository: use `--data-root` and
optionally `--feature-root`. Paths are supplied on the command line; no local
machine paths are embedded in the code.

## Training

Generate a cross-slide fold, then train it:

```bash
profuse-split --dataset DLPFC --protocol cross-slide \
  --test 151507 --val 151508 --output splits/dlpfc.json

profuse-train --config configs/dlpfc.yaml \
  --data-root data/DLPFC \
  --gene-embeddings weights/gene2vec_dim_200_iter_9_w2v.txt \
  --split splits/dlpfc.json --output outputs/dlpfc-151507
```

Use `configs/cscc.yaml` or `configs/her2st.yaml` for the other datasets. The
configuration files are transparent starting settings (32 prototypes,
alignment weight 0.25), not a claim to reproduce every tuned result in a paper.

One section is processed per optimizer step. The checkpoint with the lowest
validation MSE is selected, with early stopping. Test sections never select a
checkpoint. Each output directory contains:

- `best.pt`: model weights, constructor settings, ordered genes, and fixed gene embeddings.
- `config.json`, `split.json`: the exact settings and physical spot selections.
- `history.json`: training loss and validation MSE by epoch.
- `metrics.json`: per-section metrics and their section-wise means.
- `<section>.npz`: predictions, ordered genes, barcodes, and coordinates.

Existing nonempty run directories are rejected to prevent accidental overwrite.

## Other evaluation splits

```bash
# One held-out patient; the last section of every remaining patient is validation.
profuse-split --dataset DLPFC --protocol cross-patient \
  --test Human1 --output splits/dlpfc-human1.json

# Whole 4×4 spatial blocks, targeting 70/15/15 spot-count proportions.
profuse-split --dataset DLPFC --protocol intra-slide --test 151507 \
  --data-root data/DLPFC --output splits/dlpfc-intra.json
```

Pass either manifest to the same training command. Each command creates **one
fold**. Complete cross-validation requires one independent run for each held-out
section or patient. The built-in patient groups cover DLPFC, cSCC, and HER2ST.
For another cohort, supply a JSON manifest with `train`, `val`, and `test` slide
lists. Within-slide manifests additionally include `spot_indices` indexed by
split name and slide ID, containing barcodes or integer row indices. The loader
checks that train, validation, and test contain disjoint physical spots.

## Prediction

Predict on a new section using only its image features and coordinates:

```bash
profuse-predict --checkpoint outputs/dlpfc-151507/best.pt \
  --data-root data/new_sections --slides new_section \
  --output outputs/new_predictions
```

The checkpoint supplies the fixed gene embeddings and gene order. An expression
pickle is unnecessary at inference. Outputs are in the same preprocessed
expression scale as the training targets, not raw RNA counts. This entry point
loads checkpoints produced by this compact package; importing historical model
weights requires matching their architecture and constructor parameters.

## Validation

```bash
python -m unittest discover -s tests -v
```

Metrics rank HVGs and HEGs by variance and mean of the evaluated observed
expression, respectively. Undefined gene correlations are excluded. If a panel
has fewer than 50 or 200 genes, all available genes are used and the actual
subset size is reported. Keep the same evaluation gene sets across compared
runs. The exact historical benchmark lists may differ in tie handling.

## Sources

The implementation derives from the ProFuse-ST source, retaining its
parameter names and complete Transformer-based computation path. UNI, GigaPath,
and Gene2vec are external pretrained resources; see [their original sources](docs/data.md#feature-sources).
No third-party model implementation or pretrained weight is bundled here.
