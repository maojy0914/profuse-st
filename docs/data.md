# Input preparation

This repository starts from cached morphological, whole-section, and gene
features. It does not duplicate the separate foundation-model environments or
the dataset-specific acquisition scripts from the research workspace.

## Per-section files

Each section has a directory named by its section ID. Training reads:

| File | Keys and shapes |
|---|---|
| `gene_expression_ground_truth.pkl` | `expression`: float `[N,G]`; `barcodes`: `N` strings; `genes`: `G` strings; `normalization`: description |
| `feature_uni.h5` | `embeddings`: float `[N,D]`; `barcodes`: strings `[N]` or `[N,1]`; `coords`: float `[N,2]` |
| `slide_feature_gigapath.h5` | `features`: float `[768]` or `[1,768]` |
| `graph_faiss.pt` (optional) | `spatial_neighbors`, `semantic_neighbors`: integer `[N,K]`; optional `barcodes` in expression order |

Load only expression pickle files from a trusted source. The loader reorders
image features by expression barcode, and rejects missing/duplicate identifiers
or inconsistent gene order across sections. During inference, image-feature
barcode order becomes the output order; no expression file is loaded.

`D`, the slide feature dimension, and the gene embedding dimension are inferred
from the training inputs. The original defaults are 1,024-dimensional UNI
features, 768-dimensional GigaPath context, and 200-dimensional Gene2vec vectors.

## Preprocessing contract

1. Retain tissue-associated spots and match them to image coordinates.
2. Select one fixed target-gene panel for the cohort. Ensure every target gene
   has a Gene2vec vector, and preserve exactly the same ordered gene list across
   sections. The loader never silently drops genes.
3. Normalize counts to a total of `1e4` per spot, then apply `log1p`. In the
   original datasets the normalization denominator used all measured genes for
   DLPFC and the selected target genes for cSCC/HER2ST. Record the actual convention
   in the expression payload. Do not normalize the already processed labels again.
4. Crop spot-centered 224×224 RGB patches at registered image coordinates and
   apply ImageNet channel normalization before frozen UNI encoding.
5. Encode tissue tiles across the section with GigaPath's corresponding tile
   encoder, then aggregate them with its pretrained slide-level encoder. Do not
   substitute UNI tile vectors as inputs to a GigaPath slide encoder. Store the
   resulting section embedding once, not once per spot.
6. Supply the pretrained Gene2vec text file (gene symbol followed by vector
   values). Both word2vec-header and headerless variants are accepted.

For rigorous transfer evaluation, decide any learned gene selection or
preprocessing parameters using the permitted training data, or declare a
predefined benchmark panel. The training loader consumes the chosen panel; it
does not fit a new selection on held-out expression.

## Graphs

Existing `graph_faiss.pt` files can be reused when their spot set matches the
loaded section; barcode metadata allows a different row order to be remapped.
Only the spatial and semantic neighbor
indices are used by the model; historical graph bias arrays are unnecessary.

Without a cache, exact kNN is built in memory using PyTorch: Euclidean distance
for spatial edges and cosine similarity for semantic edges, excluding each
spot itself. Query rows are processed in chunks to limit memory. Distance ties
can produce different neighbor orderings from FAISS. This avoids a FAISS
dependency but retains quadratic graph-construction time; large slides benefit
from preparing a cache separately.

For within-slide subsets, cached graphs are restricted and remapped to that
subset. Missing neighbors are padded with the current node, matching the
original dataset adapter. If no cache exists, graphs are rebuilt within the
subset. These two conventions need not yield identical results; reuse the
same graph convention across runs. Coordinate standardization uses the spots
passed to each forward call, also matching the original adapter.

## Feature sources

- [UNI](https://github.com/mahmoodlab/UNI): frozen spot-patch encoder.
- [GigaPath](https://github.com/prov-gigapath/prov-gigapath): corresponding tile and slide encoders for whole-section context.
- [Gene2vec](https://github.com/jingcheng-du/Gene2vec): pretrained co-expression-based gene vectors.
- [DLPFC](https://research.libd.org/spatialLIBD/): spatial cortical sections.
- [HER2ST](https://github.com/almaan/her2st): HER2-positive breast cancer data.
- cSCC: *Multimodal Analysis of Composition and Spatial Architecture in Human Squamous Cell Carcinoma*, [original study](https://doi.org/10.1016/j.cell.2020.05.039).

Obtain external data and model weights from their original distributions under
their respective terms. This repository contains neither.
