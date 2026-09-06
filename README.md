# Predicting Abiotic Stress-Responsive Genes in Date Palm (*Phoenix dactylifera* L.)

Machine learning models that predict date palm genes responsive to four abiotic
stresses — **cold, heat, drought and salinity** — directly from protein
sequence, applied genome-wide to prioritise novel candidate genes.

The work proceeds in two stages. **Stage 1** reproduces the methodology of You
et al. (2025), *MLAS: Machine Learning-Based Approach for Predicting Abiotic
Stress-Responsive Genes in Chinese Cabbage* (**Horticulturae** 11, 44), on a new
species. **Stage 2** addresses a limitation that paper states in its own
Discussion — that a single gene may respond to several stresses, which its
four-separate-models design cannot represent — by reformulating the task as
multi-label learning over a protein language model representation.

## Results

### Stage 1 — single-stress models (k-mer features)

| Stress | Features | Model | Accuracy | auROC | auPRC |
|---|---|---|---|---|---|
| Cold | TPC | SVM | 81.76% | 80.14% | 67.94% |
| Heat | TPC | SVM | 81.11% | 85.81% | 71.73% |
| Drought | TPC | SVM | 83.33% | 83.76% | 71.23% |
| Salt | CKSAAP+TPC | SVM | 78.18% | 84.28% | 77.90% |

### Stage 2 — multi-label models (ESM-2 features)

Architecture ablation on identical data, partition and features (held-out
test set, auROC / auPRC):

| Stress | Per-stress SVM | Independent MLPs | Multi-task shared backbone |
|---|---|---|---|
| Cold | **86.89** / 43.21 | 81.68 / 34.47 | 83.85 / 38.31 |
| Heat | **83.36** / 36.54 | 79.87 / 21.76 | 82.46 / **38.97** |
| Drought | **87.27** / **46.55** | 84.04 / 39.94 | 80.43 / 44.49 |
| Salt | **85.63** / 38.02 | 78.02 / 37.00 | 78.73 / **42.00** |

The shared backbone improves auPRC over matched-capacity independent networks
for every stress (+23% on average, largest for heat, the stress with fewest
positives), confirming that representation sharing helps. Per-stress SVMs
nevertheless retain the best auROC throughout — classical margin-based learning
remains the stronger predictor at this data scale, even against a pretrained
protein language model backbone.

### What Stage 2 fixed

**Genome-wide calibration.** The Stage 1 negative class consists of genes
responsive to the *other* stresses, so those models were never trained to reject
ordinary non-stress proteins. Replacing it with confident negatives — proteins
carrying GO annotation but no stress term — corrected the predicted counts:

| Stress | Stage 1 | Stage 2 | Reference paper |
|---|---|---|---|
| Cold | 8,221 (28%) | **426 (1.5%)** | 240 (0.6%) |
| Heat | 5,054 (17%) | **261 (0.9%)** | 1,829 (4.5%) |
| Drought | 4,607 (16%) | **559 (2.0%)** | 2,129 (5.2%) |
| Salt | 6,671 (23%) | **699 (2.5%)** | 115 (0.3%) |

**Recovered training data.** The single-stress design discards every gene
responding to more than one stress — 735 genes, 24.1% of all labelled data,
including 239 drought+salt genes and 50 responding to all four. Retaining them
increased positives per stress by 55–89% despite stricter global redundancy
removal.

**Candidate biology.** Stage 2 recovers canonical stress genes absent from
training: three **CBL-interacting protein kinases** (the SOS pathway) for salt,
**HVA22** and MYB/WRKY factors for drought, and **HSFA2c**, three HSP81-1
chaperones and a **Clp protease** for heat.

**Co-response prediction.** Four independent sigmoid heads — rather than a
softmax, whose outputs would be forced to sum to one — allow a gene to be
predicted responsive to several stresses at once. 769 genes were predicted for
two or more, dominated by drought+salt, with VOZ1 and WRKY24 transcription
factors among those predicted for all four.

## Pipeline

| Stage | Scripts | Output |
|---|---|---|
| 1. Dataset curation | `01`–`03` | Single-stress GO categorisation, CD-HIT (0.8), 7:3 split |
| 2. Modelling | `04`–`09` | Six encodings × six classifiers, SVM-RFE, tuning, figures |
| 3. Genome-wide prediction | `10`–`13` | Whole-proteome scoring, ranked candidates, MEME motifs |
| 3b. Corrected ranking | `14`–`16` | Margin-based ranking, isoform collapsing by GeneID |
| 4. Manuscript | `17` | Word manuscript with tables and figures embedded |
| 5. Multi-label dataset | `20`–`22` | Multi-label targets, confident negatives, global leakage-free split |
| 6. PLM + multi-task | `23`–`25` | ESM-2 embeddings, shared-backbone network, ablation |
| 7. Genome-wide (Stage 2) | `26`, `28` | Calibrated predictions, candidates, co-response |
| 8. Figures & merge | `27`, `29` | Stage 2 figures, manuscript merge |

Scripts `01`–`17` live in [`stress_gene_prediction/`](stress_gene_prediction/README.md),
scripts `20`–`29` in [`multitask_stress_model/`](multitask_stress_model/README.md).
Both READMEs carry full run instructions and per-step notes.

## Repository layout

```
stress_gene_prediction/          Stage 1: single-stress k-mer pipeline
  scripts/                       code (01-17)
  data/
    raw/ nr/ final/              curated FASTA datasets and labels
    rfe/                         selected features and RFE curves
    genomewide_v2/               ranked predictions, top-10s, GeneID map
    meme/                        discriminative motif discovery
    figures/ figures_v2/         publication figures (PNG + PDF)
    *.tsv                        result tables

multitask_stress_model/          Stage 2: multi-label + ESM-2
  scripts/                       code (20-29)
  data/labels/ splits/           multi-label targets and the global split
  models/                        trained multi-task network
  outputs/tables/ figures/       results and figures
```

## Reproducing

Source data (`protein.faa` and the NCBI annotation files), ESM-2 embeddings,
feature matrices, model binaries and manuscript drafts are excluded for size or
because they are regenerable. Rerun scripts `04`, `08` and `23` to recreate them.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install networkx numpy pandas scikit-learn scipy matplotlib \
            xgboost biopython rdkit joblib python-docx
pip install --no-deps iFeatureOmegaCLI   # its pinned matplotlib fails on 3.12

# Stage 2 only
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install fair-esm
```

External tools: `cd-hit` (apt) and MEME Suite (build from source with
`--enable-serial`; see the Stage 1 README). Stage 2 embedding extraction needs
a CUDA GPU — developed on an RTX 3060 (6 GB) — and is resumable.

## Notes on interpretation

- All reported metrics are **held-out test set** figures. Cross-validation
  scores for SVM in Stage 1 are optimistically biased, because SVM-RFE selected
  the features using the same data, and are excluded from the manuscript.
- **auPRC is not comparable between stages.** It scales with the positive rate,
  which is ~26% in the Stage 1 test sets and 1.5–2.7% in Stage 2, where the
  negative class became the annotated proteome rather than other stress genes.
  Stage 2 evaluates a harder, more realistic task; compare lift over the random
  baseline, never raw auPRC.
- Stage 1 genome-wide scores are a **relative ranking**, not calibrated
  prevalence estimates. Stage 2 corrects this, which is why its counts fall in a
  biologically credible range.
- Labels derive from Arabidopsis GO transfer, so both single- and multi-label
  assignments inherit that annotation's errors and are biased toward families
  conserved between a monocot palm and a eudicot model. All predictions are
  computational hypotheses requiring experimental validation.
