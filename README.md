# Predicting Abiotic Stress-Responsive Genes in Date Palm (*Phoenix dactylifera* L.)

A machine learning pipeline that predicts date palm genes responsive to four
abiotic stresses — **cold, heat, drought and salinity** — directly from protein
sequence, and applies the resulting models genome-wide to prioritise novel
candidate genes.

The methodology follows You et al. (2025), *MLAS: Machine Learning-Based
Approach for Predicting Abiotic Stress-Responsive Genes in Chinese Cabbage*
(**Horticulturae** 11, 44), adapted to *Phoenix dactylifera*.

## Results at a glance

Final tuned model per stress, evaluated on a held-out test set:

| Stress | Features | Model | Accuracy | auROC | auPRC |
|---|---|---|---|---|---|
| Cold | TPC | SVM | 81.76% | 80.14% | 67.94% |
| Heat | TPC | SVM | 81.11% | 85.81% | 71.73% |
| Drought | TPC | SVM | 83.33% | 83.76% | 71.23% |
| Salt | CKSAAP+TPC | SVM | 78.18% | 84.28% | 77.90% |

Applied across all 48,801 date palm proteins (29,239 genes), the models recover
canonical stress genes that were absent from training — heat shock
transcription factors and HSP81 chaperones under heat, a sodium/hydrogen
exchanger (NHX1) and WRKY32 under salinity, and DEHYDRATION-INDUCED 19 paralogs
under drought — providing independent biological validation.

## Pipeline

| Milestone | Scripts | Output |
|---|---|---|
| 1. Dataset curation | `01`–`03` | Single-stress GO categorisation, CD-HIT redundancy removal (0.8), positive/negative sets, 7:3 split |
| 2. Modelling | `04`–`09` | Six sequence encodings × six classifiers, SVM-RFE feature selection, tuning, figures |
| 3. Genome-wide prediction | `10`–`13` | Whole-proteome scoring, ranked candidates, MEME motifs |
| 3b. Corrected ranking | `14`–`16` | Margin-based ranking, isoform collapsing by NCBI GeneID |
| 4. Manuscript | `17` | Word manuscript with all tables and figures embedded |

Full run instructions, prerequisites and per-step notes are in
[`stress_gene_prediction/README.md`](stress_gene_prediction/README.md).

## Repository layout

```
stress_gene_prediction/
  scripts/          all pipeline code (01-17)
  data/
    raw/ nr/ final/     curated FASTA datasets and labels
    rfe/                selected feature lists and RFE curves
    genomewide_v2/      ranked predictions, top-10 candidates, GeneID map
    meme/               discriminative motif discovery output
    figures/ figures_v2/  publication figures (PNG + PDF)
    *.tsv               result tables
  manuscript/       generated manuscript (.docx)
```

## Reproducing

Source data (the date palm proteome `protein.faa` and the NCBI annotation
files) is excluded from this repository for size. Feature matrices and trained
model binaries are likewise excluded as regenerable — rerun `04` and `08` to
recreate them.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install networkx numpy pandas scikit-learn scipy matplotlib \
            xgboost biopython rdkit joblib python-docx
pip install --no-deps iFeatureOmegaCLI   # its pinned matplotlib fails on 3.12
```

External tools: `cd-hit` (apt) and MEME Suite (build from source with
`--enable-serial`; see the pipeline README).

## Notes on interpretation

- Reported metrics are **held-out test set** figures. Cross-validation scores
  for SVM are optimistically biased because SVM-RFE selected the features, and
  are deliberately excluded from the manuscript.
- Genome-wide scores are a **relative ranking** of stress-responsiveness, not
  calibrated prevalence estimates. The negative class consists of genes
  responsive to the other three stresses rather than a random genomic
  background, so the models were never trained to reject non-stress proteins.
  This is discussed as a limitation in the manuscript.
