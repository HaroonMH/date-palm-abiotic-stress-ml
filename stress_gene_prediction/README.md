# Date Palm Stress-Responsive Gene Prediction

A machine learning pipeline for genome-wide prediction of stress-responsive
genes (SRGs) in date palm (*Phoenix dactylifera*) across four abiotic
stresses: **cold, heat, drought, and salt**.

The pipeline builds curated, non-redundant training sets from GO annotation,
encodes protein sequences with six composition-based descriptors, selects
features with SVM-RFE, benchmarks six classifiers, and applies the best model
per stress to the full proteome (48,801 proteins) to rank novel candidate
genes and discover discriminative sequence motifs.

## Pipeline overview

```
GO annotation ─► single-stress categorization ─► CD-HIT (0.8) ─► positive/negative sets ─► 7:3 split
                                                                                        │
     ┌──────────────────────────────────────────────────────────────────────────────────┘
     ▼
iFeatureOmega encodings (6) ─► encoding × classifier screening ─► SVM-RFE ─► 6 classifiers ─► tuned model per stress
                                                                                        │
     ┌──────────────────────────────────────────────────────────────────────────────────┘
     ▼
proteome-wide scoring ─► margin ranking + isoform collapsing ─► annotated candidates ─► MEME motif discovery
```

## Repository structure

```
stress_gene_prediction/
  scripts/     pipeline scripts, numbered in execution order
  data/        inputs and all generated outputs
```

## Installation

System dependency:

```bash
sudo apt-get install cd-hit
```

Python environment (tested on Python 3.12):

```bash
cd stress_gene_prediction
python3 -m venv .venv && source .venv/bin/activate
pip install networkx numpy pandas scikit-learn scipy matplotlib xgboost biopython rdkit joblib
pip install --no-deps iFeatureOmegaCLI
```

`iFeatureOmegaCLI` hard-pins an old `matplotlib` release that fails to build on
Python 3.12, so its other dependencies are installed first and the package
itself is installed with `--no-deps`.

### MEME Suite (optional, for motif discovery)

```bash
cd ~
wget https://meme-suite.org/meme/meme-software/5.5.7/meme-5.5.7.tar.gz
tar zxf meme-5.5.7.tar.gz && cd meme-5.5.7
./configure --prefix=$HOME/meme --enable-build-libxml2 --enable-build-libxslt \
            --enable-serial
make && make install
echo 'export PATH=$HOME/meme/bin:$HOME/meme/libexec/meme-5.5.7:$PATH' >> ~/.bashrc
source ~/.bashrc
```

Build notes:

- `--enable-serial` avoids an MPI build failure (`mpi.h: No such file or
  directory`) on systems with `openmpi-bin` but without `libopenmpi-dev`. MPI
  offers no benefit at the dataset sizes used here.
- The 5.5.7 tarball is missing `doc/generate_news_txt.pl`, so `make` fails in
  `doc/` after all binaries have built. Run `touch doc/news.txt` and rerun
  `make`; this only affects a bundled release-notes file.

If MEME is not installed locally, the web server
(https://meme-suite.org) can be used with the FASTA pairs produced by the
pipeline (Discriminative mode, protein alphabet, 3 motifs, widths 6–10).

## Inputs

- `protein.faa` — date palm reference proteome
- GO annotation of the proteome (literature-derived plus BLAST-homology GO
  transfer from *Arabidopsis thaliana*)
- *(optional)* NCBI genome annotation, used to build a protein → GeneID map

## Usage

All steps are run from `stress_gene_prediction/scripts` with the virtual
environment active. Each script reads the previous step's output.

```bash
cd stress_gene_prediction/scripts
source ../.venv/bin/activate
```

### 1. Dataset construction

```bash
python3 01_categorize_and_extract.py   # single-stress categorization, sequence extraction
bash    02_run_cdhit.sh                # redundancy removal, identity threshold 0.8
python3 03_build_pos_neg_split.py      # positive/negative sets, stratified 7:3 split
```

### 2. Feature extraction, selection, and model training

```bash
python3 04_extract_features.py          # CKSAAP, DDE, DPC, TPC, PAAC, APAAC
python3 05_screen_feature_methods.py    # 6 encodings × 6 classifiers, 5-fold CV
python3 06_svm_rfe_selection.py         # SVM-RFE on top-2 encodings + their concatenation
python3 07_train_evaluate_classifiers.py
python3 08_finalize_best_models.py      # hyperparameter tuning, final model per stress
```

Classifiers: SVM, XGBoost, Random Forest, Gradient Boosting, AdaBoost, Bagging.

### 3. Genome-wide prediction and candidate analysis

```bash
python3 14_genomewide_rescore.py        # score full proteome (chunked, resumable)
bash    14b_build_gene_map.sh           # optional: protein → GeneID map
python3 15_rank_candidates_v2.py        # margin ranking, isoform collapsing, annotation
bash    12_meme_motif_analysis.sh       # discriminative motif discovery
```

`14_genomewide_rescore.py` and `14b_build_gene_map.sh` are independent and can
run in parallel. Proteome scoring is chunked and resumable; it is safe to
interrupt and rerun.

## Outputs

```
data/
  summary_step1.tsv                  per-stress candidate / single-stress / missing counts
  raw/{stress}_raw.fasta             single-stress sequences, pre-CD-HIT
  nr/{stress}_nr.fasta               non-redundant positive set
  final/{stress}_{train,test}.fasta
  final/{stress}_labels.tsv          ProteinID, Label (1/0), Split
  dataset_summary.tsv                dataset composition per stress

  features/{stress}_{method}_{split}.csv
  screening_results.tsv              CV auROC/auPRC, 6 encodings × 6 classifiers
  top2_feature_methods.tsv
  rfe/{stress}_{feature_set}_selected_features.txt
  rfe/{stress}_{feature_set}_rfe_curve.tsv
  rfe_summary.tsv
  table2_results.tsv                 all stress × feature set × classifier results
  table3_best_models.tsv             best configuration per stress
  roc_data/{stress}_{feature_set}_{classifier}.tsv
  final_models/{stress}_model.joblib
  final_models/{stress}_meta.json    feature set, selected columns, hyperparameters
  final_model_metrics.tsv

  genomewide_v2/
    all_predictions.tsv              ProteinID, {stress}_prob, {stress}_margin
    datepalm_gene_map.tsv            ProteinID → GeneID, Symbol
    {stress}_ranked_proteins.tsv
    {stress}_ranked_genes.tsv        isoform-collapsed ranking
    {stress}_top10.tsv               top distinct genes with annotation
    {stress}_top10_isoform_members.tsv
    table4_top10_candidates.tsv      top candidates, all stresses
    srg_counts.tsv                   protein- and gene-level SRG counts

  meme/{stress}_{positive,negative}.fasta
  meme/{stress}_meme_out/meme.html
```

## Methodological notes

### Stress categorization

Each stress is defined by a small set of GO terms specific to that stress.
Terms shared across stresses (e.g. *response to osmotic stress*, *response to
temperature stimulus*, *response to abscisic acid*, *oxidative stress*,
*defense response*) are excluded because they do not discriminate between
the four targets. Proteins annotated to more than one target stress are
removed entirely. The exact GO ID lists are in the `STRESS_GO` dictionary in
`01_categorize_and_extract.py`.

For each stress, the negative set consists of proteins responsive to the
other three stresses.

### Evaluation

SVM-RFE uses a linear SVM, so the selected feature subsets are optimized for
SVM on the training data. Cross-validated SVM scores on those subsets are
therefore optimistically biased; the remaining classifiers do not participate
in selection and their CV scores are unaffected. Model selection and all
reported performance use the held-out test set (`test_accuracy`,
`test_auROC`, `test_auPRC`), which is never seen during feature selection.

### Candidate ranking

- **Unlabeled set.** For each stress, candidates are drawn from all proteins
  except that stress's known single-stress genes. Genes known for a
  different stress remain in the pool, so a known salt-responsive gene can
  still be proposed as a cold candidate.
- **Ranking by decision margin.** SVM probabilities from Platt scaling
  saturate near 1.0, producing large ties among top candidates. Candidates
  are therefore ranked by the raw SVM decision margin, which preserves the
  model's ordering at full resolution; probabilities are reported alongside
  for interpretability.
- **Isoform collapsing.** Multiple isoforms of the same gene are collapsed to
  the highest-scoring representative. Grouping uses NCBI GeneID when
  `datepalm_gene_map.tsv` is available; otherwise it falls back to product
  descriptions, which can merge distinct paralogs sharing a description and
  thus undercount genes. The mode used is recorded in `srg_counts.tsv`.

## Limitations

Because the negative class contains only genes responsive to other stresses,
the models are not trained to reject proteins unrelated to stress, and a
substantial fraction of the proteome scores above 0.5. The scores should be
interpreted as a relative ranking of candidates rather than a calibrated
estimate of genome-wide SRG prevalence. Stringent thresholds, or thresholds
set from training-set base rates, are recommended when selecting candidates.
Adding sampled non-stress proteins as an additional negative class is a
natural extension.
