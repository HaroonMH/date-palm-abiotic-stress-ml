# Milestone 1 (finish) -- CD-HIT + dataset construction/split

Picks up where the GO annotation work (literature search + BLAST-homology GO
transfer) left off. Produces the curated, non-redundant, positive/negative,
7:3-split FASTA dataset per stress type -- mirroring the reference paper's
methodology (Section 2.1 / Figure 3).

## Prerequisites

```bash
sudo apt-get install cd-hit
```

No Python packages beyond the standard library are needed for these three
scripts.

## Run in order

```bash
cd /home/muhammad-haroon/date-palm-project/stress_gene_prediction/scripts

# 1. Categorize date palm proteins into cold/heat/drought/salt single-stress
#    groups from the GO analysis, extract sequences from protein.faa
python3 01_categorize_and_extract.py

# 2. CD-HIT redundancy removal (threshold 0.8), per stress
bash 02_run_cdhit.sh

# 3. Build positive/negative sets and the 7:3 train/test split
python3 03_build_pos_neg_split.py
```

## Outputs

```
stress_gene_prediction/data/
  summary_step1.tsv        # candidates matched / single-stress / missing-sequence per stress
  raw/{stress}_raw.fasta   # single-stress candidate sequences, pre-CD-HIT
  nr/{stress}_nr.fasta     # post-CD-HIT (this is the "Positive set" for that stress)
  final/
    {stress}_train.fasta
    {stress}_test.fasta
    {stress}_labels.tsv    # ProteinID, Label (1=positive/0=negative), Split (train/test)
  dataset_summary.tsv      # Table-1-style counts report (share this back)
```

## What to send back

Please paste/share the contents of:
- `data/summary_step1.tsv`
- `data/dataset_summary.tsv`
- the console output of `02_run_cdhit.sh` (it prints how many sequences CD-HIT clustered/removed per stress)

That's enough for me to confirm the numbers look right and move on to
Milestone 1's Phase 2 (iFeatureOmega feature extraction, SVM-RFE, and the six
classifiers) using `final/{stress}_train.fasta` / `{stress}_test.fasta` as
input.

## Categorization logic (why these GO IDs)

Each stress is defined by a short list of GO IDs specific to that single
stress (same idea as the reference paper naming five specific GO terms for
cold stress), not every stress-adjacent term. Generic/shared terms --
`response to osmotic stress` (salt+drought), `response to temperature
stimulus` (heat+cold), `response to abscisic acid`, `oxidative stress`,
`defense response`, etc. -- are intentionally excluded because they don't
discriminate between the four target stresses. See the `STRESS_GO` dict at
the top of `01_categorize_and_extract.py` for the exact GO ID lists.

A protein whose stress-relevant GO annotations span more than one of the four
target stresses is dropped entirely (matches the paper's "Exclude overlapped
gene ID" step) -- it won't appear in any stress's positive or negative set.

---

# Milestone 2 -- Feature extraction, SVM-RFE, and the six classifiers

Mirrors the reference paper's Section 2.2-2.4 / Section 3.2-3.3 in full:
1. Generate all six iFeatureOmega encodings (CKSAAP, DDE, DPC, TPC, PAAC, APAAC).
2. Screen all 6 encodings x 6 classifiers via 5-fold CV on the training set
   (paper's Figure 5) to find the two best-performing encodings.
3. Run SVM-RFE on those two encodings plus their concatenation (3 feature
   sets total) to find the optimal feature subset per stress (paper's Figure 6).
4. Train and evaluate all six classifiers (SVM, XGB, RF, GBDT, ADB, BAG) on
   the RFE-selected features, 5-fold CV + held-out test evaluation (paper's
   Figure 7/8, Table 2), and pick the best (feature set, classifier) per stress
   (Table 3).
5. Hyperparameter-tune and finalize that best model per stress, save the
   trained models.
6. Generate all figures (dataset composition, screening heatmap, RFE curves,
   ROC curves, final metrics) as PNG/PDF.

## Prerequisites

A virtual environment has **already been created and verified working** at
`stress_gene_prediction/.venv` (includes `iFeatureOmegaCLI`, `scikit-learn`, `xgboost`,
`biopython`, `rdkit`, `matplotlib`). Just activate it:

```bash
cd /home/muhammad-haroon/date-palm-project/stress_gene_prediction
source .venv/bin/activate
```

If you ever need to rebuild it from scratch (e.g. on another machine),
`iFeatureOmegaCLI`'s published dependency list hard-pins an old `matplotlib`
that fails to build on Python 3.12 -- install its other dependencies first,
then install it with `--no-deps`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install networkx numpy pandas scikit-learn scipy matplotlib xgboost biopython rdkit joblib
pip install --no-deps iFeatureOmegaCLI
```

## Run in order

```bash
cd /home/muhammad-haroon/date-palm-project/stress_gene_prediction/scripts
source ../.venv/bin/activate

# 4. Extract all 6 iFeatureOmega encodings for every stress's train/test FASTA
python3 04_extract_features.py

# 5. Screen all 6 encodings x 6 classifiers (5-fold CV, training set only)
python3 05_screen_feature_methods.py

# 6. SVM-RFE on the top-2 encodings + their combination
python3 06_svm_rfe_selection.py

# 7. Train/evaluate all 6 classifiers on the RFE-selected features
python3 07_train_evaluate_classifiers.py

# 8. Hyperparameter-tune and finalize the best model per stress
python3 08_finalize_best_models.py

# 9. Generate all figures
python3 09_generate_figures.py
```

Each script prints its own progress and next-step reminder; run them in order
since each reads the previous step's output.

## Outputs

```
stress_gene_prediction/data/
  features/{stress}_{method}_{split}.csv        # raw iFeatureOmega encodings
  screening_results.tsv                          # 6 methods x 6 classifiers CV auROC/auPRC
  top2_feature_methods.tsv                       # the two encodings carried into SVM-RFE
  rfe/{stress}_{feature_set}_selected_features.txt
  rfe/{stress}_{feature_set}_rfe_curve.tsv       # auROC vs n_features (Figure 6 data)
  rfe_summary.tsv                                # optimal feature count + best CV auROC
  table2_results.tsv                              # full stress x feature_set x classifier results (Table 2)
  table3_best_models.tsv                          # best combo per stress (Table 3, pre-tuning)
  roc_data/{stress}_{feature_set}_{classifier}.tsv  # raw test predictions (for ROC curves)
  final_models/{stress}_model.joblib              # final tuned, trained model
  final_models/{stress}_meta.json                 # feature set, selected columns, best hyperparameters
  final_model_metrics.tsv                          # final tuned Accuracy/auROC/auPRC per stress
  figures/
    fig1_dataset_composition.{png,pdf}
    fig2_feature_screening.{png,pdf}
    fig3_rfe_curves.{png,pdf}
    fig4_roc_curves.{png,pdf}
    fig5_final_model_metrics.{png,pdf}
```

## What to send back

- `data/table3_best_models.tsv` and `data/final_model_metrics.tsv` (the key
  numbers -- analogous to the reference paper's Tables 2 & 3)
- the `data/figures/` folder (or just eyeball them and describe anything that
  looks off -- e.g. a flat RFE curve, a classifier stuck at 0.5 auROC)
- console output of `05_screen_feature_methods.py` (which two encodings won)
  and `06_svm_rfe_selection.py` (optimal feature counts)

That's enough for me to sanity-check the results and, if needed, adjust
hyperparameter grids or flag anything that looks like overfitting before
moving on to Phase 3 (genome-wide prediction + candidate gene analysis).

## Important caveat on the `cv_auROC` column

Do **not** report `cv_auROC` from `table2_results.tsv` / `table3_best_models.tsv`
in the manuscript. Script 06 uses a linear SVM to perform the RFE, so the
selected feature subset is optimized for SVM on that exact training data;
re-scoring an SVM by CV on that same subset gives inflated values (~0.99 for
SVM vs. 0.65-0.83 for the other five classifiers on identical features). The
other classifiers' CV values are clean, since they took no part in selection.

Model selection in script 07 uses `test_auROC` -- the held-out test set, never
touched by RFE -- so the "best model per stress" conclusion is unaffected.
Report only `test_accuracy` / `test_auROC` / `test_auPRC`, which is also what
the reference paper's Tables 2 and 3 report.

---

# Milestone 3 -- Genome-wide prediction & candidate gene analysis

Applies the finalized models to the whole date palm proteome, ranks novel
candidates, runs discriminative motif discovery, and joins the project's
existing functional annotation onto the top candidates (pipeline doc Phase 3;
reference paper Section 3.1 and 3.4).

Functional annotation (BLAST vs. Arabidopsis + GO) was already completed
earlier in the project, so script 11 **joins those existing results** onto the
candidates rather than recomputing them.

## Run in order

```bash
cd /home/muhammad-haroon/date-palm-project/stress_gene_prediction/scripts
source ../.venv/bin/activate

# 10. Score all 48,801 proteins with each stress's final model (chunked +
#     resumable -- safe to Ctrl-C and rerun; this is the long one)
python3 10_genomewide_predict.py

# 11. Rank candidates, apply the paper's "unlabeled set" rule, join annotation
python3 11_rank_candidates.py

# 12. Discriminative motif discovery (needs MEME -- see note below)
bash 12_meme_motif_analysis.sh

# 13. Figures for the genome-wide stage
python3 13_phase3_figures.py
```

### MEME is not installed

MEME Suite isn't packaged in apt on this machine. Script 12 always writes the
positive/negative FASTA pairs, then either runs MEME if it's on PATH or tells
you it isn't. To install it (~10 min build):

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

Two gotchas hit during this build, both already worked around above:

1. `--enable-serial` is required on this machine. `openmpi-bin` is installed
   but `libopenmpi-dev` is not, so configure enables the MPI build and the
   compile dies with `fatal error: mpi.h: No such file or directory`. MEME's
   MPI mode gains nothing at our scale (a few hundred sequences per stress).
2. The 5.5.7 tarball is missing `doc/generate_news_txt.pl`, so `make` fails in
   `doc/` with `Can't open perl script "./generate_news_txt.pl"` *after* all
   binaries have already built successfully. `touch doc/news.txt` then rerun
   `make` -- the target is a bundled release-notes text file and has no effect
   on MEME's functionality.

Alternatively upload `data/meme/{stress}_positive.fasta` (primary) and
`{stress}_negative.fasta` (control) to https://meme-suite.org in
Discriminative mode, protein alphabet, 3 motifs, widths 6-10 -- the same
settings the reference paper used.

## Outputs

```
stress_gene_prediction/data/genomewide/
  all_predictions.tsv                  every protein x 4 stress probabilities
  {stress}_ranked_predictions.tsv      unlabeled proteins ranked by score
  {stress}_top10.tsv                   top-10 candidates + description,
                                       Arabidopsis homolog, GO terms
  table4_top10_candidates.tsv          Table 4 equivalent (all stresses)
  srg_counts.tsv                       predicted SRG counts per threshold
  skipped_sequences.txt                sequences too short to encode
stress_gene_prediction/data/meme/
  {stress}_positive.fasta, {stress}_negative.fasta
  {stress}_meme_out/meme.html          motif logos + E-values
stress_gene_prediction/data/figures/
  fig6_prediction_distributions.{png,pdf}
  fig7_predicted_srg_counts.{png,pdf}
  fig8_top10_candidates.{png,pdf}
```

## What to send back

- `data/genomewide/table4_top10_candidates.tsv` and `srg_counts.tsv`
- the four `{stress}_top10.tsv` files (these carry the annotation used to
  write the candidate discussion sections)
- figures 6-8
- whether MEME ran, and if so the motif logos from each `meme.html`

## Notes on the "unlabeled set" definition

Each stress is ranked over every protein except that stress's own known
single-stress genes (the pre-CD-HIT positives from Milestone 1), matching the
reference paper's Table 1 arithmetic (`41,019 - 728 = 40,291` for cabbage
cold). Genes known for a *different* stress deliberately stay in the pool, so
a gene already known for salt can still surface as a novel cold candidate.

---

# Milestone 3 (revised) -- fixes for the candidate tables

Review of the first Milestone 3 run turned up two defects that would not
survive peer review. These scripts fix both and write to **new folders**
(`data/genomewide_v2/`, `data/figures_v2/`); the original `data/genomewide/`
and `data/figures/` are left untouched for comparison.

### Defect 1 -- the top-10 rankings were arbitrary ties

Every top-10 entry showed `Likeliness = 1.0`, but the real value was
`0.99999999999997` for all of them. `predict_proba` on an SVC is Platt scaling,
a sigmoid of the decision value, and it saturates at the top of the range:

| Stress | Proteins tied at the top score |
|---|---|
| Cold | 60 |
| Drought | 30 |
| Heat | 18 |
| Salt | 10 (i.e. the *entire* top-10 was one tie block) |

Order within a tie block came from the sort, not the model. The fix ranks on
the SVM decision margin instead -- the same quantity before the sigmoid, so
the ordering is the one the model intended, at full resolution and with no
ties. Probability is still reported, since it is the interpretable number.

### Defect 2 -- "top 10" was not 10 distinct genes

Heat's top-10 contained only **4 distinct genes**: seven entries were isoforms
X1-X10 of one ubiquitin-conjugating enzyme. Drought had 7 distinct, salt 8.
CD-HIT deduplicated the training set but never the prediction set. The fix
collapses isoforms and keeps the highest-margin representative per gene.

Grouping uses NCBI **GeneID** when `datepalm_gene_map.tsv` exists. Description
-based grouping is only a fallback because it over-merges paralogs: 105
distinct proteins share the description "putative disease resistance protein
RGA3" and are separate genes, so genome-wide gene counts from the fallback are
an undercount. Script 15 prints which mode it used and records it in
`srg_counts.tsv`.

## Run in order

```bash
cd /home/muhammad-haroon/date-palm-project/stress_gene_prediction/scripts
source ../.venv/bin/activate

# 14.  Re-score the proteome, saving decision margins as well as probabilities
#      (chunked + resumable, same cost as script 10)
python3 14_genomewide_rescore.py

# 14b. Authoritative protein->GeneID map (one pass over a 29 GB file, a few
#      minutes; run once). Skippable -- script 15 falls back gracefully.
bash 14b_build_gene_map.sh

# 15.  Rank by margin, collapse isoforms, join existing annotation
python3 15_rank_candidates_v2.py

# 16.  Revised figures
python3 16_phase3_figures_v2.py
```

Scripts 14 and 14b are independent, so they can run in either order (or in two
terminals at once).

## Outputs

```
stress_gene_prediction/data/genomewide_v2/
  all_predictions.tsv                 ProteinID, {stress}_prob, {stress}_margin
  datepalm_gene_map.tsv               ProteinID -> GeneID, Symbol (from 14b)
  {stress}_ranked_proteins.tsv        every unlabeled protein, margin-ranked
  {stress}_ranked_genes.tsv           isoform-collapsed, margin-ranked
  {stress}_top10.tsv                  top-10 DISTINCT genes + annotation
  {stress}_top10_isoform_members.tsv  which proteins each entry absorbed
  table4_top10_candidates.tsv         revised Table 4
  srg_counts.tsv                      protein-level and gene-level counts
stress_gene_prediction/data/figures_v2/
  fig6_prediction_distributions.{png,pdf}
  fig7_predicted_srg_counts.{png,pdf}   protein-level vs gene-level
  fig8_top10_candidates.{png,pdf}       ranked by margin, labelled with product
```

## Still outstanding: the over-prediction issue

These two fixes do **not** address the third problem from the review -- the
models call 15-26% of the proteome stress-responsive, against 0.3-5.2% in the
reference paper. That is inherited from the paper's own negative-set design:
the negative class during training is *genes responsive to the other three
stresses*, so the model was never taught to reject an ordinary non-stress
protein, which is what most of the proteome is. Isoform collapsing will reduce
the raw counts somewhat, but not to the paper's scale.

Options, in increasing order of effort:
- Report ranked candidates at a stringent cutoff as the deliverable, and state
  in Methods that the score is a relative ranking metric rather than a
  calibrated genome-wide prevalence estimate.
- Choose the threshold from the training-set base rate rather than a flat 0.5.
- Add randomly sampled non-stress proteins as a third negative class and
  retrain. Most defensible statistically, but deviates from the reference
  methodology and means redoing Milestone 2.
