# Multi-task shared-backbone model for date palm stress genes

Addresses the limitation the reference paper states in its own Discussion
(Horticulturae 2025, 11, 44, p. 17):

> "A key challenge in this research is that a single gene can be associated
> with multiple types of stress... If a multiclassification model were
> employed... the sum of the probabilities for all four stresses would equal
> to 1... thereby failing to account for the possibility of a gene responding
> to multiple stresses. To overcome this limitation, we constructed separate
> models for each type of stress."

Four separate models is a workaround, not a fix: it forces every multi-stress
gene to be discarded from training. In our data that is **735 genes, 24.1% of
all labelled data** — including the 239 drought+salt genes that represent the
osmotic-stress crosstalk central to the biology, and 50 genes responding to
all four stresses.

**The paper's objection applies only to softmax.** Four independent sigmoid
heads emit independent Bernoulli probabilities, which need not sum to 1, so a
gene can score high for drought and salt simultaneously. That makes a unified
model valid, and it is what this folder implements.

## Architecture

```
              ESM-2 embedding (frozen, 1280-d)
                          |
               shared backbone: 1280 -> 512 -> 256
                  LayerNorm + GELU + dropout
                          |
     +------------+-------+-------+------------+
   cold         heat         drought         salt
  head          head          head           head
     |            |             |              |
  sigmoid      sigmoid       sigmoid        sigmoid
```

A **pretrained** backbone is what makes a deep model viable here. The paper
correctly warns that deep learning "require[s] a large amount of data" — true
when the representation must be learned from task data alone. ESM-2 is
pretrained on ~65M proteins, so only a small head is fitted to our data.

## Four dataset changes versus the single-task pipeline

| # | Change | Why |
|---|---|---|
| 1 | Multi-label 4-bit vectors; no multi-stress exclusion | Recovers 735 discarded genes (+63–102% positives per stress) |
| 2 | Negatives = GO-annotated proteins with no stress term (43,164) | Old negatives were "other-stress genes", so models never learned to reject ordinary proteins — the cause of 15–26% of the proteome being called stress-responsive |
| 3 | **One global CD-HIT + one global split** | Previously each stress split independently, so a protein could be in cold's train and salt's test. Harmless for 4 separate models; **direct leakage for a shared backbone** |
| 4 | Per-head `pos_weight` instead of resampling | Handles 14:1 imbalance without discarding data |

Proteins with no GO annotation at all (2,584) are **excluded from training** —
absence of annotation is absence of evidence, not evidence of absence, so they
would be unreliable negatives. They are still scored genome-wide.

## Setup

```bash
source ../stress_gene_prediction/.venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install fair-esm
```

## Run order

```bash
cd multitask_stress_model/scripts
source ../../stress_gene_prediction/.venv/bin/activate

python3 20_build_multilabel_dataset.py      # done: 46,217 proteins
bash    21_run_cdhit_global.sh              # global redundancy removal
python3 22_make_global_split.py             # one leakage-free 7:3 split

python3 23_extract_esm_embeddings.py            # GPU, ~30-60 min
python3 23_extract_esm_embeddings.py --proteome # all 48,801, ~1-2 h

python3 24_train_multitask.py               # shared backbone + 4 heads
python3 25_evaluate_and_compare.py          # ablation: SVM / independent MLP / shared
python3 26_genomewide_multitask.py          # genome-wide + co-response
python3 27_figures_multitask.py             # figures
```

Both embedding steps are resumable (completed shards are skipped). If you hit
CUDA OOM on the 6 GB card, lower `--token-budget` (default 4096) or use
`--model esm2_t30_150M_UR50D`.

## The ablation is the scientific claim

Script 25 trains everything on **identical data, split and features**, so only
the architecture varies:

- **A** per-stress SVM
- **B** per-stress independent MLPs (same capacity, four separate backbones)
- **C** multi-task shared backbone

A vs C measures total gain; **B vs C isolates the shared backbone specifically**,
since B and C differ only in whether the representation is shared. Comparing
against the original pipeline directly would be confounded by the different
split, negatives and features, so those numbers are reported for context only
and explicitly marked as not comparable.

## Honest expectations

Stage 1 (data recovery) is near-certain to help — it is strictly more training
data. The shared backbone is a genuine empirical question at ~46k proteins
with only ~1k positives per stress. If the classical model stays competitive,
that is still a reportable finding, and the co-response output remains a
capability the single-task design cannot provide at all.

Note that global CD-HIT is stricter than the previous per-stress passes, so
final positive counts will fall below the raw multi-label numbers (987/731/
1028/1329) while still exceeding the original single-task counts
(601/448/510/759). That trade is worth it: it is what removes the leakage.

## Known caveat inherited from the labels

Labels come from Arabidopsis GO transfer, so multi-label combinations inherit
that noise — a gene labelled drought+salt may reflect ortholog annotation
breadth rather than date palm biology. This applies to the single-task
pipeline equally and should be stated as a limitation in the manuscript.
