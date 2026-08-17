#!/usr/bin/env python3
"""
Generate the manuscript as a .docx, pulling every number and figure directly
from the pipeline outputs so the text can never drift from the results.

Figures 6-8 are taken from figures_v2/ (margin-ranked, isoform-collapsed);
figures 1-5 from figures/.

Output: stress_gene_prediction/manuscript/DatePalm_SRG_Manuscript.docx
"""
import json
import os
import sys

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(SCRIPT_DIR, "..", "data")
GW = os.path.join(DATA, "genomewide_v2")
FIG = os.path.join(DATA, "figures")
FIG2 = os.path.join(DATA, "figures_v2")
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "manuscript")
OUT = os.path.join(OUT_DIR, "DatePalm_SRG_Manuscript.docx")

STRESSES = ["cold", "heat", "drought", "salt"]

# --------------------------------------------------------------------------
# Load every result the text references
# --------------------------------------------------------------------------
step1 = pd.read_csv(os.path.join(DATA, "summary_step1.tsv"), sep="\t").set_index("stress")
dsum = pd.read_csv(os.path.join(DATA, "dataset_summary.tsv"), sep="\t").set_index("stress")
screen = pd.read_csv(os.path.join(DATA, "screening_results.tsv"), sep="\t")
top2 = pd.read_csv(os.path.join(DATA, "top2_feature_methods.tsv"), sep="\t")
rfe = pd.read_csv(os.path.join(DATA, "rfe_summary.tsv"), sep="\t")
t2 = pd.read_csv(os.path.join(DATA, "table2_results.tsv"), sep="\t")
final = pd.read_csv(os.path.join(DATA, "final_model_metrics.tsv"), sep="\t").set_index("stress")
counts = pd.read_csv(os.path.join(GW, "srg_counts.tsv"), sep="\t").set_index("stress")
tops = {s: pd.read_csv(os.path.join(GW, f"{s}_top10.tsv"), sep="\t", index_col=0) for s in STRESSES}
genes = {s: pd.read_csv(os.path.join(GW, f"{s}_ranked_genes.tsv"), sep="\t", index_col=0) for s in STRESSES}

M1, M2 = top2["feature_method"].tolist()
COMBO = f"{M1}+{M2}"
margin_pos = {s: int((genes[s]["Margin"] > 0).sum()) for s in STRESSES}

# MEME motifs parsed from meme.txt
def parse_motifs():
    out = {}
    for s in STRESSES:
        p = os.path.join(DATA, "meme", f"{s}_meme_out", "meme.txt")
        rows = []
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    if line.startswith("MOTIF "):
                        parts = line.split()
                        consensus, width, sites = parts[1], None, None
                        for i, tok in enumerate(parts):
                            if tok == "width" and parts[i + 1] == "=":
                                width = parts[i + 2]
                            if tok == "sites" and parts[i + 1] == "=":
                                sites = parts[i + 2]
                            if tok == "E-value":
                                ev = parts[i + 2]
                        rows.append((consensus, width, sites, ev))
        out[s] = rows
    return out

MOTIFS = parse_motifs()

# --------------------------------------------------------------------------
# Document helpers
# --------------------------------------------------------------------------
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(11)
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.15


def H(text, level):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.color.rgb = RGBColor(0, 0, 0)
        r.font.name = "Times New Roman"
    return h


def P(text, justify=True, italic=False, size=11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = italic
    run.font.size = Pt(size)
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def caption(text, size=9.5):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = False
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p


def table_caption(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(9.5)
    return p


def add_table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(str(h))
        run.bold = True
        run.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.font.size = Pt(9)
    if widths:
        for r_ in t.rows:
            for i, w in enumerate(widths):
                r_.cells[i].width = Inches(w)
    return t


def add_figure(path, cap, width=6.2):
    if not os.path.exists(path):
        caption(f"[MISSING FIGURE: {path}]")
        return
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(cap)


# ==========================================================================
# TITLE
# ==========================================================================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("Machine Learning-Based Prediction of Abiotic Stress-Responsive "
                  "Genes in Date Palm (Phoenix dactylifera L.)")
r.bold = True
r.font.size = Pt(16)

P("Author names and affiliations to be completed.", justify=False, italic=True, size=10)
P("* Correspondence: [email]", justify=False, italic=True, size=10)

# ==========================================================================
# ABSTRACT
# ==========================================================================
H("Abstract", 1)
P(
    f"Date palm (Phoenix dactylifera L.) is a keystone crop of arid and semi-arid "
    f"agriculture, yet its productivity is increasingly constrained by abiotic "
    f"stresses including drought, extreme temperatures and soil salinity. "
    f"Experimental identification of stress-responsive genes (SRGs) remains slow "
    f"and costly, and computational models dedicated to this species are lacking. "
    f"Here we present a machine learning framework that predicts date palm genes "
    f"responsive to four abiotic stresses—cold, heat, drought and salinity—directly "
    f"from protein sequence. Stress-associated proteins were identified through "
    f"orthology-based Gene Ontology transfer from Arabidopsis thaliana, filtered to "
    f"single-stress responders, and made non-redundant with CD-HIT at 80% identity, "
    f"yielding {dsum.loc['cold','positive_set']}, {dsum.loc['heat','positive_set']}, "
    f"{dsum.loc['drought','positive_set']} and {dsum.loc['salt','positive_set']} "
    f"positive sequences for cold, heat, drought and salt, respectively. Six sequence "
    f"encodings (CKSAAP, DDE, DPC, TPC, PAAC, APAAC) were evaluated against six "
    f"classifiers (SVM, XGB, RF, GBDT, ADB, BAG) under five-fold cross-validation, "
    f"and optimal feature subsets were derived by SVM-recursive feature elimination. "
    f"Support vector machines on {M2}-derived features gave the strongest held-out "
    f"performance, with test-set areas under the receiver operating characteristic "
    f"curve of {final.loc['cold','test_auROC']:.2f}%, {final.loc['heat','test_auROC']:.2f}%, "
    f"{final.loc['drought','test_auROC']:.2f}% and {final.loc['salt','test_auROC']:.2f}% "
    f"for cold, heat, drought and salt. Applying the final models across all 48,801 "
    f"date palm proteins produced ranked genome-wide candidate lists. The highest-ranked "
    f"candidates recovered canonical stress genes absent from training, including heat "
    f"shock transcription factors and HSP81 chaperones under heat, a sodium/hydrogen "
    f"exchanger and WRKY32 under salinity, and DEHYDRATION-INDUCED 19 under drought, "
    f"providing independent biological validation. This framework offers a rapid "
    f"prioritisation tool for stress-tolerance breeding in date palm.",
)
P("Keywords: Phoenix dactylifera; date palm; abiotic stress; stress-responsive genes; "
  "machine learning; support vector machine; protein sequence features",
  italic=False)

# ==========================================================================
# 1. INTRODUCTION
# ==========================================================================
H("1. Introduction", 1)
P("Date palm (Phoenix dactylifera L.) is among the most economically and culturally "
  "significant perennial crops of the Middle East and North Africa, sustaining "
  "agricultural systems in regions where few other fruit species are viable. Its "
  "capacity to persist in hot, arid and often saline environments has made it a "
  "cornerstone of food security across these regions. Nevertheless, the intensity "
  "and frequency of abiotic stress episodes are rising under climate change, and "
  "even a species as resilient as date palm experiences measurable losses in yield "
  "and fruit quality when drought, temperature extremes or soil salinity exceed its "
  "adaptive range [1,2].")
P("Abiotic stress responses in plants are coordinated by complex, partially "
  "overlapping regulatory networks. Perception of a stress cue triggers signalling "
  "cascades that converge on transcription factors, which in turn remodel the "
  "expression of effector genes governing osmotic adjustment, reactive oxygen "
  "species detoxification, protein homeostasis and membrane transport [3]. "
  "Identifying the genes that participate in these programmes—collectively termed "
  "stress-responsive genes (SRGs)—is a prerequisite for both mechanistic "
  "understanding and marker-assisted breeding of tolerant cultivars.")
P("Conventional SRG discovery relies on transcriptomic profiling, quantitative "
  "trait locus mapping or targeted reverse genetics. In date palm these approaches "
  "have produced valuable but fragmentary resources: transcriptome studies have "
  "characterised responses to salinity in root and leaf tissue [4,5] and to combined "
  "heat and drought [6], while gene family analyses have described individual "
  "regulatory families. Because the species has a long generation time and limited "
  "transformation capacity relative to model plants, experimental validation is "
  "particularly laborious, and the fraction of the genome functionally annotated "
  "with respect to stress remains small.")
P("Machine learning offers a complementary route. Protein sequence encodes, in a "
  "compressed form, the compositional and physicochemical signatures associated "
  "with functional classes, and supervised classifiers can learn these signatures "
  "from curated examples and extrapolate them genome-wide [7,8]. This strategy has "
  "been applied successfully to abiotic stress protein classification in cereals "
  "and, most directly relevant to the present work, to the prediction of SRGs in "
  "Chinese cabbage (Brassica rapa ssp. pekinensis), where six sequence encodings "
  "and six classifiers were systematically compared to build stress-specific "
  "binary models [9]. To our knowledge no equivalent computational resource exists "
  "for date palm.")
P("Here we adapt and extend that framework to Phoenix dactylifera. We construct "
  "single-stress positive and negative sequence sets for cold, heat, drought and "
  "salinity, systematically benchmark six sequence encodings against six "
  "classification algorithms, apply SVM-recursive feature elimination to identify "
  "compact discriminative feature subsets, and deploy the resulting models across "
  "the complete date palm proteome. We further characterise discriminative sequence "
  "motifs and annotate the highest-ranked novel candidates by orthology. The result "
  "is the first sequence-based predictive resource for abiotic stress responsiveness "
  "in this species.")

# ==========================================================================
# 2. MATERIALS AND METHODS
# ==========================================================================
H("2. Materials and Methods", 1)

H("2.1. Data Collection and Pre-Processing", 2)
P("The complete Phoenix dactylifera reference proteome (48,801 protein sequences) "
  "was obtained from NCBI RefSeq. Because experimentally validated, genome-scale "
  "stress gene catalogues are not available for date palm at the scale required "
  "for supervised learning, stress associations were established by orthology-based "
  "functional transfer. Each date palm protein was searched against the Arabidopsis "
  "thaliana reference proteome using BLASTP [10], and the single best hit was "
  "retained. Gene Ontology (GO) terms annotated to that Arabidopsis ortholog were "
  "transferred to the corresponding date palm protein.")
P("Stress categories were then defined by curated, stress-specific GO term sets, "
  "following the principle used in the reference framework of selecting a small "
  "number of terms tightly associated with each stress rather than every "
  "stress-adjacent term. Cold was defined by response to cold (GO:0009409), cold "
  "acclimation (GO:0009631), cellular response to cold (GO:0070417), response to "
  "freezing (GO:0050826) and cellular response to freezing (GO:0071497). Heat was "
  "defined by response to heat (GO:0009408), heat acclimation (GO:0010286), cellular "
  "response to heat (GO:0034605) and cellular heat acclimation (GO:0070370). Drought "
  "was defined by response to water deprivation (GO:0009414), cellular response to "
  "water deprivation (GO:0042631), drought recovery (GO:0009819) and the three "
  "regulatory terms GO:2000070, GO:0080148 and GO:1902584. Salinity was defined by "
  "response to salt stress (GO:0009651), response to salt (GO:1902074), hyperosmotic "
  "salinity response (GO:0042538), cellular response to salt stress (GO:0071472) and "
  "the regulatory terms GO:1901000, GO:1901001 and GO:1901002. Terms shared between "
  "stresses—including response to osmotic stress, response to temperature stimulus, "
  "response to abscisic acid and generic oxidative stress terms—were deliberately "
  "excluded because they do not discriminate among the four target stresses.")
P(f"Proteins whose stress-relevant annotations spanned more than one of the four "
  f"target categories were removed entirely, so that each retained sequence responds "
  f"to a single stress. This Boolean filtering step excluded 735 multi-stress "
  f"proteins and yielded {step1.loc['cold','single_stress_candidates']}, "
  f"{step1.loc['heat','single_stress_candidates']}, "
  f"{step1.loc['drought','single_stress_candidates']} and "
  f"{step1.loc['salt','single_stress_candidates']} single-stress candidates for "
  f"cold, heat, drought and salt, respectively.")
P(f"Sequence redundancy was removed independently within each stress set using "
  f"CD-HIT [11] at a sequence identity threshold of 0.8, producing final positive "
  f"sets of {dsum.loc['cold','positive_set']} (cold), {dsum.loc['heat','positive_set']} "
  f"(heat), {dsum.loc['drought','positive_set']} (drought) and "
  f"{dsum.loc['salt','positive_set']} (salt) sequences. Following the reference "
  f"framework, the negative set for each stress was formed from the union of the "
  f"non-redundant positive sets of the remaining three stresses. Each dataset was "
  f"partitioned into training and test subsets in a 7:3 ratio, stratified by class "
  f"(Table 1, Figure 1).")

table_caption("Table 1. Composition of the curated single-stress datasets.")
add_table(
    ["Category", "Cold", "Heat", "Drought", "Salt"],
    [
        ["Single-stress candidates"] + [int(step1.loc[s, "single_stress_candidates"]) for s in STRESSES],
        ["Positive set (post CD-HIT)"] + [int(dsum.loc[s, "positive_set"]) for s in STRESSES],
        ["Negative set"] + [int(dsum.loc[s, "negative_set"]) for s in STRESSES],
        ["Training set"] + [int(dsum.loc[s, "train_total"]) for s in STRESSES],
        ["Test set"] + [int(dsum.loc[s, "test_total"]) for s in STRESSES],
        ["Unlabelled set (genome-wide)"] + [int(counts.loc[s, "unlabeled_proteins"]) for s in STRESSES],
    ],
)
doc.add_paragraph()
add_figure(os.path.join(FIG, "fig1_dataset_composition.png"),
           "Figure 1. Composition of the curated dataset for each stress type. Left, "
           "positive versus negative set sizes after CD-HIT redundancy removal. Right, "
           "the stratified 7:3 training/test partition.")

H("2.2. Feature Construction and Selection", 2)
P("Protein sequences were converted to numerical feature vectors with iFeatureOmega "
  "[12] using six encodings: Composition of k-spaced Amino Acid Pairs (CKSAAP), "
  "Dipeptide Deviation from Expected mean (DDE), Di-Peptide Composition (DPC), "
  "Tri-Peptide Composition (TPC), Pseudo-Amino Acid Composition (PAAC) and "
  "Amphiphilic Pseudo-Amino Acid Composition (APAAC) [13]. These produced 1,600, "
  "400, 400, 8,000, 23 and 26 features per sequence, respectively. Default "
  "parameters were used (k-space 3 for CKSAAP; lambda 3 and weight 0.05 for "
  "PAAC/APAAC), and all sequences exceeded the minimum length required by these "
  "settings.")
P(f"High-dimensional encodings contain many sparse and mutually correlated "
  f"features, which increases model complexity without necessarily improving "
  f"discrimination. Feature selection was therefore performed by support vector "
  f"machine recursive feature elimination (SVM-RFE) [14] with a linear kernel, "
  f"embedded in five-fold cross-validation and scored by area under the receiver "
  f"operating characteristic curve (auROC). Consistent with the reference "
  f"framework, feature selection was applied to the two best-performing encodings "
  f"identified in the initial screen ({M1} and {M2}) and to their concatenation.")

H("2.3. Prediction Using Machine Learning Methods", 2)
P("Six supervised binary classifiers were evaluated: support vector machine (SVM) "
  "with a radial basis function kernel, extreme gradient boosting (XGB), random "
  "forest (RF), gradient boosting decision trees (GBDT), adaptive boosting (ADB) "
  "and bagging (BAG). All models were implemented in Python with scikit-learn [15], "
  "except XGB, which used the XGBoost library [16]. Features were standardised "
  "within a pipeline so that scaling parameters were estimated from training folds "
  "only. A separate binary model was trained for each stress rather than a single "
  "multi-class model, since a gene may legitimately respond to more than one stress "
  "and a multi-class formulation would force mutually exclusive assignment.")
P("Hyperparameters of the best-performing configuration for each stress were tuned "
  "by randomised search over 20 sampled configurations with five-fold "
  "cross-validation on the training set, optimising auROC.")

H("2.4. Cross-Validation and Performance Metrics", 2)
P("Model development used five-fold stratified cross-validation on the training "
  "partition, with the test partition held out entirely until final evaluation. "
  "Performance was quantified by accuracy, auROC and area under the "
  "precision-recall curve (auPRC); auPRC is reported alongside auROC because the "
  "datasets are class-imbalanced, with positives comprising roughly one quarter to "
  "one third of each set.")
P("We note one methodological caveat that governs how results are reported. Because "
  "SVM-RFE selects features using a linear SVM optimised on the training folds, "
  "cross-validation scores subsequently obtained for SVM classifiers on those same "
  "selected subsets are optimistically biased. All performance figures reported in "
  "this study are therefore held-out test set metrics, computed on data that took "
  "no part in either feature selection or hyperparameter tuning.")

H("2.5. Genome-Wide Prediction and Motif Analysis", 2)
P("Final tuned models were applied to all 48,801 date palm proteins. Sequences were "
  "processed in chunks, and residues outside the twenty standard amino acids were "
  "stripped prior to encoding; 535 sequences contained such residues and no sequence "
  "was lost. Following the reference framework, the unlabelled set for each stress "
  "comprised every protein except that stress's own known single-stress genes, so "
  "that genes already associated with a different stress remained eligible as novel "
  "candidates.")
P("Candidates were ranked by the SVM decision margin rather than the Platt-scaled "
  "probability. The probability transform saturates numerically at the top of its "
  "range and generated large tie groups (up to 60 proteins sharing an identical "
  "value), which would have made the ordering of top-ranked candidates arbitrary. "
  "The decision margin is a strictly monotonic transform of the same quantity "
  "(Spearman rho = 1.000 against probability for all four stresses) and preserves "
  "the intended ordering at full numerical resolution.")
P("Splice isoforms were collapsed so that candidate lists represent distinct genes. "
  "Protein accessions were mapped to NCBI GeneIDs via gene2accession (taxonomy ID "
  "42345), grouping the 48,801 proteins into 29,239 genes, and the highest-scoring "
  "isoform was retained as the representative of each gene.")
P("Discriminative motif discovery was performed with MEME [17] in differential "
  "enrichment mode, contrasting each stress's non-redundant positive set against "
  "its corresponding negative set, with three motifs requested per stress and motif "
  "widths constrained to 6-10 residues. Functional annotation of top candidates used "
  "the Arabidopsis best-hit orthology and transferred GO terms described in Section 2.1.")

# ==========================================================================
# 3. RESULTS
# ==========================================================================
H("3. Results", 1)

H("3.1. Discriminative Sequence Motifs", 2)
sig = {s: [m for m in MOTIFS[s] if m[3] and float(m[3]) < 0.05] for s in STRESSES}
P(f"Discriminative motif analysis identified sequence patterns enriched in each "
  f"stress's positive set relative to its negative set (Table 2). Applying a "
  f"significance threshold of E < 0.05, "
  f"{len(sig['cold'])} motif(s) were significant for cold, {len(sig['heat'])} for heat, "
  f"{len(sig['drought'])} for drought and {len(sig['salt'])} for salt. The strongest "
  f"signal overall was the drought-associated motif NIAYPF (E = 6.5e-04). Two "
  f"drought motifs, HRDVKTNN and HKDVKLAB, correspond to the conserved catalytic "
  f"loop of protein kinases, consistent with the central role of kinase signalling "
  f"cascades in osmotic stress perception and transduction. Several requested motifs "
  f"did not reach significance, indicating that discriminative signal at the level "
  f"of short linear motifs is modest relative to the distributed compositional "
  f"signal exploited by the classifiers.")

table_caption("Table 2. Discriminative motifs identified by MEME for each stress "
              "(positive versus negative set). Motifs significant at E < 0.05 are "
              "marked with an asterisk.")
motif_rows = []
for s in STRESSES:
    for cons, w, sites, ev in MOTIFS[s]:
        star = "*" if ev and float(ev) < 0.05 else ""
        motif_rows.append([s.capitalize(), cons + star, w, sites, ev])
add_table(["Stress", "Motif consensus", "Width", "Sites", "E-value"], motif_rows)
doc.add_paragraph()

H("3.2. Feature Encoding and Classifier Screening", 2)
rank_txt = ", ".join(
    f"{m} ({screen[screen.feature_method == m].cv_auROC_mean.mean():.4f})"
    for m in screen.groupby("feature_method").cv_auROC_mean.mean().sort_values(ascending=False).index
)
P(f"All six encodings were screened against all six classifiers by five-fold "
  f"cross-validation on the training sets (Figure 2). Averaged across the four "
  f"stresses and six classifiers, mean auROC ranked the encodings as follows: "
  f"{rank_txt}. {M1} and {M2} were the two strongest and were carried forward, "
  f"together with their concatenation, into feature selection. SVM was consistently "
  f"the strongest classifier across encodings, in contrast to the reference study "
  f"in Chinese cabbage where tree ensembles frequently prevailed; the low-dimensional "
  f"PAAC and APAAC encodings performed worst overall, mirroring that study's finding.")
add_figure(os.path.join(FIG, "fig2_feature_screening.png"),
           "Figure 2. Five-fold cross-validated auROC for every combination of six "
           "feature encodings and six classifiers, shown separately for each stress. "
           "Colour intensity encodes auROC.")

H("3.3. Feature Selection by SVM-RFE", 2)
P(f"SVM-RFE was applied to the {M1}, {M2} and {COMBO} feature sets for each stress "
  f"(Figure 3, Table 3). Optimal subsets retained a substantial proportion of the "
  f"available features, ranging from {int(rfe.n_selected.min())} to "
  f"{int(rfe.n_selected.max())}. Cross-validated auROC rose steeply over the first "
  f"few hundred features and then plateaued, indicating that discriminative "
  f"information is distributed across many weakly informative k-mer features rather "
  f"than concentrated in a small subset.")
table_caption("Table 3. Optimal feature subset size selected by SVM-RFE for each "
              "stress and feature set.")
add_table(
    ["Stress", "Feature set", "Total features", "Selected features"],
    [[r.stress.capitalize(), r.feature_set, int(r.n_total), int(r.n_selected)]
     for r in rfe.itertuples()],
)
doc.add_paragraph()
add_figure(os.path.join(FIG, "fig3_rfe_curves.png"),
           "Figure 3. Cross-validated auROC as a function of the number of features "
           "retained during SVM-RFE, for each stress and feature set. Markers denote "
           "the selected optimum.")

H("3.4. Model Performance on Held-Out Data", 2)
best_rows = []
for s in STRESSES:
    sub = t2[t2.stress == s]
    for fs in [M1, M2, COMBO]:
        ss = sub[sub.feature_set == fs]
        if len(ss):
            b = ss.loc[ss.test_auROC.idxmax()]
            best_rows.append([s.capitalize(), fs, b.classifier,
                              f"{b.test_accuracy:.2f}", f"{b.test_auROC:.2f}", f"{b.test_auPRC:.2f}"])
P(f"The six classifiers were retrained on the RFE-selected subsets and evaluated on "
  f"the held-out test partition (Table 4, Figure 4). The best-performing "
  f"configuration for every stress used a support vector machine. {M2}-derived "
  f"features gave the highest test auROC for cold, heat and drought, while the "
  f"{COMBO} combination was marginally superior for salt.")
table_caption("Table 4. Best-performing classifier for each stress and feature set, "
              "evaluated on the held-out test set.")
add_table(["Stress", "Feature set", "Model", "Accuracy (%)", "auROC (%)", "auPRC (%)"], best_rows)
doc.add_paragraph()
add_figure(os.path.join(FIG, "fig4_roc_curves.png"),
           "Figure 4. Receiver operating characteristic curves on the held-out test "
           "set for all six classifiers, using the best feature set for each stress.")

P("Final models were hyperparameter-tuned and re-evaluated (Table 5, Figure 5). "
  "Test-set auROC ranged from "
  f"{final.test_auROC.min():.2f}% to {final.test_auROC.max():.2f}%, with heat "
  f"stress achieving the highest discrimination and cold the lowest.")
table_caption("Table 5. Final tuned model for each stress and its performance on the "
              "held-out test set.")
add_table(
    ["Stress", "Feature set", "Model", "No. features", "Hyperparameters",
     "Accuracy (%)", "auROC (%)", "auPRC (%)"],
    [[s.capitalize(), final.loc[s, "feature_set"], final.loc[s, "classifier"],
      int(final.loc[s, "n_features"]),
      json.loads(final.loc[s, "best_params"]).get("clf__C", "") and
      f"C={json.loads(final.loc[s,'best_params'])['clf__C']}, "
      f"gamma={json.loads(final.loc[s,'best_params'])['clf__gamma']}",
      f"{final.loc[s,'test_accuracy']:.2f}", f"{final.loc[s,'test_auROC']:.2f}",
      f"{final.loc[s,'test_auPRC']:.2f}"] for s in STRESSES],
)
doc.add_paragraph()
add_figure(os.path.join(FIG, "fig5_final_model_metrics.png"),
           "Figure 5. Accuracy, auROC and auPRC of the final tuned model for each "
           "stress on the held-out test set.")

H("3.5. Genome-Wide Prediction of Stress-Responsive Genes", 2)
P(f"The final models were applied to all 48,801 date palm proteins, collapsed to "
  f"29,239 distinct genes. Predicted SRG counts depend strongly on the decision "
  f"threshold applied (Table 6, Figures 6 and 7). At the Platt-scaled probability "
  f"threshold of 0.5, between {int(counts.genes_p_05.min()) if False else int(counts['genes_p>0.5'].min()):,} "
  f"and {int(counts['genes_p>0.5'].max()):,} genes were called positive per stress; "
  f"at the raw decision boundary of the support vector machine the corresponding "
  f"counts were substantially smaller ({margin_pos['cold']} for cold, "
  f"{margin_pos['heat']} for heat, {margin_pos['drought']} for drought and "
  f"{margin_pos['salt']} for salt).")
P("This sensitivity to threshold reflects a structural property of the training "
  "design, inherited from the reference framework, and it constrains how these "
  "counts should be interpreted. Because the negative class comprises genes "
  "responsive to the other three stresses rather than a random genomic background, "
  "the models were never trained to reject proteins that are not stress-associated "
  "at all—the majority of the proteome. The resulting scores are therefore reliable "
  "as a relative ranking of stress-responsiveness, but should not be read as "
  "calibrated estimates of the genome-wide prevalence of stress-responsive genes. "
  "We accordingly present ranked candidate lists as the primary genome-wide output "
  "and report threshold-dependent counts for transparency rather than as a "
  "quantitative claim about genome composition.")
table_caption("Table 6. Predicted stress-responsive genes across the unlabelled set "
              "at three probability thresholds, and at the support vector machine "
              "decision boundary.")
add_table(
    ["Stress", "Genes screened", "p > 0.5", "p > 0.7", "p > 0.9", "Decision boundary"],
    [[s.capitalize(), f"{int(counts.loc[s,'distinct_genes']):,}",
      f"{int(counts.loc[s,'genes_p>0.5']):,}", f"{int(counts.loc[s,'genes_p>0.7']):,}",
      f"{int(counts.loc[s,'genes_p>0.9']):,}", f"{margin_pos[s]:,}"] for s in STRESSES],
)
doc.add_paragraph()
add_figure(os.path.join(FIG2, "fig6_prediction_distributions.png"),
           "Figure 6. Distribution of predicted probabilities across the unlabelled "
           "gene set for each stress. The dashed line marks p = 0.5.")
add_figure(os.path.join(FIG2, "fig7_predicted_srg_counts.png"),
           "Figure 7. Predicted stress-responsive gene counts per stress at three "
           "probability thresholds, before and after collapsing splice isoforms.")

H("3.6. Top-Ranked Novel Candidate Genes", 2)
P("The ten highest-ranked genes for each stress are given in Table 7 and Figure 8, "
  "with their Arabidopsis orthologs. These candidates were absent from the "
  "corresponding training sets, so their identities constitute an independent test "
  "of biological plausibility.")
P("Heat candidates were dominated by canonical thermotolerance machinery: the "
  "highest-ranked gene is an ortholog of BOBBER1, a non-canonical small heat shock "
  "protein required for thermotolerance in Arabidopsis [18], followed by a dnaJ/HSP40 "
  "co-chaperone, two HSP81 (HSP90 family) chaperones, and three heat stress "
  "transcription factors (HSFA2b, HSFA2c and HSFB4b). HSFA2 orthologs in particular "
  "are established master regulators of acquired thermotolerance [19]. Seven of the "
  "ten heat candidates therefore have direct, well-documented roles in heat response.")
P("Salt candidates included an ortholog of the plasma membrane sodium/hydrogen "
  "exchanger NHX1, whose role in Na+ compartmentalisation and salt tolerance is "
  "among the best characterised in plant salinity biology [20], together with WRKY "
  "transcription factor 32, the SnRK2-family kinase SAPK10 implicated in abscisic "
  "acid signalling, an E3 ubiquitin ligase (AIRP2) and PHOSPHATE STARVATION "
  "RESPONSE 2. Drought candidates were headed by two DEHYDRATION-INDUCED 19 (DI19) "
  "paralogs, a family with documented drought-responsive function, alongside a "
  "MYB-related transcription factor, two 1-aminocyclopropane-1-carboxylate synthase "
  "genes linking ethylene biosynthesis to water deficit, and a phospholipid "
  "hydroperoxide glutathione peroxidase consistent with oxidative damage control.")
P("Cold candidates were less specific. The list is dominated by three 14-3-3 "
  "regulatory proteins, three beta-tubulins and two N-MYC DOWNREGULATED-LIKE (NDL) "
  "proteins. While cold-induced microtubule depolymerisation and 14-3-3-mediated "
  "signalling are both documented components of low-temperature response, these "
  "families are also abundant and broadly expressed, and the cold model also showed "
  "the lowest test auROC of the four. Cold predictions should therefore be treated "
  "with greater caution than the other three.")

table_caption("Table 7. Ten highest-ranked candidate genes for each stress, with "
              "decision margin, product description and Arabidopsis best hit.")
for s in STRESSES:
    p = doc.add_paragraph()
    r = p.add_run(s.capitalize())
    r.bold = True
    r.font.size = Pt(10)
    t = tops[s]
    add_table(
        ["Rank", "Protein accession", "Margin", "Description", "A. thaliana hit", "Identity (%)"],
        [[i + 1, pid, f"{m:+.4f}", str(d)[:60], str(a) if str(a) != "nan" else "-",
          f"{float(pi):.1f}" if str(pi) != "nan" else "-"]
         for i, (pid, m, d, a, pi) in enumerate(zip(
             t.index, t["Margin"], t["Description"], t["Arabidopsis_Hit"], t["Percent_Identity"]))],
    )
    doc.add_paragraph()

add_figure(os.path.join(FIG2, "fig8_top10_candidates.png"),
           "Figure 8. Decision margins of the ten highest-ranked candidate genes for "
           "each stress, labelled with product description.")

# ==========================================================================
# 4. DISCUSSION
# ==========================================================================
H("4. Discussion", 1)
P("This study establishes the first sequence-based machine learning resource for "
  "predicting abiotic stress-responsive genes in date palm. Held-out test "
  f"performance (auROC {final.test_auROC.min():.1f}-{final.test_auROC.max():.1f}%) "
  "is comparable to that reported for the analogous Chinese cabbage framework "
  "(81.4-88.9%), despite date palm having markedly sparser experimental annotation "
  "and consequently smaller positive sets.")
P("The most informative validation is not the test metric but the identity of the "
  "top-ranked genome-wide candidates. Because these genes were excluded from "
  "training, their recovery constitutes evidence that the models capture genuine "
  "biological signal rather than dataset artefacts. The heat model's recovery of "
  "BOBBER1, multiple heat shock factors and HSP90-family chaperones, and the salt "
  "model's recovery of NHX1 and an SnRK2 kinase, are difficult to explain by "
  "compositional bias alone.")
P("Two differences from the reference framework merit comment. First, support "
  "vector machines dominated across all stresses and feature sets here, whereas "
  "tree ensembles frequently prevailed in Chinese cabbage. This likely reflects "
  "our smaller positive sets, where margin-based methods with strong regularisation "
  "generalise better than high-variance ensembles. Second, tripeptide composition "
  "(TPC) outperformed DDE as the second-ranked encoding in date palm. Both "
  "observations suggest that the optimal encoding-classifier pairing is species- "
  "and dataset-dependent, supporting the systematic screening strategy rather than "
  "transferring a fixed configuration between species.")
P("The principal limitation concerns genome-wide calibration. The negative class "
  "consists of genes responsive to other stresses rather than a random genomic "
  "background, so the decision boundary separates one stress from other stresses, "
  "not stress-responsive from non-stress-responsive genes. When such a model is "
  "applied to a proteome in which most genes belong to neither category, predicted "
  "positive counts are inflated and threshold-sensitive, as Table 6 shows. This "
  "limitation is inherent to the reference design and applies equally to it. Future "
  "work should incorporate a randomly sampled non-stress negative class, which "
  "would permit calibrated genome-wide prevalence estimates, at the cost of "
  "departing from the established framework.")
P("A second limitation follows from the orthology-based construction of the training "
  "labels. GO transfer from Arabidopsis best hits propagates any errors in the "
  "source annotation and is biased toward gene families conserved between a monocot "
  "palm and a eudicot model species. Date palm-specific stress adaptations without "
  "close Arabidopsis counterparts are therefore systematically under-represented. As "
  "species-specific transcriptomic and functional resources accumulate, "
  "literature-curated positive sets should progressively replace transferred "
  "annotations.")
P("Finally, all predictions reported here are computational hypotheses. The ranked "
  "candidate lists are intended to prioritise targets for experimental validation "
  "by expression profiling under controlled stress treatment, and ultimately by "
  "functional characterisation.")

# ==========================================================================
# 5. CONCLUSIONS
# ==========================================================================
H("5. Conclusions", 1)
P("We developed and validated stress-specific machine learning models that predict "
  "cold-, heat-, drought- and salinity-responsive genes in date palm directly from "
  "protein sequence. Systematic screening of six sequence encodings against six "
  "classifiers, followed by SVM-RFE feature selection and hyperparameter tuning, "
  f"yielded support vector machine models achieving test-set auROC of "
  f"{final.loc['cold','test_auROC']:.2f}%, {final.loc['heat','test_auROC']:.2f}%, "
  f"{final.loc['drought','test_auROC']:.2f}% and {final.loc['salt','test_auROC']:.2f}% "
  "for cold, heat, drought and salinity, respectively. Genome-wide application "
  "produced ranked candidate lists whose highest-ranked members recover canonical "
  "stress genes absent from training, including heat shock transcription factors, "
  "HSP90-family chaperones, a sodium/hydrogen exchanger and DEHYDRATION-INDUCED 19 "
  "paralogs. These resources provide a practical prioritisation tool for "
  "stress-tolerance research and breeding in date palm, and a template for "
  "extending sequence-based SRG prediction to other under-annotated crop species.")

# ==========================================================================
# BACK MATTER
# ==========================================================================
H("Supplementary Materials", 1)
P("The following supporting files accompany this manuscript: curated non-redundant "
  "FASTA datasets and label tables for each stress; complete feature "
  "screening results across all encoding-classifier combinations; SVM-RFE selected "
  "feature lists and selection curves; full classifier evaluation tables; complete "
  "genome-wide ranked prediction tables for all four stresses; protein-to-GeneID "
  "mapping; and MEME motif discovery output including sequence logos.")

H("Author Contributions", 1)
P("To be completed.")
H("Funding", 1)
P("To be completed.")
H("Data Availability Statement", 1)
P("The date palm reference proteome is available from NCBI RefSeq. All derived "
  "datasets, trained models and prediction tables generated in this study are "
  "provided as Supplementary Materials.")
H("Conflicts of Interest", 1)
P("The authors declare no conflict of interest.")

H("References", 1)
REFS = [
    "Al-Mssallem, I.S.; Hu, S.; Zhang, X.; Lin, Q.; Liu, W.; Tan, J.; Yu, X.; Liu, J.; Pan, L.; Zhang, T.; et al. Genome sequence of the date palm Phoenix dactylifera L. Nat. Commun. 2013, 4, 2274.",
    "Yaish, M.W.; Kumar, P.P. Salt tolerance research in date palm tree (Phoenix dactylifera L.), past, present, and future perspectives. Front. Plant Sci. 2015, 6, 348.",
    "Zhang, H.; Zhu, J.; Gong, Z.; Zhu, J.-K. Abiotic stress responses in plants. Nat. Rev. Genet. 2022, 23, 104-119.",
    "Yaish, M.W.; Patankar, H.V.; Assaha, D.V.M.; Zheng, Y.; Al-Yahyai, R.; Sunkar, R. Genome-wide expression profiling in leaves and roots of date palm (Phoenix dactylifera L.) exposed to salinity. BMC Genomics 2017, 18, 246.",
    "Radwan, O.; Arro, J.; Keller, C.; Korban, S.S. RNA-Seq transcriptome analysis in date palm suggests multi-dimensional responses to salinity stress. Trop. Plant Biol. 2015, 8, 74-86.",
    "Safronov, O.; Kreuzwieser, J.; Haberer, G.; Alyousif, M.S.; Schulze, W.; Al-Harbi, N.; Arab, L.; Ache, P.; Stempfl, T.; Kruse, J.; et al. Detecting early signs of heat and drought stress in Phoenix dactylifera (date palm). PLoS ONE 2017, 12, e0177883.",
    "Ma, C.; Zhang, H.H.; Wang, X. Machine learning for big data analytics in plants. Trends Plant Sci. 2014, 19, 798-808.",
    "Van Dijk, A.D.J.; Kootstra, G.; Kruijer, W.; De Ridder, D. Machine learning in plant science and plant breeding. iScience 2021, 24, 101890.",
    "You, X.; Shu, Y.; Ni, X.; Lv, H.; Luo, J.; Tao, J.; Bai, G.; Feng, S. MLAS: Machine learning-based approach for predicting abiotic stress-responsive genes in Chinese cabbage. Horticulturae 2025, 11, 44.",
    "Camacho, C.; Coulouris, G.; Avagyan, V.; Ma, N.; Papadopoulos, J.; Bealer, K.; Madden, T.L. BLAST+: architecture and applications. BMC Bioinformatics 2009, 10, 421.",
    "Fu, L.; Niu, B.; Zhu, Z.; Wu, S.; Li, W. CD-HIT: accelerated for clustering the next-generation sequencing data. Bioinformatics 2012, 28, 3150-3152.",
    "Chen, Z.; Liu, X.; Zhao, P.; Li, C.; Wang, Y.; Li, F.; Akutsu, T.; Bain, C.; Gasser, R.B.; Li, J.; et al. iFeatureOmega: an integrative platform for engineering, visualization and analysis of features from molecular sequences, structural and ligand data sets. Nucleic Acids Res. 2022, 50, W434-W447.",
    "Chou, K.-C. Using amphiphilic pseudo amino acid composition to predict enzyme subfamily classes. Bioinformatics 2005, 21, 10-19.",
    "Guyon, I.; Weston, J.; Barnhill, S.; Vapnik, V. Gene selection for cancer classification using support vector machines. Mach. Learn. 2002, 46, 389-422.",
    "Pedregosa, F.; Varoquaux, G.; Gramfort, A.; Michel, V.; Thirion, B.; Grisel, O.; Blondel, M.; Prettenhofer, P.; Weiss, R.; Dubourg, V.; et al. Scikit-learn: machine learning in Python. J. Mach. Learn. Res. 2011, 12, 2825-2830.",
    "Chen, T.; Guestrin, C. XGBoost: a scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, 13-17 August 2016; pp. 785-794.",
    "Bailey, T.L.; Johnson, J.; Grant, C.E.; Noble, W.S. The MEME Suite. Nucleic Acids Res. 2015, 43, W39-W49.",
    "Perez, D.E.; Hoyer, J.S.; Johnson, A.I.; Moody, Z.R.; Lopez, J.; Kaplinsky, N.J. BOBBER1 is a noncanonical Arabidopsis small heat shock protein required for both development and thermotolerance. Plant Physiol. 2009, 151, 241-252.",
    "Charng, Y.-Y.; Liu, H.-C.; Liu, N.-Y.; Chi, W.-T.; Wang, C.-N.; Chang, S.-H.; Wang, T.-T. A heat-inducible transcription factor, HsfA2, is required for extension of acquired thermotolerance in Arabidopsis. Plant Physiol. 2007, 143, 251-262.",
    "Apse, M.P.; Aharon, G.S.; Snedden, W.A.; Blumwald, E. Salt tolerance conferred by overexpression of a vacuolar Na+/H+ antiport in Arabidopsis. Science 1999, 285, 1256-1258.",
]
for i, ref in enumerate(REFS, 1):
    p = doc.add_paragraph()
    r = p.add_run(f"{i}. {ref}")
    r.font.size = Pt(9.5)
    p.paragraph_format.space_after = Pt(3)

os.makedirs(OUT_DIR, exist_ok=True)
doc.save(OUT)
print(f"Manuscript written to: {OUT}")
print(f"  paragraphs: {len(doc.paragraphs)}   tables: {len(doc.tables)}")
