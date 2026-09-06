"""Shared paths, constants and helpers for the multi-task stress model."""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE, ".."))

DATA = os.path.join(BASE, "data")
LABELS_DIR = os.path.join(DATA, "labels")
NR_DIR = os.path.join(DATA, "nr")
EMB_DIR = os.path.join(DATA, "embeddings")
SPLIT_DIR = os.path.join(DATA, "splits")
MODELS_DIR = os.path.join(BASE, "models")
TABLES_DIR = os.path.join(BASE, "outputs", "tables")
FIGURES_DIR = os.path.join(BASE, "outputs", "figures")

PROTEOME_FASTA = os.path.join(PROJECT_ROOT, "protein.faa")
ANNOT_DIR = os.path.join(PROJECT_ROOT, "Date_Palm_Project Data", "DatePalm_Annotation")
GO_FULL_TSV = os.path.join(ANNOT_DIR, "Results files", "abiotic_stress_GO_full.tsv")
GO_ANNOTATIONS = os.path.join(ANNOT_DIR, "datepalm_GO_annotations.tsv")
GENE_MAP = os.path.join(PROJECT_ROOT, "stress_gene_prediction", "data",
                        "genomewide_v2", "datepalm_gene_map.tsv")

STRESSES = ["cold", "heat", "drought", "salt"]

# Same stress-specific GO term sets used in the single-task pipeline, so the
# two approaches are compared on identical label semantics.
STRESS_GO = {
    "cold": {"GO:0009409", "GO:0009631", "GO:0070417", "GO:0050826", "GO:0071497"},
    "heat": {"GO:0009408", "GO:0010286", "GO:0034605", "GO:0070370"},
    "drought": {"GO:0009414", "GO:0042631", "GO:0009819",
                "GO:2000070", "GO:0080148", "GO:1902584"},
    "salt": {"GO:0009651", "GO:1902074", "GO:0042538", "GO:0071472",
             "GO:1901000", "GO:1901001", "GO:1901002"},
}

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
RANDOM_STATE = 42

# Okabe-Ito, consistent with the single-task figures
OKABE_ITO = ["#0072B2", "#D55E00", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]
STRESS_COLORS = dict(zip(STRESSES, OKABE_ITO[:4]))


def parse_fasta(path):
    """seq_id -> (header, sequence)."""
    recs, header, chunks = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    recs[header.split()[0]] = (header, "".join(chunks))
                header, chunks = line[1:], []
            else:
                chunks.append(line)
        if header is not None:
            recs[header.split()[0]] = (header, "".join(chunks))
    return recs


def write_fasta(records, path, ids=None, wrap=80):
    ids = ids if ids is not None else list(records)
    with open(path, "w") as f:
        for i in ids:
            header, seq = records[i]
            f.write(f">{header}\n")
            for k in range(0, len(seq), wrap):
                f.write(seq[k:k + wrap] + "\n")


def sanitize(seq):
    return "".join(c for c in seq.upper() if c in STANDARD_AA)


def label_combo(row):
    """'1010' style key for stratification / reporting."""
    return "".join(str(int(row[s])) for s in STRESSES)
