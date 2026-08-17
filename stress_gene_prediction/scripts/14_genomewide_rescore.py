#!/usr/bin/env python3
"""
Milestone 3, revised step 1: re-score the whole proteome, saving the SVM
decision-function margin alongside the probability.

Why this rerun is needed
------------------------
`predict_proba` on an SVC is Platt scaling -- a sigmoid of the decision value.
It is monotonic in that decision value, so it preserves ranking *in principle*,
but it saturates numerically: 811 cold proteins came back as exactly
0.99999999999997, with 60 of them tied at the top. The "top 10" was therefore
decided by sort order, not by the model.

The decision-function margin is the same quantity before the sigmoid, so it
gives an identical ordering with full floating-point resolution and no ties.
Probability is still written out (it stays the interpretable number to report),
but ranking now uses the margin.

This writes to data/genomewide_v2/ and does not touch the original
data/genomewide/ results. Resumable: existing chunk files are skipped.

Output:
  data/genomewide_v2/chunks/chunk_XXXX.tsv
  data/genomewide_v2/all_predictions.tsv   ProteinID, {stress}_prob, {stress}_margin
  data/genomewide_v2/skipped_sequences.txt
"""
import json
import os
import sys

import joblib
import pandas as pd

import iFeatureOmegaCLI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from fasta_utils import parse_fasta, write_fasta  # noqa: E402
from ml_utils import STRESSES, FEATURE_METHODS, DATA_DIR  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PROTEOME_FASTA = os.path.join(PROJECT_ROOT, "protein.faa")
FINAL_MODELS_DIR = os.path.join(DATA_DIR, "final_models")
GW_DIR = os.path.join(DATA_DIR, "genomewide_v2")
CHUNKS_DIR = os.path.join(GW_DIR, "chunks")
TMP_DIR = os.path.join(GW_DIR, "tmp")

CHUNK_SIZE = 1000
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 5


def sanitize(records):
    clean, skipped, modified = {}, [], 0
    for seq_id, (header, seq) in records.items():
        filtered = "".join(c for c in seq.upper() if c in STANDARD_AA)
        if len(filtered) != len(seq):
            modified += 1
        if len(filtered) < MIN_LENGTH:
            skipped.append(seq_id)
            continue
        clean[seq_id] = (header, filtered)
    return clean, skipped, modified


def load_models():
    models, needed = {}, set()
    for stress in STRESSES:
        model_path = os.path.join(FINAL_MODELS_DIR, f"{stress}_model.joblib")
        meta_path = os.path.join(FINAL_MODELS_DIR, f"{stress}_meta.json")
        if not (os.path.exists(model_path) and os.path.exists(meta_path)):
            sys.exit(f"ERROR: missing model for {stress} -- run 08_finalize_best_models.py first")
        with open(meta_path) as f:
            meta = json.load(f)
        models[stress] = (joblib.load(model_path), meta)
        needed.update(meta["feature_set"].split("+"))
    return models, needed


def features_for_chunk(chunk_fasta, needed_methods):
    out = {}
    for method in needed_methods:
        obj = iFeatureOmegaCLI.iProtein(chunk_fasta)
        obj.get_descriptor(FEATURE_METHODS[method])
        if obj.encodings is None:
            sys.exit(f"ERROR: descriptor {method} returned nothing for {chunk_fasta}")
        out[method] = obj.encodings
    return out


def main():
    for d in (GW_DIR, CHUNKS_DIR, TMP_DIR):
        os.makedirs(d, exist_ok=True)

    models, needed_methods = load_models()
    print(f"Descriptors required: {sorted(needed_methods)}")
    for stress, (_, meta) in models.items():
        print(f"  {stress:8s} {meta['feature_set']:12s} {meta['classifier']:5s} "
              f"({len(meta['selected_features'])} features)")
    print()

    print(f"Loading proteome: {PROTEOME_FASTA}")
    proteome = parse_fasta(PROTEOME_FASTA)
    proteome, skipped, modified = sanitize(proteome)
    print(f"  {modified} sequences had non-standard residues stripped")
    print(f"  {len(skipped)} sequences skipped (<{MIN_LENGTH} standard residues)")
    with open(os.path.join(GW_DIR, "skipped_sequences.txt"), "w") as f:
        f.write("\n".join(skipped) + ("\n" if skipped else ""))
    print(f"  {len(proteome)} sequences will be scored\n")

    all_ids = sorted(proteome.keys())
    n_chunks = (len(all_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_idx in range(n_chunks):
        out_path = os.path.join(CHUNKS_DIR, f"chunk_{chunk_idx:04d}.tsv")
        if os.path.exists(out_path):
            print(f"[{chunk_idx + 1}/{n_chunks}] already done, skipping")
            continue

        chunk_ids = all_ids[chunk_idx * CHUNK_SIZE:(chunk_idx + 1) * CHUNK_SIZE]
        chunk_fasta = os.path.join(TMP_DIR, f"chunk_{chunk_idx:04d}.fasta")
        write_fasta(proteome, chunk_fasta, ids=chunk_ids)

        feats = features_for_chunk(chunk_fasta, needed_methods)

        indexes = {m: list(df.index) for m, df in feats.items()}
        reference = indexes[sorted(indexes)[0]]
        for method, idx in indexes.items():
            if idx != reference:
                sys.exit(f"ERROR: descriptor {method} returned a different sequence set "
                         f"for chunk {chunk_idx}")

        result = pd.DataFrame(index=pd.Index(reference, name="ProteinID"))
        for stress, (pipe, meta) in models.items():
            methods = meta["feature_set"].split("+")
            X = pd.concat([feats[m] for m in methods], axis=1) if len(methods) > 1 else feats[methods[0]]
            X = X[meta["selected_features"]]
            if X.isna().any().any():
                sys.exit(f"ERROR: NaN features for {stress} in chunk {chunk_idx}")
            result[f"{stress}_prob"] = pipe.predict_proba(X)[:, 1]
            result[f"{stress}_margin"] = pipe.decision_function(X)

        result.to_csv(out_path, sep="\t")
        os.remove(chunk_fasta)
        del feats, result
        print(f"[{chunk_idx + 1}/{n_chunks}] scored {len(chunk_ids)} sequences")

    print("\nConcatenating chunks...")
    frames = [
        pd.read_csv(os.path.join(CHUNKS_DIR, f), sep="\t", index_col=0)
        for f in sorted(os.listdir(CHUNKS_DIR)) if f.endswith(".tsv")
    ]
    combined = pd.concat(frames)
    combined_path = os.path.join(GW_DIR, "all_predictions.tsv")
    combined.to_csv(combined_path, sep="\t")
    print(f"  {len(combined)} proteins scored -> {combined_path}")

    print("\nTie check (proteins sharing the single highest score):")
    for stress in STRESSES:
        prob = combined[f"{stress}_prob"]
        margin = combined[f"{stress}_margin"]
        n_prob_tied = int((prob >= prob.max() - 1e-12).sum())
        n_margin_tied = int((margin >= margin.max() - 1e-12).sum())
        print(f"  {stress:8s} by probability: {n_prob_tied:4d} tied   "
              f"by margin: {n_margin_tied:4d} tied   (max margin {margin.max():.4f})")

    print("\nNext step: python3 15_rank_candidates_v2.py")


if __name__ == "__main__":
    main()
