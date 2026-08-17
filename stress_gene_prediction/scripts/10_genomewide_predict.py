#!/usr/bin/env python3
"""
Milestone 3, step 1: apply the finalized per-stress models to ALL date palm
protein-coding genes for genome-wide stress-gene prediction (pipeline doc
Phase 3; reference paper Section 3.4).

The full proteome (48,801 sequences) is processed in chunks so peak memory
stays low -- TPC alone is 8,000 features, so materializing the whole proteome
at once would need many GB. For each chunk we:
  1. write the chunk to a temporary FASTA
  2. run only the iFeatureOmega descriptors the final models actually need
     (read from final_models/{stress}_meta.json -- currently TPC for
     cold/heat/drought and CKSAAP+TPC for salt)
  3. subset to each model's RFE-selected columns and predict a probability
  4. write that chunk's probabilities to disk and free the features

The run is RESUMABLE: a chunk whose output file already exists is skipped, so
you can safely Ctrl-C and rerun.

Sequence sanitation: residues outside the 20 standard amino acids (X, B, Z,
U, *, etc.) are stripped, since the k-mer descriptors are only defined over
the standard alphabet. Sequences left with <5 residues are skipped. Both
counts are reported and the skipped IDs are written out.

Output:
  data/genomewide/chunks/chunk_XXXX.tsv   per-chunk probabilities (intermediate)
  data/genomewide/all_predictions.tsv     ProteinID + one probability column per stress
  data/genomewide/skipped_sequences.txt   IDs skipped as too short / unusable
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
GW_DIR = os.path.join(DATA_DIR, "genomewide")
CHUNKS_DIR = os.path.join(GW_DIR, "chunks")
TMP_DIR = os.path.join(GW_DIR, "tmp")

CHUNK_SIZE = 1000
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH = 5


def sanitize(records):
    """Strip non-standard residues; drop sequences left too short."""
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
    """Return {stress: (pipeline, meta)} plus the set of descriptors needed."""
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
    """Run the required descriptors on one chunk -> {method: DataFrame}."""
    out = {}
    for method in needed_methods:
        descriptor = FEATURE_METHODS[method]
        obj = iFeatureOmegaCLI.iProtein(chunk_fasta)
        obj.get_descriptor(descriptor)
        if obj.encodings is None:
            sys.exit(f"ERROR: descriptor {descriptor} returned nothing for {chunk_fasta}")
        out[method] = obj.encodings
    return out


def main():
    for d in (GW_DIR, CHUNKS_DIR, TMP_DIR):
        os.makedirs(d, exist_ok=True)

    models, needed_methods = load_models()
    print(f"Descriptors required by the final models: {sorted(needed_methods)}")
    for stress, (_, meta) in models.items():
        print(f"  {stress:8s} {meta['feature_set']:12s} {meta['classifier']:5s} "
              f"({len(meta['selected_features'])} features)")
    print()

    print(f"Loading proteome: {PROTEOME_FASTA}")
    proteome = parse_fasta(PROTEOME_FASTA)
    print(f"  {len(proteome)} sequences")

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

        # All descriptors must cover exactly the same sequences, otherwise a
        # concat would align on index and silently inject NaNs into the model.
        indexes = {m: list(df.index) for m, df in feats.items()}
        reference = indexes[sorted(indexes)[0]]
        for method, idx in indexes.items():
            if idx != reference:
                sys.exit(f"ERROR: descriptor {method} returned a different sequence set "
                         f"for chunk {chunk_idx} ({len(idx)} vs {len(reference)} rows)")
        if len(reference) != len(chunk_ids):
            print(f"  NOTE: {len(chunk_ids) - len(reference)} sequence(s) in this chunk "
                  f"were dropped by iFeatureOmega")

        result = pd.DataFrame(index=pd.Index(reference, name="ProteinID"))
        for stress, (pipe, meta) in models.items():
            methods = meta["feature_set"].split("+")
            X = pd.concat([feats[m] for m in methods], axis=1) if len(methods) > 1 else feats[methods[0]]
            X = X[meta["selected_features"]]
            if X.isna().any().any():
                sys.exit(f"ERROR: NaN features for {stress} in chunk {chunk_idx}")
            result[f"{stress}_prob"] = pipe.predict_proba(X)[:, 1]

        result.to_csv(out_path, sep="\t")
        os.remove(chunk_fasta)
        del feats, result
        print(f"[{chunk_idx + 1}/{n_chunks}] scored {len(chunk_ids)} sequences -> {out_path}")

    print("\nConcatenating chunks...")
    frames = [
        pd.read_csv(os.path.join(CHUNKS_DIR, f), sep="\t", index_col=0)
        for f in sorted(os.listdir(CHUNKS_DIR)) if f.endswith(".tsv")
    ]
    combined = pd.concat(frames)
    combined_path = os.path.join(GW_DIR, "all_predictions.tsv")
    combined.to_csv(combined_path, sep="\t")
    print(f"  {len(combined)} proteins scored -> {combined_path}")

    print("\nPredicted stress-responsive proteins at probability > 0.5:")
    for stress in STRESSES:
        n = int((combined[f"{stress}_prob"] > 0.5).sum())
        print(f"  {stress:8s} {n:6d} / {len(combined)}")

    print("\nNext step: python3 11_rank_candidates.py")


if __name__ == "__main__":
    main()
