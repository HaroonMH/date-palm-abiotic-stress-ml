#!/usr/bin/env python3
"""
Step 4: extract ESM-2 protein language model embeddings -- the shared backbone.

Why a pretrained backbone rather than one trained from scratch: the reference
paper cautions that deep models "require a large amount of data" and that
classical ML suits small datasets better. That is true when the representation
must be learned from the task data alone. ESM-2 was pretrained on ~65M protein
sequences, so representation learning is already done; our ~46k proteins only
need to fit a small classification head. This is what makes a deep multi-task
architecture viable at this data scale.

Long sequences: ESM-2 accepts 1022 residues. 6.4% of date palm proteins are
longer (max 5,445 aa). Truncating would silently discard C-terminal domains,
so long sequences are instead split into overlapping windows and the residue
embeddings are mean-pooled across all windows.

Resumable: completed shards are skipped, so this can be interrupted safely.

Usage:
  python3 23_extract_esm_embeddings.py                     # training dataset
  python3 23_extract_esm_embeddings.py --proteome          # whole proteome
  python3 23_extract_esm_embeddings.py --model esm2_t30_150M_UR50D
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EMB_DIR, NR_DIR, PROTEOME_FASTA, parse_fasta, sanitize  # noqa: E402

WINDOW = 1022
OVERLAP = 256
SHARD_SIZE = 2000


def embed_batch(model, alphabet, batch_converter, seqs, device, repr_layer):
    """Mean-pooled per-protein embeddings for a list of (id, seq)."""
    _, _, tokens = batch_converter(seqs)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    reps = out["representations"][repr_layer]
    pooled = []
    for i, (_, seq) in enumerate(seqs):
        # positions 1..len(seq) exclude BOS/EOS
        pooled.append(reps[i, 1:len(seq) + 1].mean(0).float().cpu().numpy())
    return pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2_t33_650M_UR50D",
                    help="fair-esm model name (esm2_t30_150M_UR50D is a lighter option)")
    ap.add_argument("--proteome", action="store_true",
                    help="embed the full proteome instead of the training dataset")
    ap.add_argument("--token-budget", type=int, default=4096,
                    help="max residues per batch; lower this if you hit OOM")
    args = ap.parse_args()

    try:
        import esm
    except ImportError:
        sys.exit("ERROR: fair-esm not installed.\n"
                 "  pip install fair-esm\n"
                 "  pip install torch --index-url https://download.pytorch.org/whl/cu121")

    fasta = PROTEOME_FASTA if args.proteome else os.path.join(NR_DIR, "dataset_nr.fasta")
    tag = "proteome" if args.proteome else "dataset"
    if not os.path.exists(fasta):
        sys.exit(f"ERROR: {fasta} not found")

    out_dir = os.path.join(EMB_DIR, f"{tag}_{args.model}")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")
    print(f"Loading {args.model} ...")
    model, alphabet = esm.pretrained.__dict__[args.model]()
    batch_converter = alphabet.get_batch_converter()
    repr_layer = model.num_layers
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()

    records = parse_fasta(fasta)
    items = []
    for pid, (_, seq) in records.items():
        s = sanitize(seq)
        if s:
            items.append((pid, s))
    # sort by length so batches are homogeneous -> far less padding waste
    items.sort(key=lambda x: len(x[1]))
    print(f"{len(items)} sequences to embed from {os.path.basename(fasta)}")

    n_shards = (len(items) + SHARD_SIZE - 1) // SHARD_SIZE
    for shard in range(n_shards):
        emb_path = os.path.join(out_dir, f"shard_{shard:04d}.npz")
        if os.path.exists(emb_path):
            print(f"[{shard + 1}/{n_shards}] cached, skipping")
            continue

        chunk = items[shard * SHARD_SIZE:(shard + 1) * SHARD_SIZE]
        ids, vecs = [], []
        batch, budget = [], 0

        def flush(batch):
            if not batch:
                return
            short = [(i, s) for i, s in batch if len(s) <= WINDOW]
            long_ = [(i, s) for i, s in batch if len(s) > WINDOW]
            if short:
                for (pid, _), v in zip(short, embed_batch(
                        model, alphabet, batch_converter, short, device, repr_layer)):
                    ids.append(pid)
                    vecs.append(v)
            for pid, seq in long_:
                # overlapping windows, mean-pooled across windows
                wins, start = [], 0
                step = WINDOW - OVERLAP
                while start < len(seq):
                    wins.append((pid, seq[start:start + WINDOW]))
                    if start + WINDOW >= len(seq):
                        break
                    start += step
                parts = []
                for w in range(0, len(wins), 2):
                    parts.extend(embed_batch(model, alphabet, batch_converter,
                                             wins[w:w + 2], device, repr_layer))
                ids.append(pid)
                vecs.append(np.mean(parts, axis=0))

        for pid, seq in chunk:
            cost = min(len(seq), WINDOW)
            if batch and budget + cost > args.token_budget:
                flush(batch)
                batch, budget = [], 0
            batch.append((pid, seq))
            budget += cost
        flush(batch)

        np.savez_compressed(emb_path, ids=np.array(ids), emb=np.array(vecs, dtype=np.float32))
        print(f"[{shard + 1}/{n_shards}] embedded {len(ids)} sequences "
              f"(dim {len(vecs[0]) if vecs else '?'})")

    # ---- merge shards ----
    print("\nMerging shards...")
    all_ids, all_emb = [], []
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("shard_") and f.endswith(".npz"):
            z = np.load(os.path.join(out_dir, f), allow_pickle=True)
            all_ids.extend(z["ids"].tolist())
            all_emb.append(z["emb"])
    emb = np.concatenate(all_emb, axis=0)
    merged = os.path.join(EMB_DIR, f"{tag}_{args.model}.npz")
    np.savez_compressed(merged, ids=np.array(all_ids), emb=emb)
    print(f"  {emb.shape[0]} proteins x {emb.shape[1]} dims -> {merged}")
    print("\nNext: python3 24_train_multitask.py" if not args.proteome
          else "\nNext: python3 26_genomewide_multitask.py")


if __name__ == "__main__":
    main()
