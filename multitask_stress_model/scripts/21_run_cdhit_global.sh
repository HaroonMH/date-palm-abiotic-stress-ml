#!/usr/bin/env bash
# Step 2: ONE global CD-HIT pass over the entire dataset (positives and
# negatives together) at 80% identity.
#
# Why this differs from the single-task pipeline, and why it matters:
# there, CD-HIT ran separately within each stress set. Redundancy was
# therefore removed *within* cold and *within* salt, but a cold protein and a
# near-identical salt protein both survived, and the two could land on
# opposite sides of the train/test boundary. With four independent models that
# was tolerable. With a SHARED backbone it is train/test leakage: the encoder
# would see a sequence in training and be evaluated on its near-duplicate.
#
# Running CD-HIT once over the union guarantees no two sequences in the final
# dataset exceed 80% identity, regardless of label.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABELS="$SCRIPT_DIR/../data/labels"
NR="$SCRIPT_DIR/../data/nr"

IN="$LABELS/dataset_all.fasta"
OUT="$NR/dataset_nr.fasta"

if [ ! -f "$IN" ]; then
    echo "ERROR: $IN not found -- run 20_build_multilabel_dataset.py first"
    exit 1
fi
if ! command -v cd-hit >/dev/null 2>&1; then
    echo "ERROR: cd-hit not on PATH. Install with: sudo apt-get install cd-hit"
    exit 1
fi

mkdir -p "$NR"

echo "Input sequences: $(grep -c '^>' "$IN")"
echo "Running global CD-HIT at c=0.8 (this covers ~46k sequences, allow a few minutes)..."

# -c 0.8 identity, -n 5 word size for that range, -M 0 unlimited memory,
# -T 0 all threads, -d 0 keep full header in the cluster file
cd-hit -i "$IN" -o "$OUT" -c 0.8 -n 5 -M 0 -T 0 -d 0

echo ""
echo "Retained after redundancy removal: $(grep -c '^>' "$OUT")"
echo "Cluster assignments: $OUT.clstr"
echo ""
echo "Next: python3 22_make_global_split.py"
