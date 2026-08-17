#!/usr/bin/env bash
# Step 2 of Milestone 1: redundancy removal with CD-HIT (threshold 0.8),
# matching the reference paper's Section 2.1 ("CD-HIT tool ... applying a
# threshold of 0.8"). Run separately per stress, same as the paper.
#
# Prerequisite (one-time): sudo apt-get install cd-hit
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
RAW_DIR="$DATA_DIR/raw"
NR_DIR="$DATA_DIR/nr"

mkdir -p "$NR_DIR"

if ! command -v cd-hit >/dev/null 2>&1; then
    echo "ERROR: cd-hit not found on PATH. Install it with:"
    echo "  sudo apt-get install cd-hit"
    exit 1
fi

for stress in cold heat drought salt; do
    in_fasta="$RAW_DIR/${stress}_raw.fasta"
    out_fasta="$NR_DIR/${stress}_nr.fasta"
    if [ ! -f "$in_fasta" ]; then
        echo "ERROR: missing $in_fasta -- run 01_categorize_and_extract.py first"
        exit 1
    fi
    echo "=== CD-HIT: $stress ==="
    # -c 0.8 : 80% sequence identity threshold (matches reference paper)
    # -n 5   : word size recommended by CD-HIT for thresholds in [0.7, 1.0)
    # -M 0   : unlimited memory, -T 0 : use all available threads
    cd-hit -i "$in_fasta" -o "$out_fasta" -c 0.8 -n 5 -M 0 -T 0
    echo ""
done

echo "Done. Non-redundant FASTA files written to $NR_DIR"
echo "Next step: python3 03_build_pos_neg_split.py"
