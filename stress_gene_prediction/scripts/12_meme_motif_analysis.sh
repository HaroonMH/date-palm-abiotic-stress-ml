#!/usr/bin/env bash
# Milestone 3, step 3: discriminative motif discovery with MEME on the
# positive vs. negative sequence sets (pipeline doc Phase 3; reference paper
# Section 3.1 / Figure 4).
#
# The paper ran MEME with the number of motifs preset to 3 and motif widths
# from 6 to 10, comparing each stress's positive set against its negative set,
# then kept the two most significant motifs per stress. `-objfun de` with
# `-neg` is MEME's discriminative mode, which is what "discriminative motif
# discovery" in the paper's Figure 4 refers to.
#
# PREREQUISITE -- MEME Suite is not packaged in apt, build it from source once:
#
#   cd ~
#   wget https://meme-suite.org/meme/meme-software/5.5.7/meme-5.5.7.tar.gz
#   tar zxf meme-5.5.7.tar.gz && cd meme-5.5.7
#   ./configure --prefix=$HOME/meme --enable-build-libxml2 --enable-build-libxslt \
#               --enable-serial
#   make && make install
#   echo 'export PATH=$HOME/meme/bin:$HOME/meme/libexec/meme-5.5.7:$PATH' >> ~/.bashrc
#   source ~/.bashrc
#
# --enable-serial matters: if openmpi-bin is installed without libopenmpi-dev,
# configure enables the MPI build and it then dies on a missing mpi.h. MEME's
# MPI mode buys nothing at our scale (a few hundred sequences per stress).
#
# (The build takes roughly 10 minutes. If you would rather not build it, the
# same analysis can be run on the MEME web server at https://meme-suite.org --
# upload data/meme/{stress}_positive.fasta as the primary sequences and
# {stress}_negative.fasta as the control, mode "Discriminative", protein
# alphabet, 3 motifs, widths 6-10.)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
NR_DIR="$DATA_DIR/nr"
MEME_DIR="$DATA_DIR/meme"

STRESSES=(cold heat drought salt)

mkdir -p "$MEME_DIR"

# 1. Build the positive / negative FASTA pair for each stress, using the same
#    definition as Milestone 1: positive = that stress's non-redundant set,
#    negative = the union of the other three.
for stress in "${STRESSES[@]}"; do
    if [ ! -f "$NR_DIR/${stress}_nr.fasta" ]; then
        echo "ERROR: $NR_DIR/${stress}_nr.fasta missing -- run 02_run_cdhit.sh first"
        exit 1
    fi
    cp "$NR_DIR/${stress}_nr.fasta" "$MEME_DIR/${stress}_positive.fasta"

    : > "$MEME_DIR/${stress}_negative.fasta"
    for other in "${STRESSES[@]}"; do
        if [ "$other" != "$stress" ]; then
            cat "$NR_DIR/${other}_nr.fasta" >> "$MEME_DIR/${stress}_negative.fasta"
        fi
    done

    pos_n=$(grep -c "^>" "$MEME_DIR/${stress}_positive.fasta")
    neg_n=$(grep -c "^>" "$MEME_DIR/${stress}_negative.fasta")
    echo "$stress: positive=$pos_n negative=$neg_n"
done
echo ""

if ! command -v meme >/dev/null 2>&1; then
    echo "MEME is not on PATH -- the positive/negative FASTA files above are ready"
    echo "in $MEME_DIR, but motif discovery was not run."
    echo "Install MEME (see the header of this script) or use the web server."
    exit 1
fi

# 2. Discriminative motif discovery, per stress.
for stress in "${STRESSES[@]}"; do
    echo "=== MEME: $stress ==="
    meme "$MEME_DIR/${stress}_positive.fasta" \
        -neg "$MEME_DIR/${stress}_negative.fasta" \
        -objfun de \
        -protein \
        -mod zoops \
        -nmotifs 3 \
        -minw 6 -maxw 10 \
        -oc "$MEME_DIR/${stress}_meme_out"
    echo "  -> $MEME_DIR/${stress}_meme_out/meme.html (logos + E-values)"
    echo ""
done

echo "Done. Each ${stress}_meme_out/ holds meme.html (with sequence logos),"
echo "meme.txt, and meme.xml. Pick the two most significant motifs per stress"
echo "for the manuscript figure, as the reference paper did."
