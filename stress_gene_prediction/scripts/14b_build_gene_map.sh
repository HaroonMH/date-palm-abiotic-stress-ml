#!/usr/bin/env bash
# Milestone 3, revised step 1b: extract an authoritative protein -> GeneID map
# for Phoenix dactylifera from NCBI's gene2accession.
#
# Why: splice isoforms must be collapsed so the candidate table lists distinct
# GENES. Grouping by product description works for a top-10 table but
# over-merges paralogs genome-wide -- 105 separate proteins all carry the
# description "putative disease resistance protein RGA3", and they are
# different genes, not isoforms. GeneID is the real answer.
#
# This does one pass over a 29 GB file, so expect a few minutes. It only needs
# to be run once; 15_rank_candidates_v2.py picks the map up automatically and
# falls back to description grouping if it is absent.
#
# Phoenix dactylifera = NCBI taxonomy ID 42345.
# gene2accession columns: 1=tax_id 2=GeneID 6=protein_accession.version 16=Symbol
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GENE2ACC="$PROJECT_ROOT/Date_Palm_Project Data/DatePalm_Annotation/gene2accession"
OUT_DIR="$SCRIPT_DIR/../data/genomewide_v2"
OUT="$OUT_DIR/datepalm_gene_map.tsv"

TAXID=42345

if [ ! -f "$GENE2ACC" ]; then
    echo "ERROR: $GENE2ACC not found."
    echo "Skip this script -- 15_rank_candidates_v2.py will fall back to"
    echo "description-based isoform grouping."
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "Scanning gene2accession for taxid $TAXID (Phoenix dactylifera)..."
echo "This reads ~29 GB and takes a few minutes."

{
    printf 'ProteinID\tGeneID\tSymbol\n'
    awk -F'\t' -v taxid="$TAXID" \
        '$1 == taxid && $6 != "-" { print $6 "\t" $2 "\t" $16 }' "$GENE2ACC" \
        | sort -u
} > "$OUT"

n=$(( $(wc -l < "$OUT") - 1 ))
genes=$(tail -n +2 "$OUT" | cut -f2 | sort -u | wc -l)

echo ""
echo "Wrote $OUT"
echo "  $n protein->gene rows covering $genes distinct GeneIDs"

if [ "$n" -eq 0 ]; then
    echo ""
    echo "WARNING: no rows matched taxid $TAXID. The proteome may be annotated"
    echo "under a different taxid; check with:"
    echo "  grep -m1 XP_008775013.1 \"$GENE2ACC\""
    exit 1
fi

echo ""
echo "Next step: python3 15_rank_candidates_v2.py"
