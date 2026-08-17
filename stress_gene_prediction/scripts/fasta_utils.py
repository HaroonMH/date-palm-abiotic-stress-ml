"""Minimal FASTA read/write helpers (stdlib only, no biopython needed)."""


def parse_fasta(path):
    """Return dict: seq_id -> (header_line_without_gt, sequence)."""
    records = {}
    header = None
    seq_chunks = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seq_id = header.split()[0]
                    records[seq_id] = (header, "".join(seq_chunks))
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            seq_id = header.split()[0]
            records[seq_id] = (header, "".join(seq_chunks))
    return records


def write_fasta(records, path, ids=None, wrap=80):
    """records: dict seq_id -> (header, sequence). ids: optional subset/order."""
    ids = ids if ids is not None else list(records.keys())
    with open(path, "w") as f:
        for seq_id in ids:
            header, seq = records[seq_id]
            f.write(f">{header}\n")
            for i in range(0, len(seq), wrap):
                f.write(seq[i:i + wrap] + "\n")
