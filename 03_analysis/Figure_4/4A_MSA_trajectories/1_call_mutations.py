"""
Reduces both sequence sets to per-sequence mutation calls against the wild-type IRED.

The campaign set is read from the collapsed unique-variant table in 02_long_read_processing
and every differing position is reported. The alignment set is read from the FASTA of BLAST
hits in this folder and is called only at the tracked positions, identity calls included;
see path_dependency.py for why the two rules differ.

Outputs:
    dETM/called_mutations.csv
    MSA/called_mutations.csv
"""

import csv

import path_dependency as pd_


def call_detm():
    """Add a mutation column to the collapsed unique-variant table."""
    source = pd_.DETM_SOURCE_CSV
    target = pd_.called_mutations_csv("dETM")
    target.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with open(source, newline="") as infile, open(target, "w", newline="") as outfile:
        reader, writer = csv.reader(infile), csv.writer(outfile)
        header = next(reader)
        writer.writerow(header + ["mutation"])
        for row in reader:
            mutations = pd_.call_all_differences(row[pd_.DETM_SEQUENCE_COLUMN])
            writer.writerow(row + ["-".join(mutations) if mutations else "No mutation"])
            rows += 1
    return rows, source, target


def read_fasta(path):
    """Yield (header, sequence) pairs, sequences joined across wrapped lines."""
    header, parts = None, []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header, parts = line[1:], []
            else:
                parts.append(line)
    if header is not None:
        yield header, "".join(parts)


def call_msa():
    """Call the tracked positions for every BLAST hit."""
    source = pd_.MSA_SOURCE_FASTA
    target = pd_.called_mutations_csv("MSA")
    target.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with open(target, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Header", "Sequence", "mutation"])
        for header, sequence in read_fasta(source):
            calls = pd_.call_at_positions(sequence)
            writer.writerow([header, sequence, "-".join(calls)])
            rows += 1
    return rows, source, target


def main():
    for rows, source, target in (call_detm(), call_msa()):
        print(f"{source.name} -> {target.parent.name}/{target.name}: {rows} sequences")


if __name__ == "__main__":
    main()
