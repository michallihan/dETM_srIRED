"""
Narrows each mutation-called table to the nested backgrounds along its trajectory.

Each background is filtered out of the one before it, so a table holds the sequences
carrying every mutation up to that point of the trajectory: for the campaign,
T241A, then S68P within T241A, then P203Q within both. The alignment branches in two
after T241A. The chains are defined in path_dependency.py.

Outputs, per branch:
    <branch>/called_mutations_in_<mutation>.csv
"""

import csv

import path_dependency as pd_


def filter_on(source, target, mutation):
    """Copy the rows of `source` whose mutation column contains `mutation`."""
    with open(source, newline="", encoding="utf-8-sig") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        header[0] = header[0].lstrip("﻿")
        column = header.index("mutation")
        rows = [row for row in reader if len(row) > column and mutation in row[column]]

    with open(target, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(rows)
    return len(rows)


def main():
    for branch in pd_.BRANCHES:
        total = sum(1 for _ in open(pd_.called_mutations_csv(branch))) - 1
        print(f"{branch}: {total} sequences")
        for mutation, background in pd_.BACKGROUND_CHAINS[branch]:
            source = pd_.called_mutations_csv(branch, background)
            target = pd_.called_mutations_csv(branch, mutation)
            kept = filter_on(source, target, mutation)
            within = background or "the full set"
            print(f"  {mutation:6s} in {within:12s} {kept:5d} -> {target.name}")


if __name__ == "__main__":
    main()
