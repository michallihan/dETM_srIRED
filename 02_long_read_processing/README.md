# Nanopore long-read processing

Output of the UMIC-seq nanopore pipeline for the CHX/CPA evolution campaign, rounds 1-5.
This is the backbone dataset the Figure 2, 3 and 4 analyses build on.

`Rike_IRED_evolution_CHX_CPA_dictionary.csv` - the full dictionary, one row per UMI cluster
(5,336 rows), each with its round, UMI, nucleotide and amino-acid mutation calls, and
sequences.

`Rike_IRED_evolution_CHX_CPA_dictionary_collapsed_unique_variants.csv` - the same data
collapsed to 1,342 unique amino-acid variants (1,341 mutants plus wild-type) with per-round
occurrence counts. This is the file the analyses read.

An occurrence is one UMI cluster, that is one barcoded molecule that received a consensus
sequence. `total_occurrences` is the sum of `round_1` ... `round_5`.
