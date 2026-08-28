# Nanopore long-read processing

Output of the UMIC-seq nanopore pipeline for the CHX/CPA evolution campaign (rounds 1-5).
These two files are the backbone dataset the Figure 2, 3 and 4 analyses build on.

## Files

`Rike_IRED_evolution_CHX_CPA_dictionary.csv` - the full dictionary, one row per UMI
cluster (5,336 rows). Each row is a single barcoded molecule that received a consensus
sequence, with its round, UMI, nucleotide and amino-acid mutation calls, and sequences.

`Rike_IRED_evolution_CHX_CPA_dictionary_collapsed_unique_variants.csv` - the same data
collapsed to 1,342 unique amino-acid variants (1,341 mutants plus wild-type), with
per-round occurrence counts. This is the file the analyses read.

## Relationship between the two

The collapsed table is the full dictionary grouped by amino-acid variant and round. The
counts reproduce exactly, for every one of the 1,342 variants, after two filters:

- 318 clusters are dropped as untranslatable: 294 `frameshift`, 24 `no_translation`,
  leaving 5,018.
- 34 variants (41 clusters) are dropped because they carry a premature stop codon. Most
  are singletons.

`total_occurrences` is the row sum of `round_1` ... `round_5`, exact for every row. It is
not an independent measurement.

An occurrence is therefore one UMI cluster - one barcoded molecule - not a read count.

No script for this collapse survives; the relationship above was established by
reproducing it from the dictionary.

## Two things to be aware of

**Round 2 does not reconcile with the pipeline summary.** Cluster counts per round in the
dictionary are 660 / 486 / 1392 / 1104 / 1694. The step 8 summary dated 02/09/2023 in the
processing pipeline reported 662 / 228 / 1394 / 1104 / 1694. Rounds 4 and 5 match, rounds
1 and 3 differ by two, but round 2 has 486 clusters against 228 reported. The run in
`2023_07_18_Basecalling` cannot have produced 486 round-2 clusters, so round 2 was
evidently re-sequenced, supplemented, or regenerated after that summary was written. This
should be resolved before any round-2 figure is quoted.

**Sampling depth varies by round.** 660 to 1,694 clusters, close to threefold. Raw
occurrence counts are not comparable between rounds without normalising - relevant to
anything that reads abundance as a function of round.
