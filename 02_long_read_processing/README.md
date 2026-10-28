# Nanopore long-read processing

The barcoded nanopore reads to the variant dictionary the rest of the repo builds on.
`1_` through `11_` are the stages, in order. Each takes `--dry-run`, which prints the
commands it would issue and checks its inputs without running anything.

The reads are not in this repo, so a full pass is not possible from a clone alone. `10_`
and `11_` run from the committed per-round tables and reproduce the two dictionary files
exactly, and are the two stages that had no script before.

## Where the published dictionary came from

It is not the output of a single run. Rounds 3, 4 and 5 come from the June 2023 sequencing,
basecalled in July. Round 2 was re-sequenced in October 2023 and the two read sets joined
before clustering. Round 1 was re-clustered in December 2023 from the same UMI file as the
July run.

| Round | Sequencing | Clustered | Rows |
| --- | --- | --- | --- |
| 1 | June 2023 | December 2023 | 660 |
| 2 | June + October 2023, joined | December 2023 | 486 |
| 3 | June 2023 | August 2023 | 1392 |
| 4 | June 2023 | August 2023 | 1104 |
| 5 | June 2023 | August 2023 | 1694 |

Round 2 therefore mixes two read pools that were filtered on different length windows.

## Stages

| | Script | Needs | Notes |
| --- | --- | --- | --- |
| 1 | `1_filter_reads.py` | NanoFilt | length and quality filter; `--dataset` picks the window |
| 2 | `2_fastq_to_fasta.py` | — | plain Python; the original used EMBOSS `seqret` |
| 3 | `3_demultiplex.py` | UMIC-seq (vendored) | splits reads by round barcode |
| 4 | `4_extract_umis.py` | UMIC-seq (vendored) | pulls the UMI off each read |
| 5 | `5_join_round2_reads.py` | — | round 2 only: initial + re-sequencing |
| 6 | `6_cluster_umis.py` | MMseqs2 | clusters UMIs into one group per variant |
| 7 | `7_split_clusters.py` | — | one FASTA per cluster, within the size window |
| 8 | `8_generate_consensus.py` | minimap2, racon, medaka, `vcf2fasta.py` | writes `auto.sh`, one block of six commands per cluster |
| 9 | `9_call_mutations.py` | `UMI_mut_patt_match_V6_Rike1.py` | calls each consensus against the reference |
| 10 | `10_build_dictionary.py` | — | joins the five per-round tables |
| 11 | `11_collapse_unique_variants.py` | — | one row per unique intact protein |

`inputs/` holds the barcodes, probes, reference and per-round wild-type sequences these
stages need. `work/` is where stage outputs go and is not tracked.

Two scripts are not in the repo and are not ours to redistribute: `vcf2fasta.py` and
`UMI_mut_patt_match_V6_Rike1.py`. Put them in `third_party/` at the repo root, as the
Figure 4D scripts do with theirs. UMIC-seq is already vendored under the lrDMS pipeline.

## Parameters

`pipeline_settings.py` holds them, each marked with where it came from.

Recovered from the run's own outputs:

| Parameter | Value | Evidence |
| --- | --- | --- |
| NanoFilt length, campaign | 1500–3000 | min and max read length in `trimmed_reads.fastq` |
| NanoFilt quality, campaign | 8 | minimum of NanoFilt's average read quality over 100,000 reads |
| NanoFilt length, re-sequencing | 2000–3500 | min and max in `reads__round2_resequences.fasta` |
| `--umi_len` | 88 | the UMI FASTAs wrap to 60 + 28 characters |
| cluster size window | 20 and up | only `>=20` gives the cluster counts that were written |
| medaka model | `r1041_e82_400bps_sup_v4.2.0` | named in `auto.sh` |
| NanoPlot | 1.41.6 | its own log |

From the lab handbooks only, with no surviving log to check them against: `--min_probe_score 60`,
and MMseqs2's `-c 0.8 --cov-mode 0 --min-seq-id 0.8 --seq-id-mode 2`. The MMseqs2 identity
can be settled by re-running the clustering at 0.8 and at 0.85 and seeing which reproduces
the 72,439 clusters in round 1's `clusters_r1.tsv`.

Not recorded anywhere: the quality cut-off for the October re-sequencing. Only FASTA
survives for that run, and FASTA carries no quality scores. `1_filter_reads.py` requires
`--quality` for that dataset rather than guessing.

## What 10_ and 11_ reproduce

`10_` is a straight concatenation of the five `full_table_barcodes_mutations.csv` files with
a `round` column in front. Nothing is filtered. Rebuilt from the five source tables it
matches the published dictionary in every column, with one exception: the published file is
missing one round-3 row (`cluster_148`, a `no_translation` row). Nothing in the data
distinguishes that row from the 23 other `no_translation` rows that were kept, so the
script does not reproduce the omission.

`11_` keeps the rows whose protein is intact — 291 residues, no premature stop in
`aa_mutations`, no insertions or deletions — and groups them by amino-acid sequence. That
excludes 359 of 5336 rows and leaves 1342 variants over 4977 occurrences, matching the
published file in every column except `name`: the `var_N` numbering follows the order
variants are first seen, which is not the order in the published file. No script in the
repo reads that column.
