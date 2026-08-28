# Fitness score processing

Turns the DiMSum variant tables into the fitness scores every downstream analysis uses.

```
02a_high_confidence_fitness_scores/   all four QC steps applied      ->  594 variants
02b_low_confidence_fitness_scores/    indel/stop filter only         ->  966 variants
```

Each folder is self-contained: run its `1_process_fitness_scores_lrDMS_V4.py` from anywhere
and it reads its own `lrDMS_all_variants_lenient.csv` and `..._stringent.csv`, writing
`final_fitness_scores_ETM.csv` plus `diagnostic_plots/` beside itself. The two scripts
differ only in their QC cutoffs (see the docstring at the top of each); the shared
`function_library_lrDMS_analysis.py` is identical between them.

`02a_high_confidence_fitness_scores/` also holds
`final_fitness_scores_ETM_with_kcat_app.csv` - the same table with modelled apparent kcat
(`kcat_app`, `kcat_app_error`) joined on. That file is the input to the Figure 2B fitness
landscape. It is a join, not a separate calculation: every other column is byte-identical
to `final_fitness_scores_ETM.csv`, so re-running the script and re-doing the join
reproduces it.

## How fitness_final is computed

Each variant is scored under two sorting gates, lenient and stringent, and each gate is
measured across three replicates. `fitness_final` combines the two **gates**:

```
fitness_final = log_average(fitness_lenient, fitness_stringent)
```

where `log_average` exponentiates, takes the mean, and logs again. Where only one gate
survived QC, `log_average` returns that gate's value unchanged - so the 389 variants
marked `only lenient gate available` have `fitness_final == fitness_lenient` exactly.
`sigma_final` is combined the same way.

The per-replicate `fitness{n}_uncorr_combined` columns combine the two gates *within* each
replicate. They are diagnostics, feeding the replicate-correlation plot; they are not the
route to `fitness_final`.

### A correction, and why it matters

An earlier revision of this script derived `fitness_final` the other way round - combining
gates within each replicate first, then averaging the three per-replicate values. That is a
different quantity, not a rounding difference: it shifted 576 of the 594 high-confidence
variants, by up to 1.06 in `fitness_final`, and left `sigma_final` correlating only 0.67
with the published values.

The published fitness scores, and every downstream analysis built on them, use the
gate-level combination documented above. The script now implements that, and reproduces the
published scores exactly: agreement is 5e-10, which is *closer than the published file
agrees with itself* when `fitness_final` is recomputed from its own stored
`fitness_lenient` and `fitness_stringent` columns (8e-10). The residual is the original
file's decimal storage, not a difference in calculation.

Only `fitness_final` and `sigma_final` were affected. All 40 other columns, the variant
sets, and the `decision` assignments were identical before and after, at both confidence
levels.
