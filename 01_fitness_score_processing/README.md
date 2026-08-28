# Fitness score processing

Turns the DiMSum variant tables into the fitness scores the downstream analyses use.

```
02a_high_confidence_fitness_scores/   all four QC steps applied   ->  594 variants
02b_low_confidence_fitness_scores/    indel/stop filter only      ->  966 variants
```

Each folder is self-contained: run its `1_process_fitness_scores_lrDMS_V4.py` and it reads
its own `lrDMS_all_variants_lenient.csv` and `..._stringent.csv`, writing
`final_fitness_scores_ETM.csv` and `diagnostic_plots/` beside itself. The two scripts differ
only in their QC cutoffs, set at the bottom of each; the shared
`function_library_lrDMS_analysis.py` is identical between them.

Each variant is scored under two sorting gates, lenient and stringent, across three
replicates. The final score combines the two gates:

```
fitness_final = log_average(fitness_lenient, fitness_stringent)
```

`sigma_final` likewise. Where only one gate survived QC, that gate's value is used.

`02a_high_confidence_fitness_scores/` also holds `final_fitness_scores_ETM_with_kcat_app.csv`
- the same table with modelled apparent kcat (`kcat_app`, `kcat_app_error`) joined on. It is
the input to the Figure 2B fitness landscape.
