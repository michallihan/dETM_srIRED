# Figure 1B - fitness replicate correlations

Plots the three sequencing replicates of the low-confidence fitness dataset against each
other, as a 3 x 3 grid: scatter plots below the diagonal, Pearson r, p and R-squared above
it, replicate names on the diagonal.

## Input

```
01_fitness_score_processing/02b_low_confidence_fitness_scores/final_fitness_scores_ETM.csv
```

Located through `repo_paths.py`, so the script runs from any directory.

## Running

```
python3 1_plot_replicate_correlations.py
```

Each replicate's value is the mean of its lenient- and stringent-gate fitness for that
variant (`fitness{n}_uncorr_lenient` and `fitness{n}_uncorr_stringent`). Where only one
gate survived QC, that gate's value is used. Pairs are dropped where either replicate is
missing.

## Output

```
replicate_plot.svg
```
