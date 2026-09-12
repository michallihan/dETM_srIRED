# Figure 3C - background mutation frequency

Frequency of four mutations (H244Y, P203Q, A72T, G37C) across three nested founder
backgrounds.

## Input

```
01_fitness_score_processing/02a_high_confidence_fitness_scores/final_fitness_scores_ETM.csv
```

The `mut_id` column, wild-type excluded. Located through `repo_paths.py`, so the script
runs from any directory.

## Running

```
python3 1_plot_background_mutation_frequency.py
```

Backgrounds are exclusive — each variant is counted in the background of the last founder
it carries, so the three sets are disjoint:

| Background | Definition | n |
| --- | --- | --- |
| WT | neither T241A nor S68P | 193 |
| T241A | T241A, without S68P | 277 |
| S68P-T241A | both S68P and T241A | 117 |

The six variants carrying S68P without T241A match none of the three and are not plotted;
the script reports the count when it runs.

## Outputs

```
figure_3F_background_mutation_frequency.pdf / .png / .svg
figure_3F_source_data.csv    counts and percentages behind the figure
```
