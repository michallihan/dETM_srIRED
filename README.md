# Laboratory evolution relies on computationally unpredictable epistatic trajectories


(Nintzel #, Lihan #, Penner #, Sieders, Gantz, Bormke, Sifri, Medcalf, Mathis, Cao,  Hollfelder *)


We carried out a directed evolution campaign on the enzyme biocatalyst SrIRED. We then ran deep sequence-function mapping to map the fitness of every variant that was selected during the campaign. This allows us to retrospectively study how mutations combine and compete, forming the basis for testing design principles for directed evolution campaigns.

We go on to test state-of-the-art methods in mutational effect prediction. We find that while they are generally good at identifying dead from alive enzymes, none effectively rank mutations in the campaign by their relative fitness. Even supervised learning approaches cannot improve prediction beyond random. 

We argue that predicting evolutionary trajectories in enzyme evolution campaigns remains an unsolved problem, and that carrying out retrospective fitness mapping of these campaigns allows the design choices taken in the campaign to be evaluated, to iteratively improve directed evolution success rates over time.

This is the codebase supporting the above manuscript. Here, you will find code to replicate and build on the analysis we carried out, as well as data in various stages of processing.

## Structure

```
01_fitness_score_processing/   Fitness scores from lrDMS
  01_lrDMS_pipeline/             raw reads to per-variant counts, via DiMSum
    01_NGS/                      download and merge the Illumina reads
    02_DiMSum_DMS/               one run per droplet gate, gives the fitness scores
    03_DiMSum_population_dynamics/  one run per round of the campaign, gives per-round counts
  02a_high_confidence_fitness_scores/
  02b_low_confidence_fitness_scores/
  02c_inactive_variants/
02_long_read_processing/       Nanopore long-read processing
03_analysis/                   One folder per figure panel
  Figure_1/  Figure_2/  Figure_3/  Figure_4/  SI_figures/
04_biophysics/                 wet-lab, low-throughput measurements on purified variants
```

![3D fitness landscape of the SrIRED campaign, from WT through T241A and S68P-T241A to S68P-P203Q-T241A-H244Y](03_analysis/Figure_2/2B_fitness_landscape/figures_fitness_landscape/3D_final_fitness_landscape.png)
