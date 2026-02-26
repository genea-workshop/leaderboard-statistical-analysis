# Analyzing the user study results
This repository provides Python scripts for statistical analysis of user study results from the GENEA leaderboard pipeline. It supports:
- ELO score computation
- Mismatch analysis

## Setup
```bash
conda create -n genea_analyze python=3.12.12 -y
conda activate genea_analyze
conda install --file requirements.txt -y
```

## Battles CSV
Sample battles `.csv` files have been provided to run the elo and mismatching studies. They are named `battles_elo.csv` and `battles_mismatch.csv` respsectively.

Note that the files may not contain all up-to-date models that have been evaluated and some visualizations may show gaps in data.

## Compute ELO score
Compute ELO scores by running `python -m elo` .

This information gets printed out to console:
- various debug logs
- the ELO ratings
- number of model pair comparisons

The following files are saved to `output/elo/`:
- `battles.csv` - a CSV of the battles
- `battle_outcomes_pie.html` - battle outcome statistics
- `battle_and_tie_count_heatmaps_combined.html` - for each combination of models 1) battle count without ties, 2) tie count
- `total_battle_count_heatmap.html` - battle count of each model combinations
- `model_battle_counts_bar.html` - per-model battle count
- `juice_realism_results.png / .pdf` - JUICE votes
- `pairwise_win_fraction_heatmap.html` - fraction of model A wins for all non-tied A vs. B battles
- `motion_realism_elo_results.png / .pdf` - motion realism ELO after bootstrapping
- `bootstrap_elo_lu.csv` - bootstrap samples of Elo scores expressed in logit (log-odds) units

For the evaluation, the most important info is:
- the printed ELO scores
- JUICE votes - `output/elo/juice_realism_results.png`
- potentially useful are `output/elo/total_battle_count_heatmap.html` and `output/elo/model_battle_counts_bar.html`

## Mismatch analysis
Analyze mismatch statistics by running `python -m mismatch` .

This information gets printed out to console:
- various debug logs
- pair comparisons vetween original and mismatched models

The following files are saved to `output/mismatch/`:
- `battles.csv` - a CSV of the battles
- `battle_outcomes_pie.html` - battle outcome statistics
- `battle_and_tie_count_heatmaps_combined.html` - for each combination of models 1) battle count without ties, 2) tie count
- `model_battle_counts_bar.html` - per-model battle count
- `total_battle_count_heatmap.html` - battle count of each model combinations
- `juice_mismatch_results.png / .pdf` - JUICE votes
- `mismatching_pref_results_split_ties.png / .pdf` - bootstrap confidence intervals for BT scores
- `preference_ratio_mismatch.png` - preference ratio for model outputs over mismatched stimuli
