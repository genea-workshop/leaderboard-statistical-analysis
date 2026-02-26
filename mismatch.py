import os

import pandas as pd

import helpers
from visualization_elo import print_pair_comparisons, visualize_battle_statistics
from visualization_mismatch import (
    visualize_bootstrap_scores_vertical,
    visualize_juice,
    visualize_preference_ratio,
)

output_dir = "output/mismatch"
os.makedirs(output_dir, exist_ok=True)

# ------------------
# PREPARE BATTLES
# ------------------
pd.options.display.float_format = "{:.2f}".format
if True:
    # Read the battles.csv file directly
    battles_csv = "battles_mismatch.csv"
    battles = pd.read_csv(battles_csv)
else:
    STUDY_NAME = ["MM_Round_1", "MM_Round_2", "MM_Round_3", "Dec_final"]
    JSON_INPUT = "studies.json"
    df = helpers.process_and_filter_database_export(JSON_INPUT, STUDY_NAME, is_mismatch_study=False)
    battles = helpers.convert_to_battle_csv(df, is_mismatch_study=True)
battles = battles.sort_values(["user", "input_code"])
battles.to_csv(os.path.join(output_dir, "battles.csv"), index=False)
battles_no_ties = battles[~battles["winner"].str.contains("tie")]

# ------------------
# VISUALIZATIONS
# ------------------
visualize_juice(battles, output_dir=output_dir)
print_pair_comparisons(battles)
visualize_battle_statistics(battles, battles_no_ties, output_dir=output_dir)
visualize_preference_ratio(battles, output_dir=output_dir)

# ------------------
# BOOTSTRAPPING
# ------------------
BOOTSTRAP_ROUNDS = 10000
bootstrap_results_split_ties = helpers.bootstrap_mismatch(battles, BOOTSTRAP_ROUNDS)
fig = visualize_bootstrap_scores_vertical(bootstrap_results_split_ties, "", output_dir=output_dir)
