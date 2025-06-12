#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:54:43 2025

@author: sanup
"""


# sace_project/analysis/run_statistics.py

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import os
import argparse
from glob import glob


def analyze_results(results_dir, reference_algo="SACE_ES"):
    """
    Analyzes all result CSVs in a directory, calculates statistics,
    and performs significance testing against a reference algorithm.

    Args:
        results_dir (str): Path to the directory containing result CSV files.
        reference_algo (str): The name of the algorithm to use as a baseline for statistical tests.
    """
    # Find all result CSV files in the specified directory
    csv_files = glob(os.path.join(results_dir, "*.csv"))

    csv_files = "./../results/csv/Experiment1_SMD_Suite_Comparison_v2_20250611-182831.csv"

    if not csv_files:
        print(f"No CSV files found in '{results_dir}'.")
        return

    all_data = pd.read_csv(csv_files)
    # Combine all data from all found CSVs into a single DataFrame
    # all_data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Handle potential 'ERROR' strings in fitness column by converting to NaN
    all_data["final_ul_fitness"] = pd.to_numeric(all_data["final_ul_fitness"], errors="coerce")

    # Group by problem to analyze each one separately
    for problem, problem_df in all_data.groupby("problem_name"):
        print(f"\n--- Analysis for Problem: {problem} ---")

        # Calculate mean and std dev for each algorithm on this problem
        summary = (
            problem_df.groupby("algorithm_name")["final_ul_fitness"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        print("\n** Performance Summary **")
        print(summary.to_string(index=False))

        # Statistical comparison against the reference algorithm
        algorithms = summary["algorithm_name"].unique()
        if reference_algo in algorithms and len(algorithms) > 1:
            print(f"\n** Wilcoxon Rank-Sum Test (vs. {reference_algo}) **")
            print("p < 0.05 indicates a statistically significant difference.")

            ref_results = problem_df[problem_df["algorithm_name"] == reference_algo][
                "final_ul_fitness"
            ].dropna()

            for algo in algorithms:
                if algo == reference_algo:
                    continue

                competitor_results = problem_df[problem_df["algorithm_name"] == algo][
                    "final_ul_fitness"
                ].dropna()

                if len(ref_results) != len(competitor_results):
                    print(f"  - Cannot compare with {algo}: unequal number of successful runs.")
                    continue
                if len(ref_results) < 2:
                    print(f"  - Cannot compare with {algo}: not enough data points.")
                    continue

                try:
                    stat, p_value = wilcoxon(ref_results, competitor_results)
                    print(f"  - {reference_algo} vs. {algo}: p-value = {p_value:.4f}")
                except ValueError as e:
                    print(f"  - Could not perform Wilcoxon test for {algo}: {e}")

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistical analysis on experiment results.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join("results", "csv"),
        help="Directory containing the CSV result files to analyze.",
    )
    parser.add_argument(
        "--ref_algo",
        type=str,
        default="SACE_ES",
        help="The reference algorithm for statistical comparisons.",
    )
    args = parser.parse_args()

    analyze_results(args.results_dir, args.ref_algo)
