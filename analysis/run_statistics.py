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


def analyze_results(results_dir):
    """
    Analyzes all result CSVs in a directory, calculates statistics,
    and performs significance testing.

    Args:
        results_dir (str): Path to the directory containing result CSV files.
    """
    # Find all result CSV files in the specified directory
    csv_files = glob(os.path.join(results_dir, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in '{results_dir}'.")
        return

    # Group files by problem
    results_by_problem = {}
    for f in csv_files:
        try:
            # We need to read the file to find out the problem name
            df_temp = pd.read_csv(f, nrows=1)
            problem_name = df_temp["problem_name"].iloc[0]

            if problem_name not in results_by_problem:
                results_by_problem[problem_name] = []

            # Store the full dataframe for later processing
            results_by_problem[problem_name].append(pd.read_csv(f))
        except (pd.errors.EmptyDataError, IndexError):
            print(f"Warning: Skipping empty or invalid file: {f}")
            continue

    # Process each problem separately
    for problem, dfs in results_by_problem.items():
        print(f"\n--- Analysis for Problem: {problem} ---")

        all_data = pd.concat(dfs, ignore_index=True)

        # Calculate mean and std dev for each algorithm
        summary = all_data.groupby("algorithm_name")["final_ul_fitness"].agg(["mean", "std"]).reset_index()

        print("\n** Performance Summary **")
        print(summary.to_string(index=False))

        # Statistical comparison against SACE-ES (if present)
        algorithms = summary["algorithm_name"].unique()
        if "SACE_ES" in algorithms and len(algorithms) > 1:
            print("\n** Wilcoxon Rank-Sum Test (vs. SACE_ES) **")
            print("p < 0.05 indicates a statistically significant difference.")

            sace_es_results = all_data[all_data["algorithm_name"] == "SACE_ES"]["final_ul_fitness"]

            for algo in algorithms:
                if algo == "SACE_ES":
                    continue

                competitor_results = all_data[all_data["algorithm_name"] == algo]["final_ul_fitness"]

                # Check if we have enough data to compare
                if len(sace_es_results) != len(competitor_results):
                    print(f"  - Cannot compare with {algo}: unequal number of runs.")
                    continue

                try:
                    stat, p_value = wilcoxon(sace_es_results, competitor_results)
                    print(f"  - SACE_ES vs. {algo}: p-value = {p_value:.4f}")
                except ValueError as e:
                    print(f"  - Could not perform Wilcoxon test for {algo}: {e}")

    print("\n--- Analysis Complete ---")


# =============================================================================
# Test Module
# =============================================================================
if __name__ == "__main__":
    print("--- Verifying run_statistics.py ---")

    # Create a dummy results directory and some CSV files
    dummy_dir = os.path.join("results", "csv")
    os.makedirs(dummy_dir, exist_ok=True)

    print(f"\nCreating dummy result files in '{dummy_dir}'...")

    # Data for SACE-ES (superior performance)
    sace_data = {
        "run_id": range(1, 31),
        "problem_name": ["SMD1_Test"] * 30,
        "algorithm_name": ["SACE_ES"] * 30,
        "final_ul_fitness": np.random.normal(loc=1.5, scale=0.1, size=30),
    }
    pd.DataFrame(sace_data).to_csv(os.path.join(dummy_dir, "sace_results.csv"), index=False)

    # Data for NestedDE (inferior performance)
    de_data = {
        "run_id": range(1, 31),
        "problem_name": ["SMD1_Test"] * 30,
        "algorithm_name": ["NestedDE"] * 30,
        "final_ul_fitness": np.random.normal(loc=2.5, scale=0.3, size=30),
    }
    pd.DataFrame(de_data).to_csv(os.path.join(dummy_dir, "de_results.csv"), index=False)

    print("Dummy files created.")

    # Run the analysis function on the dummy directory
    try:
        analyze_results(dummy_dir)
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
    finally:
        # Clean up the dummy files and directory
        print("\nCleaning up dummy files...")
        for f in glob(os.path.join(dummy_dir, "*.csv")):
            os.remove(f)
        os.rmdir(dummy_dir)
        print("Cleanup complete.")

    print("\n--- Verification Complete ---")
