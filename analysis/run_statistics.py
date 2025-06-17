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


def calculate_l2_error(row, best_known_sol):
    """Calculates L2 error against a provided best-known solution."""
    if best_known_sol is None:
        return np.nan

    try:
        ul_sol_str = row["best_ul_solution"].strip("[]")
        ll_sol_str = row["corresponding_ll_solution"].strip("[]")
        if not ul_sol_str or not ll_sol_str:
            return np.nan

        ul_sol = np.fromstring(ul_sol_str, sep=",")
        ll_sol = np.fromstring(ll_sol_str, sep=",")

        # A simple heuristic to handle potential parsing issues with nested brackets
        if len(ul_sol) > best_known_sol.shape[0] or len(ll_sol) > best_known_sol.shape[0]:
            ul_sol = np.fromstring(ul_sol_str.split("]")[0].strip("[ "), sep=",")
            ll_sol = np.fromstring(ll_sol_str.split("]")[0].strip("[ "), sep=",")

        solution_vec = np.concatenate([ul_sol, ll_sol])

        if len(solution_vec) != len(best_known_sol):
            return np.nan  # Mismatched dimensions

        error = np.linalg.norm(solution_vec - best_known_sol)
        return error
    except (ValueError, TypeError, IndexError):
        return np.nan


def analyze_results(results_dir, reference_algo="SACE_ES", l2_baseline_algo="NestedDE"):
    """
    Analyzes all result CSVs in a directory, calculates statistics,
    and performs significance testing against a reference algorithm.
    """
    csv_files = glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in '{results_dir}'.")
        return

    all_data = pd.concat([pd.read_csv(f) for f in csv_files if os.path.getsize(f) > 0], ignore_index=True)

    numeric_cols = ["final_ul_fitness", "total_ul_nfe", "total_ll_nfe"]
    for col in numeric_cols:
        all_data[col] = pd.to_numeric(all_data[col], errors="coerce")

    all_data["total_nfe"] = all_data["total_ul_nfe"] + all_data["total_ll_nfe"]

    for problem, problem_df in all_data.groupby("problem_name"):
        print(f"\n--- Analysis for Problem: {problem} ---")

        best_known_solution = None
        baseline_df = problem_df[problem_df["algorithm_name"] == l2_baseline_algo].dropna(
            subset=["final_ul_fitness"]
        )
        if not baseline_df.empty:
            best_run = baseline_df.loc[baseline_df["final_ul_fitness"].idxmin()]
            try:
                best_ul = np.fromstring(best_run["best_ul_solution"].strip("[]"), sep=",")
                best_ll = np.fromstring(best_run["corresponding_ll_solution"].strip("[]"), sep=",")
                best_known_solution = np.concatenate([best_ul, best_ll])
                print(f"  (Using best solution from '{l2_baseline_algo}' as ground truth for L2 Error)")
            except (ValueError, TypeError):
                pass

        problem_df_copy = problem_df.copy()
        problem_df_copy["l2_error"] = problem_df_copy.apply(
            lambda row: calculate_l2_error(row, best_known_solution), axis=1
        )

        summary = (
            problem_df_copy.groupby("algorithm_name")
            .agg(
                mean_fitness=("final_ul_fitness", "mean"),
                std_fitness=("final_ul_fitness", "std"),
                mean_l2_error=("l2_error", "mean"),
                mean_total_nfe=("total_nfe", "mean"),
                count=("final_ul_fitness", "count"),
            )
            .reset_index()
        )

        print("\n** Performance Summary **")
        print(summary.to_string(index=False))

        algorithms = summary["algorithm_name"].unique()
        if reference_algo in algorithms and len(algorithms) > 1:
            print(f"\n** Wilcoxon Rank-Sum Test (vs. {reference_algo}) **")
            print("p < 0.05 indicates a statistically significant difference.")

            ref_results_df = problem_df_copy[problem_df_copy["algorithm_name"] == reference_algo]

            for algo in algorithms:
                if algo == reference_algo:
                    continue

                comp_results_df = problem_df_copy[problem_df_copy["algorithm_name"] == algo]

                # Align data based on run_id for a true paired test
                aligned_df = pd.merge(
                    ref_results_df[["run_id", "final_ul_fitness"]],
                    comp_results_df[["run_id", "final_ul_fitness"]],
                    on="run_id",
                    suffixes=("_ref", "_comp"),
                ).dropna()

                if len(aligned_df) < 2:
                    print(f"  - Cannot compare with {algo}: not enough paired successful runs.")
                    continue

                # UPDATED: Convert pandas Series to NumPy arrays before passing to wilcoxon
                ref_values = aligned_df["final_ul_fitness_ref"].to_numpy()
                comp_values = aligned_df["final_ul_fitness_comp"].to_numpy()

                try:
                    if np.allclose(ref_values, comp_values):
                        print(f"  - {reference_algo} vs. {algo}: Results are statistically identical.")
                    else:
                        stat, p_value = wilcoxon(ref_values, comp_values)
                        print(f"  - {reference_algo} vs. {algo}: p-value = {p_value:.4f}")
                except ValueError as e:
                    print(f"  - Could not perform Wilcoxon test for {algo}: {e}")

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistical analysis on experiment results.")
    parser.add_argument("--results_dir", type=str, default=os.path.join("./../results", "csv"))
    args = parser.parse_args()
    if os.path.isdir(args.results_dir):
        analyze_results(args.results_dir)
    else:
        print(f"Error: Results directory not found at '{args.results_dir}'")
