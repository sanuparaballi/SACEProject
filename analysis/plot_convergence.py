#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:53:31 2025

@author: sanup
"""


# sace_project/analysis/plot_convergence.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from glob import glob


def plot_all_convergence(history_dir, output_dir):
    """
    Finds all history files, groups them by problem, and creates
    a single convergence plot for each problem comparing all algorithms.
    """
    history_files = glob(os.path.join(history_dir, "history_*.csv"))

    if not history_files:
        print(f"No history files found in '{history_dir}'.")
        print(
            "Note: The main script and algorithm classes must be updated to use the HistoryLogger for this script to work."
        )
        return

    # Group files by problem name, which is part of the filename
    files_by_problem = {}
    for f in history_files:
        try:
            # Filename format: 'history_ExperimentName_ProblemName_AlgorithmName_timestamp.csv'
            base_name = os.path.basename(f).replace("history_", "").rsplit("_", 1)[0]  # remove timestamp
            parts = base_name.split("_")
            problem_name = parts[1]
            algo_name = parts[2]

            if problem_name not in files_by_problem:
                files_by_problem[problem_name] = []
            files_by_problem[problem_name].append({"path": f, "algo": algo_name})
        except IndexError:
            print(f"Warning: Could not parse filename '{os.path.basename(f)}'. Skipping.")
            continue

    # Create one plot for each problem
    for problem_name, file_infos in files_by_problem.items():
        plt.figure(figsize=(12, 8))

        for info in file_infos:
            df = pd.read_csv(info["path"])
            algo_name = info["algo"]

            # Pivot to get median performance over all runs
            # Use aggfunc='first' in case of duplicate generations, and ffill() to handle missing data
            pivot_df = df.pivot_table(
                index="generation", columns="run_id", values="best_fitness", aggfunc="first"
            ).ffill()
            median_fitness = pivot_df.median(axis=1)

            # Use NFE for the x-axis. We'll average the NFE across runs for a representative x-axis.
            nfe_pivot_ul = df.pivot_table(
                index="generation", columns="run_id", values="cumulative_ul_nfe", aggfunc="first"
            ).ffill()
            nfe_pivot_ll = df.pivot_table(
                index="generation", columns="run_id", values="cumulative_ll_nfe", aggfunc="first"
            ).ffill()
            mean_nfe = nfe_pivot_ul.mean(axis=1) + nfe_pivot_ll.mean(axis=1)

            # Ensure x_axis and median_fitness have the same length before plotting
            min_len = min(len(mean_nfe), len(median_fitness))
            plt.plot(mean_nfe.iloc[:min_len], median_fitness.iloc[:min_len], label=f"{algo_name}")

        plt.title(f"Median Convergence on {problem_name}")
        plt.xlabel("Total Function Evaluations (NFE)")
        plt.ylabel("Best Upper-Level Objective Value")
        plt.yscale("log")  # Log scale is often essential for visualizing optimization progress
        plt.legend()
        plt.grid(True, which="both", ls="--")

        plot_filename = f"convergence_{problem_name}_comparison.png"
        output_path = os.path.join(output_dir, plot_filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Comparison convergence plot saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate convergence plots from history files.")
    parser.add_argument(
        "--history_dir",
        type=str,
        default=os.path.join("./../results", "history"),
        help="Directory containing the CSV history files to analyze.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("results", "plots"),
        help="Directory to save the generated plot images.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plot_all_convergence(args.history_dir, args.output_dir)
