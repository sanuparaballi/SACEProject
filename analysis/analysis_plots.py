#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 17:09:01 2025

@author: sanup
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results csv file
try:
    df = pd.read_csv("Full_Benchmark_V1.0_20250729-150246.csv")

    # Data Cleaning: Convert error strings to NaN and numeric columns to numbers
    numeric_cols = ["final_ul_fitness", "total_ul_nfe", "total_ll_nfe"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop runs that resulted in errors for a cleaner analysis
    df.dropna(inplace=True)

    # --- Data Aggregation ---
    # Group by problem and algorithm to calculate mean and std for key metrics
    agg_results = (
        df.groupby(["problem_name", "algorithm_name"])
        .agg(
            mean_fitness=("final_ul_fitness", "mean"),
            std_fitness=("final_ul_fitness", "std"),
            mean_ul_nfe=("total_ul_nfe", "mean"),
            mean_ll_nfe=("total_ll_nfe", "mean"),
        )
        .reset_index()
    )

    # Calculate total NFE for cost analysis
    agg_results["mean_total_nfe"] = agg_results["mean_ul_nfe"] + agg_results["mean_ll_nfe"]

    # --- Create a pivot table for easier viewing ---
    # We will show the mean fitness. Lower is better.
    pivot_fitness = agg_results.pivot(index="problem_name", columns="algorithm_name", values="mean_fitness")

    # Reorder columns for logical comparison
    algo_order = [
        "NestedDE",
        "BiGA_Lazy",
        "BiGA_Aggressive",
        "GBSA",
        "BBOA_KKT",
        "SACE_ES_Independent",
        "SACE_ES_MOGP",
        "SACE_ES_Heteroscedastic",
    ]
    # Ensure all expected columns are present, add missing ones with NaN
    for col in algo_order:
        if col not in pivot_fitness.columns:
            pivot_fitness[col] = np.nan
    pivot_fitness = pivot_fitness[algo_order]

    # Reorder index to group problems logically
    problem_order = [f"SMD{i}" for i in range(1, 13)]
    pivot_fitness = pivot_fitness.reindex(problem_order)

    print("--- Aggregated Mean Fitness (Lower is Better) ---")
    print(pivot_fitness.to_string(float_format="%.4f"))

    # --- Visualization ---
    # Select a few representative problems for plotting
    # plot_problems = ["SMD2", "SMD4", "SMD7", "SMD11"]
    plot_problems = problem_order
    plot_df = agg_results[agg_results["problem_name"].isin(plot_problems)]

    # Plot 1: Mean Final Fitness Comparison
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 6, figsize=(16, 12))
    fig.suptitle(
        "Algorithm Performance Comparison on Representative Problems (Mean Fitness)",
        fontsize=18,
        weight="bold",
    )

    for ax, problem in zip(axes.flatten(), plot_problems):
        subset = plot_df[plot_df["problem_name"] == problem].sort_values("mean_fitness")
        sns.barplot(x="mean_fitness", y="algorithm_name", data=subset, ax=ax, palette="viridis")
        ax.set_title(f"Problem: {problem}", fontsize=14)
        ax.set_xlabel("Mean Final UL Fitness (Lower is Better)")
        ax.set_ylabel("")
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("performance_comparison.png")

    # Plot 2: Computational Cost Comparison
    fig, axes = plt.subplots(2, 6, figsize=(16, 12))
    fig.suptitle("Algorithm Computational Cost Comparison (Mean Total NFE)", fontsize=18, weight="bold")

    for ax, problem in zip(axes.flatten(), plot_problems):
        subset = plot_df[plot_df["problem_name"] == problem].sort_values("mean_total_nfe")
        sns.barplot(x="mean_total_nfe", y="algorithm_name", data=subset, ax=ax, palette="plasma")
        ax.set_title(f"Problem: {problem}", fontsize=14)
        ax.set_xlabel("Mean Total NFE (Lower is Cheaper)")
        ax.set_ylabel("")
        ax.set_xscale("log")  # Use log scale for large differences in NFE
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", padding=3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("cost_comparison.png")


except FileNotFoundError:
    print("Error: The results file 'Full_Benchmark_V1.0_20250729-150246.csv' was not found.")
except Exception as e:
    print(f"An error occurred during analysis: {e}")
