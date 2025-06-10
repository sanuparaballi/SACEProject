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


def plot_convergence(csv_filepath, output_dir):
    """
    Generates a convergence plot from an experiment's history log file.

    Args:
        csv_filepath (str): Path to the history CSV file.
        output_dir (str): Directory to save the plot image.
    """
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: History file not found at {csv_filepath}")
        return

    # Extract experiment details from the first row
    problem_name = df["problem_name"].iloc[0]
    algorithm_name = df["algorithm_name"].iloc[0]

    plt.figure(figsize=(12, 8))

    # We need to pivot the data to get the median performance over all runs
    # Assuming 'generation' is the index, 'run_id' are the columns, 'best_fitness' are values
    pivot_df = df.pivot(index="generation", columns="run_id", values="best_fitness")

    median_fitness = pivot_df.median(axis=1)
    q25 = pivot_df.quantile(0.25, axis=1)
    q75 = pivot_df.quantile(0.75, axis=1)

    # Use NFE for the x-axis, assuming cumulative NFE is logged
    if "cumulative_ul_nfe" in df.columns:
        # For simplicity, we can use the NFE from the first run as representative
        nfe_df = df[df["run_id"] == df["run_id"].min()]
        x_axis = nfe_df["cumulative_ul_nfe"] + nfe_df["cumulative_ll_nfe"]
    else:
        # Fallback to generation number if NFE is not available
        x_axis = median_fitness.index

    # Plotting the median line
    plt.plot(x_axis, median_fitness, label=f"{algorithm_name} (Median)")

    # Shaded area for the interquartile range (IQR)
    plt.fill_between(x_axis, q25, q75, alpha=0.2, label="IQR over 30 runs")

    plt.title(f"Convergence Plot: {algorithm_name} on {problem_name}")
    plt.xlabel("Number of Function Evaluations (NFE)")
    plt.ylabel("Best Upper-Level Objective Value")
    plt.yscale("log")  # Log scale is often better for visualizing optimization progress
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Save the plot
    plot_filename = f"convergence_{problem_name}_{algorithm_name}.png"
    output_path = os.path.join(output_dir, plot_filename)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Convergence plot saved to: {output_path}")


# Note: This script assumes a "history" logger is added to the framework
# that logs data for each generation of each run. The current DataLogger only
# logs the final result. This script is built for the final version of the logger.

# =============================================================================
# Test Module
# =============================================================================
if __name__ == "__main__":
    print("--- Verifying plot_convergence.py ---")

    # Create a dummy history CSV for testing purposes
    output_dir = os.path.join("results", "plots")
    os.makedirs(output_dir, exist_ok=True)
    dummy_csv_path = "dummy_history.csv"

    print("\nCreating a dummy history CSV file for testing...")
    header = [
        "run_id",
        "generation",
        "best_fitness",
        "avg_fitness",
        "cumulative_ul_nfe",
        "cumulative_ll_nfe",
        "problem_name",
        "algorithm_name",
    ]
    with open(dummy_csv_path, "w", newline="") as f:
        writer = pd.DataFrame(columns=header)
        writer.to_csv(f, index=False)

        data = []
        num_runs = 30
        num_gens = 100
        for run in range(1, num_runs + 1):
            # Simulate a decreasing fitness value
            fitness_trend = 100 * np.exp(-0.05 * np.arange(num_gens)) + np.random.randn(num_gens) * 2
            for gen in range(num_gens):
                row = {
                    "run_id": run,
                    "generation": gen,
                    "best_fitness": fitness_trend[gen],
                    "avg_fitness": fitness_trend[gen] + 5,
                    "cumulative_ul_nfe": gen * 50,
                    "cumulative_ll_nfe": gen * 50 * 30,
                    "problem_name": "SMD1_Test",
                    "algorithm_name": "SACE_ES_Test",
                }
                data.append(row)

        pd.DataFrame(data).to_csv(f, header=False, index=False)

    print("Dummy file created.")

    # Run the plotting function
    print("\nRunning the plot_convergence function...")
    try:
        plot_convergence(dummy_csv_path, output_dir)
        # Check if the file was created
        expected_file = os.path.join(output_dir, "convergence_SMD1_Test_SACE_ES_Test.png")
        assert os.path.exists(expected_file), "Plot file was not created!"
        print("Assertion PASSED: Plot file was created successfully.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")
    finally:
        # Clean up the dummy file
        if os.path.exists(dummy_csv_path):
            os.remove(dummy_csv_path)
            print("Dummy CSV file removed.")

    print("\n--- Verification Complete ---")
