# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 16:15:56 2025

@author: l-ssarabal
"""

# sace_project/analysis/plot_convergence.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse


def plot_all_convergence(combined_history_path, output_dir):
    """
    Reads a single combined history file, groups by problem, and creates
    a convergence plot for each problem comparing all algorithms.

    Args:
        combined_history_path (str): Path to the combined history CSV file.
        output_dir (str): Directory to save the plot images.
    """
    try:
        all_history_df = pd.read_csv(combined_history_path)
    except FileNotFoundError:
        print(f"Error: Combined history file not found at '{combined_history_path}'.")
        print("Please run the combine_history.py script first.")
        return

    # Create one plot for each problem
    for problem_name, problem_df in all_history_df.groupby('problem_name'):
        plt.figure(figsize=(12, 8))

        # Plot each algorithm's convergence on the same axes
        for algo_name, algo_df in problem_df.groupby('algorithm_name'):

            # Pivot to get median performance over all runs
            # ffill() handles runs that may have terminated early
            pivot_df = algo_df.pivot_table(index='generation', columns='run_id',
                                           values='best_fitness', aggfunc='first').ffill()
            median_fitness = pivot_df.median(axis=1)

            # Use NFE for the x-axis. We'll average the NFE across runs for a representative x-axis.
            nfe_pivot_ul = algo_df.pivot_table(index='generation', columns='run_id',
                                               values='cumulative_ul_nfe', aggfunc='first').ffill()
            nfe_pivot_ll = algo_df.pivot_table(index='generation', columns='run_id',
                                               values='cumulative_ll_nfe', aggfunc='first').ffill()

            # Ensure all NFE columns are numeric before summing
            nfe_pivot_ul = nfe_pivot_ul.apply(pd.to_numeric, errors='coerce')
            nfe_pivot_ll = nfe_pivot_ll.apply(pd.to_numeric, errors='coerce')

            mean_nfe = nfe_pivot_ul.mean(axis=1) + nfe_pivot_ll.mean(axis=1)

            # Ensure x_axis and median_fitness have the same length before plotting
            min_len = min(len(mean_nfe), len(median_fitness))
            plt.plot(mean_nfe.iloc[:min_len], median_fitness.iloc[:min_len], label=f'{algo_name}')

        plt.title(f'Median Convergence on {problem_name}', fontsize=16)
        plt.xlabel('Total Function Evaluations (NFE)', fontsize=12)
        plt.ylabel('Best Upper-Level Objective Value (Median)', fontsize=12)
        plt.yscale('log')  # Log scale is often essential for visualizing optimization progress
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()

        plot_filename = f"convergence_{problem_name}_comparison.png"
        output_path = os.path.join(output_dir, plot_filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Comparison convergence plot saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate convergence plots from a combined history file.")
    parser.add_argument(
        '--history_file',
        type=str,
        default=os.path.join('./../results', 'combined_history.csv'),
        help='Path to the combined CSV history file.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join('./../results', 'plots'),
        help='Directory to save the generated plot images.'
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plot_all_convergence(args.history_file, args.output_dir)
