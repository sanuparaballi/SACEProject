# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 02:47:51 2025

@author: l-ssarabal
"""

# sace_project/analysis/generate_summary_plots.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from glob import glob


def generate_plots(results_dir, output_dir, reference_algo='SACE_ES', l2_baseline_algo='NestedDE'):
    """
    Analyzes result CSVs to generate summary box plots and scatter plots.
    """
    csv_files = glob(os.path.join(results_dir, '*.csv'))
    if not csv_files:
        print(f"No CSV files found in '{results_dir}'.")
        return

    all_data = pd.concat([pd.read_csv(f) for f in csv_files if os.path.getsize(f) > 0], ignore_index=True)
    all_data['final_ul_fitness'] = pd.to_numeric(all_data['final_ul_fitness'], errors='coerce')

    # --- Plot 1: Box plot for a representative hard problem (e.g., SMD2) ---
    problem_to_plot = 'SP2'
    df_smd2 = all_data[all_data['problem_name'] == problem_to_plot].dropna(subset=['final_ul_fitness'])
    if not df_smd2.empty:
        plt.figure(figsize=(10, 7))
        sns.boxplot(x='algorithm_name', y='final_ul_fitness', data=df_smd2, showfliers=False)
        plt.title(f'Final Fitness Distribution on {problem_to_plot}')
        plt.ylabel('Final Upper-Level Fitness')
        plt.xlabel('Algorithm')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'boxplot_{problem_to_plot}.png'), dpi=300)
        plt.close()
        print(f"Generated box plot for {problem_to_plot}")

    # --- Plot 2: L2 Error vs. Fitness scatter plot for SMD6 ---
    problem_to_plot = 'SP2'
    df_smd6 = all_data[all_data['problem_name'] == problem_to_plot].dropna(subset=['final_ul_fitness'])

    # Calculate L2 error for this plot
    baseline_df = df_smd6[df_smd6['algorithm_name'] == l2_baseline_algo]
    if not baseline_df.empty:
        best_run = baseline_df.loc[baseline_df['final_ul_fitness'].idxmin()]
        try:
            best_ul = np.fromstring(best_run['best_ul_solution'].strip('[]'), sep=',')
            best_ll = np.fromstring(best_run['corresponding_ll_solution'].strip('[]'), sep=',')
            best_known_solution = np.concatenate([best_ul, best_ll])

            # Creating a safe copy for calculations
            df_smd6_copy = df_smd6.copy()
            df_smd6_copy['l2_error'] = df_smd6_copy.apply(
                lambda row: np.linalg.norm(np.concatenate([np.fromstring(row['best_ul_solution'].strip('[]'), sep=','), np.fromstring(
                    row['corresponding_ll_solution'].strip('[]'), sep=',')]) - best_known_solution),
                axis=1
            )

            plt.figure(figsize=(10, 7))
            sns.scatterplot(x='l2_error', y='final_ul_fitness', hue='algorithm_name',
                            data=df_smd6_copy, s=100, alpha=0.7)
            plt.title(f'L2 Error vs. Final Fitness on {problem_to_plot}')
            plt.xlabel(f'L2 Distance from {l2_baseline_algo} Solution')
            plt.ylabel('Final Upper-Level Fitness')
            plt.grid(True, which="both", ls="--")
            plt.legend(title='Algorithm')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'scatter_{problem_to_plot}.png'), dpi=300)
            plt.close()
            print(f"Generated scatter plot for {problem_to_plot}")
        except Exception as e:
            print(f"Could not generate scatter plot for {problem_to_plot}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate summary plots from experiment results.")
    parser.add_argument('--results_dir', type=str, default=os.path.join('results', 'csv'))
    parser.add_argument('--output_dir', type=str, default=os.path.join('results', 'plots'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generate_plots(args.results_dir, args.output_dir)
