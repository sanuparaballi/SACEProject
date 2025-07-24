# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:57:47 2025

@author: l-ssarabal
"""

# sace_project/analysis/combine_history.py

import pandas as pd
import os
import argparse
from glob import glob


def combine_history_files(history_dir, output_file):
    """
    Scans a directory for history CSVs, combines them into a single file,
    and adds columns for problem and algorithm names based on the filenames.

    Args:
        history_dir (str): The directory containing the individual history CSV files.
        output_file (str): The path to save the combined CSV file.
    """
    history_files = glob(os.path.join(history_dir, 'history_*.csv'))

    if not history_files:
        print(f"No history files found in '{history_dir}'. Aborting.")
        return

    all_history_data = []
    print(f"Found {len(history_files)} history files to combine...")

    for f in history_files:
        try:
            # Extract problem and algorithm name from filename
            # Format: history_ExperimentName_ProblemName_AlgorithmName_timestamp.csv
            base_name = os.path.basename(f).replace('history_', '').rsplit('_', 1)[0]
            parts = base_name.split('_')

            print(parts)

            if len(parts) < 2:
                print('Is it coming here?')
                print(f"Warning: Skipping malformed filename: {os.path.basename(f)}")
                continue

            # print('TEST!')
            problem_name = parts[0]
            algorithm_name = parts[1]
            # ignore = parts[3]
            # print(problem_name)
            # print(algorithm_name)

            df = pd.read_csv(f)
            # print('TEST@')
            df['problem_name'] = problem_name
            df['algorithm_name'] = algorithm_name
            all_history_data.append(df)

        except Exception as e:
            print(f"Warning: Could not process file {f}. Error: {e}")
            continue

    if not all_history_data:
        print("No valid history data could be loaded.")
        return

    # Combine all dataframes into one
    combined_df = pd.concat(all_history_data, ignore_index=True)

    # Save to the final output file
    try:
        combined_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully combined all history data into: {output_file}")
    except IOError as e:
        print(f"\nError: Could not save combined file. Exception: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Combine all history CSVs into a single file.")
    parser.add_argument(
        '--history_dir',
        type=str,
        default=os.path.join('./../results', 'history'),
        help='Directory containing the CSV history files.'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=os.path.join('./../results', 'combined_history.csv'),
        help='Path for the final combined CSV file.'
    )
    args = parser.parse_args()

    if not os.path.isdir(args.history_dir):
        print(f"Error: History directory not found at '{args.history_dir}'")
    else:
        combine_history_files(args.history_dir, args.output_file)
