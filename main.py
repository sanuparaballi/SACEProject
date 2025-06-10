#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:50:44 2025

@author: sanup
"""


# sace_project/main.py

import json
import argparse
import time
import numpy as np
from tqdm import tqdm
import os
import traceback  # Import the traceback module for detailed error reporting

from src.problems import get_problem
from src.algorithms import get_algorithm
from src.utils.data_logger import DataLogger


def main(config_path):
    """
    Main function to run a batch of bilevel optimization experiments.
    """
    # 1. Load Configuration
    print(f"--- Loading batch configuration from: {config_path} ---")
    try:
        with open(config_path, "r", encoding="utf-8-sig") as f:
            config = json.load(f)
    except Exception as e:
        print(f"FATAL ERROR: Could not load or parse config file. Details: {e}")
        return

    exp_name = config.get("experiment_name", "BilevelBatchExperiment")
    settings = config.get("settings", {})
    problems_to_run = config.get("problems", [])
    algorithms_to_run = config.get("algorithms", [])

    if not problems_to_run or not algorithms_to_run:
        print("FATAL ERROR: The config file must contain 'problems' and 'algorithms' lists.")
        return

    # 2. Initialize a single logger for the entire batch
    logger = DataLogger(exp_name)
    print(f"Results for this entire batch will be saved to: {logger.filepath}")

    # 3. Main Experiment Loop for all combinations
    total_combinations = len(problems_to_run) * len(algorithms_to_run)
    print(f"\nStarting batch execution for {total_combinations} combinations.")

    combination_pbar = tqdm(total=total_combinations, desc="Overall Progress")

    for problem_config in problems_to_run:
        for algorithm_config in algorithms_to_run:

            current_combination_name = f"{algorithm_config['name']} on {problem_config['name']}"
            combination_pbar.set_description(f"Running: {current_combination_name}")

            # --- Per-combination setup ---
            num_runs = settings.get("independent_runs", 30)
            base_seed = settings.get("seed", int(time.time()))

            for i in range(num_runs):
                run_id = i + 1

                # UPDATED: Use the modulo operator to keep the seed within the valid 32-bit range.
                # The hash() of a string can be a large negative number, so we take abs().
                unbounded_seed = base_seed + run_id + abs(hash(current_combination_name))
                current_seed = unbounded_seed % (2**32)
                np.random.seed(current_seed)

                try:
                    # Instantiate Problem and Algorithm for the current combination
                    problem = get_problem(problem_config["name"], problem_config.get("params", {}))
                    algorithm = get_algorithm(
                        algorithm_config["name"], problem, algorithm_config.get("params", {})
                    )

                    # Run the optimization
                    final_results = algorithm.solve()

                    # Add problem and algorithm names for the logger
                    final_results["problem_name"] = problem_config["name"]
                    final_results["algorithm_name"] = algorithm_config["name"]

                    # Log the results
                    logger.log_run(run_id, final_results)

                except Exception as e:
                    # UPDATED: Use traceback to print the full, detailed error stack
                    print(f"\n--- ERROR during run {run_id} of {current_combination_name} ---")
                    print("An unhandled exception occurred. Full traceback below:")
                    traceback.print_exc()

                    # Log the error and continue
                    error_results = {
                        "final_ul_fitness": "ERROR",
                        "total_ul_nfe": "ERROR",
                        "total_ll_nfe": "ERROR",
                        "best_ul_solution": str(e),
                        "corresponding_ll_solution": "",
                        "problem_name": problem_config["name"],  # Ensure names are logged on error
                        "algorithm_name": algorithm_config["name"],
                    }
                    logger.log_run(run_id, error_results)

            combination_pbar.update(1)

    combination_pbar.close()
    print(f"\n--- Batch Experiment Finished ---")
    print(f"All results have been saved to: {logger.filepath}")


if __name__ == "__main__":
    # --- For running in an IDE like Spyder ---
    config_file_path = "configs/exp1_smd_suite_batch.json"
    main(config_file_path)
