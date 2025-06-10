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
import os  # Import the os module for file system checks

# Import the actual factory functions, replacing the placeholders.
from src.problems import get_problem
from src.algorithms import get_algorithm
from src.utils.data_logger import DataLogger


def main(config_path):
    """
    Main function to configure and run a bilevel optimization experiment.

    Args:
        config_path (str): The file path to the JSON experiment configuration.
    """
    # --- Diagnostic Pre-flight Check ---
    print(f"--- Attempting to load configuration from: {config_path} ---")
    if not os.path.exists(config_path):
        print(f"FATAL ERROR: The file '{config_path}' does not exist.")
        print(
            f"Please ensure you are running this script from the project's root directory ('sace_project/')."
        )
        return

    if os.path.getsize(config_path) == 0:
        print(f"FATAL ERROR: The configuration file '{config_path}' is empty.")
        print("Please copy the content into the file and save it again.")
        return
    # --- End of Diagnostic Check ---

    # 1. Load Configuration from JSON file
    try:
        # FINAL UPDATE: Use 'utf-8-sig' encoding. This is specifically designed to
        # handle the invisible BOM character that can cause "char 0" errors.
        with open(config_path, "r", encoding="utf-8-sig") as f:
            config = json.load(f)
    except FileNotFoundError:
        # This case is now handled by the pre-flight check, but kept for safety.
        print(f"Error: Configuration file not found at '{config_path}'")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{config_path}'. Invalid JSON syntax. Details: {e}")
        return

    exp_name = config.get("experiment_name", "BilevelExperiment")
    problem_config = config.get("problem", {})
    algorithm_config = config.get("algorithm", {})
    settings = config.get("settings", {})

    print(f"--- Starting Experiment: {exp_name} ---")

    # 2. Initialize the Data Logger
    logger = DataLogger(exp_name, config)

    # 3. Main Experiment Loop for Independent Runs
    num_runs = settings.get("independent_runs", 30)
    base_seed = settings.get("seed", int(time.time()))

    for i in tqdm(range(num_runs), desc="Total Experiment Progress"):
        run_id = i + 1
        current_seed = base_seed + run_id
        np.random.seed(current_seed)

        try:
            problem = get_problem(problem_config["name"], problem_config.get("params", {}))
            algorithm = get_algorithm(algorithm_config["name"], problem, algorithm_config.get("params", {}))
            final_results = algorithm.solve()
            logger.log_run(run_id, final_results)

        except Exception as e:
            print(f"\n--- ERROR during run {run_id} ---")
            print(f"An exception occurred: {e}")
            print("Skipping this run and continuing with the next one.")
            error_results = {
                "final_ul_fitness": "ERROR",
                "total_ul_nfe": "ERROR",
                "total_ll_nfe": "ERROR",
                "best_ul_solution": str(e),
                "corresponding_ll_solution": "",
            }
            logger.log_run(run_id, error_results)

    print(f"\n--- Experiment Finished ---")
    print(f"All results have been saved to: {logger.filepath}")


if __name__ == "__main__":
    config_file_path = "configs/custom_suite_config.json"
    main(config_file_path)
