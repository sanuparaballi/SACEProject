#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:46:29 2025

@author: sanup
"""

# sace_project/src/utils/data_logger.py

import csv
import os
import time
import numpy as np


class DataLogger:
    """
    Handles logging of experimental results to a CSV file in a structured format.
    """

    def __init__(self, experiment_name, config):
        """
        Initializes the logger and creates the output directory and file.

        The filename will include a timestamp to ensure that results from
        different experiments do not overwrite each other.

        Args:
            experiment_name (str): The name of the experiment, used for the filename.
            config (dict): The full configuration dictionary for the experiment.
        """
        # Create the 'results/csv' directory if it doesn't already exist.
        output_dir = os.path.join("results", "csv")
        os.makedirs(output_dir, exist_ok=True)

        # Generate a unique filename using a timestamp.
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.filepath = os.path.join(output_dir, f"{experiment_name}_{timestamp}.csv")

        self.config = config

        # Define the headers for our CSV file. This standardizes the output.
        self.fieldnames = [
            "run_id",
            "problem_name",
            "algorithm_name",
            "final_ul_fitness",
            "total_ul_nfe",
            "total_ll_nfe",
            "best_ul_solution",
            "corresponding_ll_solution",
        ]

        # Write the header row to the newly created CSV file.
        try:
            with open(self.filepath, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
        except IOError as e:
            print(f"Error: Could not write header to log file {self.filepath}. Exception: {e}")
            raise

    def log_run(self, run_id, results):
        """
        Logs the final results of a single independent run to the CSV file.

        Args:
            run_id (int): The identifier of the independent run (e.g., 1 to 30).
            results (dict): A dictionary containing the final results of the run,
                            matching the structure from BaseOptimizer.solve().
        """
        # Prepare the row dictionary with data from the experiment run.
        # We convert numpy arrays to strings for clean CSV storage.
        ul_solution = results.get("best_ul_solution", np.array([]))
        ll_solution = results.get("corresponding_ll_solution", np.array([]))

        row = {
            "run_id": run_id,
            "problem_name": self.config["problem"]["name"],
            "algorithm_name": self.config["algorithm"]["name"],
            "final_ul_fitness": f"{results.get('final_ul_fitness'):.6e}",  # Scientific notation for precision
            "total_ul_nfe": results.get("total_ul_nfe"),
            "total_ll_nfe": results.get("total_ll_nfe"),
            "best_ul_solution": np.array2string(ul_solution, separator=","),
            "corresponding_ll_solution": np.array2string(ll_solution, separator=","),
        }

        # Append the new row to the existing CSV file.
        try:
            with open(self.filepath, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writerow(row)
        except IOError as e:
            print(f"Error: Could not log run {run_id} to file {self.filepath}. Exception: {e}")
