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
    Updated to handle batch runs and detailed error messages.
    """

    def __init__(self, experiment_name):
        """
        Initializes the logger and creates the output directory and file.

        Args:
            experiment_name (str): The base name for the experiment, used for the filename.
        """
        output_dir = os.path.join("results", "csv")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.filepath = os.path.join(output_dir, f"{experiment_name}_{timestamp}.csv")

        # Define the headers for our CSV file. This standardizes the output for all experiments.
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
            with open(self.filepath, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
        except IOError as e:
            print(f"Error: Could not write header to log file {self.filepath}. Exception: {e}")
            raise

    def log_run(self, run_id, results):
        """
        Logs the final results of a single independent run to the CSV file.
        """
        ul_solution = results.get("best_ul_solution", np.array([]))
        ll_solution = results.get("corresponding_ll_solution", np.array([]))

        problem_name = results.get("problem_name", "N/A")
        algorithm_name = results.get("algorithm_name", "N/A")

        row = {
            "run_id": run_id,
            "problem_name": problem_name,
            "algorithm_name": algorithm_name,
            "final_ul_fitness": (
                f"{results.get('final_ul_fitness', 'ERROR'):.6e}"
                if isinstance(results.get("final_ul_fitness"), (int, float))
                else str(results.get("final_ul_fitness"))
            ),
            "total_ul_nfe": results.get("total_ul_nfe"),
            "total_ll_nfe": results.get("total_ll_nfe"),
            "best_ul_solution": (
                np.array2string(ul_solution, separator=",")
                if isinstance(ul_solution, np.ndarray)
                else ul_solution
            ),
            "corresponding_ll_solution": (
                np.array2string(ll_solution, separator=",")
                if isinstance(ll_solution, np.ndarray)
                else ll_solution
            ),
        }

        try:
            with open(self.filepath, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writerow(row)
        except IOError as e:
            print(f"Error: Could not log run {run_id} to file {self.filepath}. Exception: {e}")
