#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 23:05:52 2025

@author: sanup
"""


# sace_project/src/utils/history_logger.py

import csv
import os
import time


class HistoryLogger:
    """
    Logs the generation-by-generation history of an algorithm's run.
    This is used to generate convergence plots.
    """

    def __init__(self, problem_name, algorithm_name):
        output_dir = os.path.join("results", "history")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"history_{problem_name}_{algorithm_name}_{timestamp}.csv"
        self.filepath = os.path.join(output_dir, filename)

        self.fieldnames = [
            "run_id",
            "generation",
            "best_fitness",
            "avg_fitness",
            "cumulative_ul_nfe",
            "cumulative_ll_nfe",
        ]

        try:
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        except IOError as e:
            print(f"Error creating history log file: {e}")

    def log_generation_batch(self, run_id, history_data):
        """
        Logs the entire history of a single run at once.
        """
        try:
            with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                for gen_data in history_data:
                    gen_data["run_id"] = run_id
                    writer.writerow(gen_data)
        except IOError as e:
            print(f"Error writing to history log file: {e}")
