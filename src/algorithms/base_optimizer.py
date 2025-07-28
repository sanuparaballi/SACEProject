#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:44:24 2025

@author: sanup
"""


# sace_project/src/algorithms/base_optimizer.py

from abc import ABC, abstractmethod

# Import the new history logger
from src.utils.history_logger import HistoryLogger


class BaseOptimizer(ABC):
    """
    Abstract Base Class for all bilevel optimization algorithms.

    This class defines the common interface that all implemented algorithms must follow.
    This ensures that the main experiment runner can interact with any algorithm
    in a standardized way.

    UPDATED: Now includes a history logger for convergence analysis.
    """

    def __init__(self, problem, config):
        """
        Initializes the optimizer.

        Args:
            problem: An object representing the bilevel optimization problem.
                     It is expected to have properties like ul_dim, ll_dim, ul_bounds,
                     ll_bounds, and an evaluate() method.
            config (dict): A dictionary containing algorithm-specific parameters
                           extracted from the main JSON config file.

        UPDATED: Added experiment_name for the logger.
        """
        self.problem = problem
        self.config = config
        self.history = []  # A list to store convergence data for each generation

        # Common NFE counters
        self.ul_nfe = 0
        self.ll_nfe = 0

        # Initialize the history logger for this specific run
        self.history_logger = HistoryLogger(
            problem_name=problem.__class__.__name__,  # Use class name for consistency
            algorithm_name=self.__class__.__name__,
        )

    @abstractmethod
    def solve(self):
        """
        The main method to run the optimization process.

        This method should implement the core logic of the specific bilevel algorithm.
        It must be overridden by any concrete subclass.

        Returns:
            A dictionary containing the final results of the run, for example:
            {
                'final_ul_fitness': float,
                'total_ul_nfe': int,
                'total_ll_nfe': int,
                'best_ul_solution': np.array,
                'corresponding_ll_solution': np.array
            }
        """
        pass

    def log_generation(self, gen_num, best_fitness, avg_fitness):
        """
        Logs the state of a generation for later analysis of convergence.

        This can be called within the solve() method of a subclass at the end
        of each generation.

        Args:
            gen_num (int): The current generation number.
            best_fitness (float): The best fitness value in the current population.
            avg_fitness (float): The average fitness of the current population.

        Stores the state of a generation in memory.
        """
        log_entry = {
            "generation": gen_num,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "cumulative_ul_nfe": self.ul_nfe,
            "cumulative_ll_nfe": self.ll_nfe,
        }
        self.history.append(log_entry)

    def _commit_history(self, run_id):
        """
        Writes the entire accumulated history for a run to the CSV file.
        """
        self.history_logger.log_generation_batch(run_id, self.history)
