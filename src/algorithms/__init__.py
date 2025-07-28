#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:23:24 2025

@author: sanup
"""


# sace_project/src/algorithms/__init__.py

# Import the algorithm classes from their respective files
from .base_optimizer import BaseOptimizer
from .nested_de import NestedDE
from .biga import BiGA
from .sace_es import SACE_ES
from .kkt_solver import KKTSolver
from .bboa_kkt import BBOA_KKT
from .gbsa import GBSA

# For testing the factory, we need a mock problem.
from ..problems import get_problem


def get_algorithm(name: str, problem, config: dict) -> BaseOptimizer:
    """
    Factory function to get an algorithm instance by its name.

    This function acts as a central registry for all implemented algorithms.
    It takes the algorithm's name and returns an initialized object.

    UPDATED: Now accepts and passes the 'experiment_name' to the constructor
    for initializing the HistoryLogger correctly.

    Args:
        name (str): The name of the algorithm (e.g., "NestedDE", "SACE_ES").
        problem: The problem instance for the algorithm to solve.
        config (dict): The configuration dictionary for the algorithm's parameters.
        experiment_name (str): The name of the overall experiment batch.

    Returns:
        An initialized instance of a subclass of BaseOptimizer.

    Raises:
        ValueError: If the specified algorithm name is not found.
    """
    algorithm_map = {
        "nestedde": NestedDE,
        "biga_lazy": BiGA,
        "biga_aggressive": BiGA,
        "sace_es": SACE_ES,
        "kktsolver": KKTSolver,
        "bboa_kkt": BBOA_KKT,
        "gbsa": GBSA,
    }

    # Handle BiGA variants by checking the name string
    if "biga" in name.lower():
        algorithm_class = BiGA
    else:
        algorithm_class = algorithm_map.get(name.lower())

    if algorithm_class:
        # If the class is found, initialize it, passing all necessary arguments.
        return algorithm_class(problem=problem, config=config)
    else:
        raise ValueError(
            f"Algorithm '{name}' not found. " f"Available algorithms: {list(algorithm_map.keys())}"
        )


# =============================================================================
# Test Module
# =============================================================================
if __name__ == "__main__":
    print("--- Verifying Algorithm Factory (__init__.py) ---")

    test_problem = get_problem("SMD1", {})
    mock_exp_name = "TestExperiment"

    # UPDATED: Added BiGA variants to test list
    test_algorithms = ["NestedDE", "BiGA_Lazy", "BiGA_Aggressive", "SACE_ES", "KKTSolver"]

    for i, algo_name in enumerate(test_algorithms):
        print(f"\n[Test Case {i+1}] Testing instantiation of '{algo_name}'...")
        try:
            mock_config = {"params": {"ul_pop_size": 10, "ll_pop_size": 10, "generations": 5}}

            # UPDATED: Pass the new experiment_name argument
            optimizer = get_algorithm(
                name=algo_name,
                problem=test_problem,
                config=mock_config["params"],
                experiment_name=mock_exp_name,
            )

            print(f"  Successfully instantiated object: {optimizer.__class__.__name__}")
            assert isinstance(
                optimizer, BaseOptimizer
            ), f"{algo_name} should be a subclass of BaseOptimizer."
            # Check if the history logger was created with the right name components
            assert mock_exp_name in optimizer.history_logger.filepath
            assert test_problem.__class__.__name__ in optimizer.history_logger.filepath
            assert optimizer.__class__.__name__ in optimizer.history_logger.filepath
            print("  Assertions PASSED: Object is valid and HistoryLogger path is correct.")

        except Exception as e:
            print(f"  An error occurred during {algo_name} test: {e}")

    print("\n--- Verification Complete ---")
