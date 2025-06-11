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

# For testing the factory, we need a mock problem.
from ..problems import get_problem


def get_algorithm(name: str, problem, config: dict) -> BaseOptimizer:
    """
    Factory function to get an algorithm instance by its name.

    This function acts as a central registry for all implemented algorithms.
    It takes the algorithm's name and returns an initialized object.

    Args:
        name (str): The name of the algorithm (e.g., "NestedDE", "SACE_ES").
        problem: The problem instance for the algorithm to solve.
        config (dict): The configuration dictionary for the algorithm.

    Returns:
        An initialized instance of a subclass of BaseOptimizer.

    Raises:
        ValueError: If the specified algorithm name is not found.
    """
    # A dictionary mapping names to the algorithm classes.
    # We use lowercase names for case-insensitive matching.
    algorithm_map = {
        "nestedde": NestedDE,
        "biga_lazy": BiGA,
        "biga_aggressive": BiGA,
        "sace_es": SACE_ES,
        "kktsolver": KKTSolver,
    }

    algorithm_class = algorithm_map.get(name.lower())

    if algorithm_class:
        # If the class is found, initialize it with the problem and config
        return algorithm_class(problem=problem, config=config)
    else:
        # If the name is not in our map, raise an error.
        raise ValueError(
            f"Algorithm '{name}' not found. " f"Available algorithms: {list(algorithm_map.keys())}"
        )


# =============================================================================
# Test Module
# =============================================================================
if __name__ == "__main__":
    print("--- Verifying Algorithm Factory (__init__.py) ---")

    # We need a problem instance that supports gradients for the KKT test.
    # SMD1 is suitable for this.
    test_problem = get_problem("SMD1", {})

    test_algorithms = ["NestedDE", "BiGA", "SACE_ES", "KKTSolver"]

    for i, algo_name in enumerate(test_algorithms):
        print(f"\n[Test Case {i+1}] Testing instantiation of '{algo_name}'...")
        try:
            # A minimal config for each algorithm
            mock_config = {"params": {"ul_pop_size": 10, "ll_pop_size": 10, "generations": 5}}

            optimizer = get_algorithm(name=algo_name, problem=test_problem, config=mock_config["params"])

            print(f"  Successfully instantiated object: {optimizer.__class__.__name__}")
            assert isinstance(
                optimizer, BaseOptimizer
            ), f"{algo_name} should be a subclass of BaseOptimizer."
            print("  Assertion PASSED: Object is a valid BaseOptimizer instance.")

        except Exception as e:
            print(f"  An error occurred during {algo_name} test: {e}")

    # Test for an unknown algorithm
    print(f"\n[Test Case {len(test_algorithms)+1}] Testing for an unknown algorithm...")
    try:
        get_algorithm(name="UnknownAlgo", problem=test_problem, config={})
    except ValueError as e:
        print(f"  Successfully caught expected error: {e}")
        print("  Assertion PASSED: Factory correctly handles unknown names.")

    print("\n--- Verification Complete ---")
