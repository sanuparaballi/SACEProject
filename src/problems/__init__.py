#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:30:32 2025

@author: sanup
"""


# sace_project/src/problems/__init__.py

# Import the specific factory functions from each suite file
# Import all problem definition functions/classes
from .smd_suite import SMD1, SMD2, SMD3, SMD4, SMD5, SMD6, SMD7, SMD8, SMD9, SMD10, SMD11, SMD12
from .synthetic_suite import SP1, SP2
from .multimodal_suite import SA1
from .hyper_suite import HyperRepresentation


def get_problem(name: str, params: dict):
    """
    Main factory function to get any problem instance by its name.

    This function delegates the creation of the problem to the appropriate
    suite-specific factory function based on the problem name.
    Factory function to get a problem instance by its name.
    UPDATED: Now includes all new problem suites.
    """
    problem_map = {
        # SMD Suite
        "smd1": SMD1,
        "smd2": SMD2,
        "smd3": SMD3,
        "smd4": SMD4,
        "smd5": SMD5,
        "smd6": SMD6,
        "smd7": SMD7,
        "smd8": SMD8,
        "smd9": SMD9,
        "smd10": SMD10,
        "smd11": SMD11,
        "smd12": SMD12,
        # Synthetic Suite
        "sp1": SP1,
        "sp2": SP2,
        # Multimodal Suite
        "sa1": SA1,
        # Hyper-representation Suite
        "hyper_representation": HyperRepresentation,
    }

    problem_class = problem_map.get(name.lower())

    if not problem_class:
        raise ValueError(f"Problem '{name}' not found in the factory. Available: {list(problem_map.keys())}")

    # Instantiate the class with its specific parameters from the JSON file
    return problem_class(**params)


# =============================================================================
# Test Module
# =============================================================================
if __name__ == "__main__":
    print("--- Verifying Problem Factory (__init__.py) ---")

    # Test cases for each suite
    test_cases = [
        {"name": "SMD1", "params": {}},
        {"name": "SP1", "params": {"n_dim": 20}},
        {"name": "VC1", "params": {}},
        {"name": "Hyper_Representation", "params": {"n": 100, "m": 10}},  # New test case
    ]

    for i, case in enumerate(test_cases):
        print(f"\n[Test Case {i+1}] Testing instantiation of '{case['name']}'...")
        try:
            problem = get_problem(case["name"], case["params"])
            print(f"  Successfully instantiated object: {problem}")

            # Verify the type and parameters
            if case["name"] == "SMD1":
                assert isinstance(problem, SMD1), "Incorrect problem type."
            elif case["name"] == "SP1":
                assert isinstance(problem, SP1), "Incorrect problem type."
                assert problem.n_dim == 20, "Dimension not set correctly."
            elif case["name"] == "VC1":
                assert isinstance(problem, SA1), "Incorrect problem type."
            elif case["name"] == "Hyper_Representation":
                assert isinstance(problem, HyperRepresentation), "Incorrect problem type."

            print("  Assertion PASSED: Correct type and parameters.")

        except Exception as e:
            print(f"  An error occurred during {case['name']} test: {e}")

    # Test for an unknown problem
    print("\n[Test Case 5] Testing for an unknown problem...")
    try:
        get_problem(name="UnknownProblem", params={})
    except ValueError as e:
        print(f"  Successfully caught expected error: {e}")
        print("  Assertion PASSED: Factory correctly handles unknown names.")

    print("\n--- Verification Complete ---")
