#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:30:32 2025

@author: sanup
"""


# sace_project/src/problems/__init__.py

# Import the specific factory functions from each suite file
from .smd_suite import get_smd_problem, SMD1
from .synthetic_suite import get_synthetic_problem, SP1
from .multimodal_suite import get_multimodal_problem, VC1
from .hyper_suite import get_hyper_problem, HyperRepresentation  # UPDATED: Import the new suite


def get_problem(name: str, params: dict):
    """
    Main factory function to get any problem instance by its name.

    This function delegates the creation of the problem to the appropriate
    suite-specific factory function based on the problem name.

    Args:
        name (str): The name of the problem (e.g., "SMD1", "SP1", "VC1").
        params (dict): A dictionary of parameters for the problem, such as
                       'n_dim' for scalable problems.

    Returns:
        An initialized instance of a BilevelProblem subclass.

    Raises:
        ValueError: If the specified problem name is not found in any suite.
    """
    name_lower = name.lower()

    # Check which suite the problem belongs to
    if name_lower.startswith("smd"):
        return get_smd_problem(name_lower)
    elif name_lower.startswith("sp"):
        n_dim = params.get("n_dim", 10)
        return get_synthetic_problem(name_lower, n_dim=n_dim)
    elif name_lower.startswith("vc"):
        return get_multimodal_problem(name_lower)
    # UPDATED: Add logic to handle the new Hyper-representation problem
    elif name_lower.startswith("hyper"):
        return get_hyper_problem(name_lower, params)
    else:
        # If the problem does not match any known prefix, raise an error.
        raise ValueError(f"Problem '{name}' not recognized or suite not supported.")


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
                assert isinstance(problem, VC1), "Incorrect problem type."
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
