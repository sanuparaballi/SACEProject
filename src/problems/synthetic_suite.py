#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:00:31 2025

@author: sanup
"""

# sace_project/src/problems/synthetic_suite.py

import numpy as np

# We import the base class from our existing smd_suite file to ensure consistency.
from .smd_suite import BilevelProblem


class SP1(BilevelProblem):
    """
    Implements the Synthetic Problem 1 (SP1).

    This is a scalable bilevel problem with a non-convex lower level.
    - Upper Level (UL): A simple sphere function centered on the lower-level solution.
    - Lower Level (LL): The Rastrigin function, which is highly multimodal.
    """

    def __init__(self, n_dim=10):
        """
        Initializes the scalable problem.

        Args:
            n_dim (int): The number of dimensions for both UL and LL variables.
        """
        ul_dim = n_dim
        ll_dim = n_dim
        ul_bounds = (-5.12, 5.12)
        ll_bounds = (-5.12, 5.12)
        super().__init__(ul_dim, ll_dim, ul_bounds, ll_bounds)
        self.n_dim = n_dim

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        """
        Computes objective values for SP1.

        Args:
            ul_vars (np.array): Upper-level vector 'x' of shape (n_dim,).
            ll_vars (np.array): Lower-level vector 'y' of shape (n_dim,).
        """
        # Upper-Level Objective: Sphere function
        ul_objective = np.sum((ul_vars - ll_vars) ** 2)

        # Lower-Level Objective: Rastrigin function
        # The term `(ll_vars - ul_vars)` links the two levels.
        z = ll_vars - ul_vars
        ll_objective = 10 * self.n_dim + np.sum(z**2 - 10 * np.cos(2 * np.pi * z))

        return ul_objective, ll_objective


class SP2(BilevelProblem):
    """
    Implements the Synthetic Problem 2 (SP2).

    This is a scalable mixed-integer problem. The upper level is continuous,
    while the lower level must select integer quantities. This mimics a simple
    resource allocation or knapsack-type problem.
    """

    def __init__(self, n_dim=10):
        """
        Args:
            n_dim (int): The number of items/dimensions.
        """
        ul_dim = n_dim
        ll_dim = n_dim
        ul_bounds = (0.5, 1.5)  # Represents pricing factors or priorities
        ll_bounds = (0, 10)  # Represents integer quantities of items
        super().__init__(ul_dim, ll_dim, ul_bounds, ll_bounds)
        self.n_dim = n_dim

        # Pre-defined weights and values for the items (for reproducibility)
        np.random.seed(42)
        self.item_values = np.random.rand(n_dim) * 10
        self.item_weights = np.random.rand(n_dim) * 5
        self.knapsack_capacity = n_dim * 2

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        """
        Computes objective values for SP2.

        Args:
            ul_vars (np.array): Continuous UL vars 'x' (priorities).
            ll_vars (np.array): LL vars 'y'. Note: these should be treated as integers.
        """
        # Ensure lower-level variables are integers for the evaluation
        y_int = np.round(ll_vars).astype(int)

        # Upper-Level Objective: Maximize the value of chosen items, influenced
        # by the leader's priority vector 'ul_vars'.
        # We negate it since we are minimizing.
        ul_objective = -np.sum(ul_vars * self.item_values * y_int)

        # Lower-Level Objective: The follower tries to select items that have low
        # weight to conserve capacity.
        total_weight = np.sum(self.item_weights * y_int)
        ll_objective = total_weight

        # The follower is constrained by the knapsack capacity.
        # We add a penalty if the capacity is exceeded.
        penalty_param = 1e6
        if total_weight > self.knapsack_capacity:
            ll_objective += penalty_param * (total_weight - self.knapsack_capacity)

        return ul_objective, ll_objective


def get_synthetic_problem(name: str, n_dim: int = 10):
    """
    Factory function to get a synthetic problem instance.
    """
    problem_map = {
        "sp1": SP1,
        "sp2": SP2,
    }
    problem_class = problem_map.get(name.lower())
    if problem_class:
        return problem_class(n_dim=n_dim)
    else:
        raise ValueError(f"Problem '{name}' not found in the synthetic suite.")


# =============================================================================
# Test Module
# =============================================================================
if __name__ == "__main__":
    print("--- Verifying synthetic_suite.py Implementations ---")

    # Test Case 1: SP1
    print("\n[Test Case 1] Instantiating and evaluating SP1 (10D)...")
    try:
        sp1_problem = get_synthetic_problem("SP1", n_dim=10)
        print(f"Successfully instantiated: {sp1_problem}")

        ul_vars = np.ones(sp1_problem.ul_dim)
        ll_vars = np.ones(sp1_problem.ll_dim)

        ul_obj, ll_obj = sp1_problem.evaluate(ul_vars, ll_vars)
        print(f"Evaluating at x=ones(10), y=ones(10):")
        print(f"  UL Obj: {ul_obj:.4f}, LL Obj: {ll_obj:.4f}")

        # Optimal for LL is y=x, which makes z=0. LL obj should be 0.
        ll_vars_opt = np.copy(ul_vars)
        ul_obj_opt, ll_obj_opt = sp1_problem.evaluate(ul_vars, ll_vars_opt)
        print(f"Evaluating at rational response y=x:")
        print(f"  UL Obj: {ul_obj_opt:.4f} (Expected: 0.0)")
        print(f"  LL Obj: {ll_obj_opt:.4f} (Expected: 0.0)")
        assert np.isclose(ll_obj_opt, 0.0)
        print("  Assertion PASSED.")

    except Exception as e:
        print(f"An error occurred during SP1 test: {e}")

    # Test Case 2: SP2
    print("\n[Test Case 2] Instantiating and evaluating SP2 (5D)...")
    try:
        sp2_problem = get_synthetic_problem("SP2", n_dim=5)
        print(f"Successfully instantiated: {sp2_problem}")

        ul_vars = np.random.uniform(sp2_problem.ul_bounds[0], sp2_problem.ul_bounds[1], sp2_problem.ul_dim)
        ll_vars = np.random.uniform(sp2_problem.ll_bounds[0], sp2_problem.ll_bounds[1], sp2_problem.ll_dim)

        ul_obj, ll_obj = sp2_problem.evaluate(ul_vars, ll_vars)
        print(f"Evaluating at random point:")
        print(f"  UL Vars (priorities): {np.round(ul_vars, 2)}")
        print(f"  LL Vars (quantities): {np.round(ll_vars).astype(int)}")
        print(f"  UL Obj (neg-value): {ul_obj:.4f}")
        print(f"  LL Obj (weight): {ll_obj:.4f}")
        print("  Evaluation successful.")

    except Exception as e:
        print(f"An error occurred during SP2 test: {e}")

    print("\n--- Verification Complete ---")
