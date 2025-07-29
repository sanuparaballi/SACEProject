#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:01:56 2025

@author: sanup
"""

# sace_project/src/problems/multimodal_suite.py


import numpy as np
from .base_problem import BaseBilevelProblem


class SA1(BaseBilevelProblem):
    """
    Implements the Sanup Araballi 1 (SA1) test problem.

    This is a massively multimodal bilevel problem designed to be extremely
    difficult for local searchers and simple heuristics.
    - Upper Level (UL): A simple sphere function trying to match the follower's choice.
    - Lower Level (LL): The Shubert function, which has 760 local minima and
      18 global minima in its 2D form.
    """

    def __init__(self):
        """
        Initializes the VC1 problem. The dimensions are fixed at 2 for both levels.
        """
        ul_dim = 2
        ll_dim = 2
        ul_bounds = (-5, 5)
        ll_bounds = (-10, 10)
        super().__init__(ul_dim, ll_dim, ul_bounds, ll_bounds, "SA1")
        self.beta = 0.1  # Coupling parameter

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        """
        Computes objective values for VC1.

        Args:
            ul_vars (np.array): Upper-level vector 'x' of shape (2,).
            ll_vars (np.array): Lower-level vector 'y' of shape (2,).
        """
        x1, x2 = ul_vars[0], ul_vars[1]
        y1, y2 = ll_vars[0], ll_vars[1]

        # Upper-Level Objective: Simple Sphere function
        ul_objective = (x1 - y1) ** 2 + (x2 - y2) ** 2

        # Lower-Level Objective: Shubert function + coupling term
        sum1 = 0
        sum2 = 0
        for i in range(1, 6):
            sum1 += i * np.cos((i + 1) * y1 + i)
            sum2 += i * np.cos((i + 1) * y2 + i)

        shubert_value = sum1 * sum2

        # Coupling term links the leader's decision to the follower's landscape
        coupling_term = self.beta * ((y1 - x1) ** 2 + (y2 - x2) ** 2)
        ll_objective = shubert_value + coupling_term

        return ul_objective, ll_objective


def get_multimodal_problem(name: str):
    """
    Factory function to get a multimodal problem instance.
    """
    problem_map = {
        "sa1": SA1,
    }
    problem_class = problem_map.get(name.lower())
    if problem_class:
        return problem_class()
    else:
        raise ValueError(f"Problem '{name}' not found in the multimodal suite.")


# =============================================================================
# Test Module
# =============================================================================
if __name__ == "__main__":
    print("--- Verifying multimodal_suite.py Implementations ---")

    # Test Case 1: SA1
    print("\n[Test Case 1] Instantiating and evaluating SA1...")
    try:
        sa1_problem = get_multimodal_problem("SA1")
        print(f"Successfully instantiated: {sa1_problem}")
        print(f"  UL variable dimension: {sa1_problem.ul_dim}")
        print(f"  LL variable dimension: {sa1_problem.ll_dim}")
        print(f"  UL variable bounds: {sa1_problem.ul_bounds.tolist()}")
        print(f"  LL variable bounds: {sa1_problem.ll_bounds.tolist()}")

        # Evaluate at an arbitrary point to ensure the function works
        ul_vars = np.array([1.0, 1.0])
        ll_vars = np.array([-1.0, -1.0])

        ul_obj, ll_obj = sa1_problem.evaluate(ul_vars, ll_vars)

        print(f"\nEvaluating at an arbitrary point x=(1,1), y=(-1,-1):")
        print(f"  Upper-Level Objective F(x,y): {ul_obj:.4f}")
        print(f"  Lower-Level Objective f(x,y): {ll_obj:.4f}")
        print("  Evaluation successful.")

        # The global minimum of the standard Shubert function is approx -186.7309.
        # Our function is slightly different due to the coupling term.
        # This test mainly confirms the implementation runs without error.
        # The true test will be observing if an algorithm can find a very low
        # value for the lower-level objective.

    except Exception as e:
        print(f"An error occurred during SA1 test: {e}")

    print("\n--- Verification Complete ---")
