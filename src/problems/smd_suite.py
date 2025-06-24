#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:52:47 2025

@author: sanup
"""


# sace_project/src/problems/smd_suite.py

import numpy as np


class BilevelProblem:
    """
    Base class for a bilevel optimization problem.
    UPDATED: The evaluate method now accepts the 'add_penalty' flag.
    """

    def __init__(self, ul_dim, ll_dim, ul_bounds, ll_bounds):
        self.ul_dim = ul_dim
        self.ll_dim = ll_dim
        self.ul_bounds = np.array(ul_bounds)
        self.ll_bounds = np.array(ll_bounds)
        self.num_ll_constraints = 0

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        """
        The 'add_penalty' flag is included for compatibility but may not be used
        by unconstrained problems.
        """
        raise NotImplementedError("The evaluate method must be implemented by a subclass.")

    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        raise NotImplementedError(
            f"KKT Inapplicable: LL gradient not implemented for {self.__class__.__name__} (likely non-differentiable)."
        )

    def evaluate_ll_constraints(self, ul_vars, ll_vars):
        return np.array([])

    def evaluate_ll_constraint_gradient(self, ul_vars, ll_vars):
        raise NotImplementedError(
            f"KKT Inapplicable: LL constraint gradient not implemented for {self.__class__.__name__}"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(UL_dim={self.ul_dim}, LL_dim={self.ll_dim})"


class SMD1(BilevelProblem):
    def __init__(self):
        super().__init__(ul_dim=2, ll_dim=2, ul_bounds=(-5, 10), ll_bounds=(-5, 10))

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x1, x2 = ul_vars[0], ul_vars[1]
        y1, y2 = ll_vars[0], ll_vars[1]
        ul_objective = (x1 - 1) ** 2 + (x2 - 1) ** 2 + (y1 - 1) ** 2
        term1_ll = y1 - (x1**2 - x2)
        term2_ll = y2 - (-x1 + x2**2)
        ll_objective = term1_ll**2 + term2_ll**2
        return ul_objective, ll_objective

    # UPDATED: Restored the gradient calculation for SMD1
    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        """
        Calculates the gradient of the lower-level objective w.r.t y.
        f(y) = (y1 - A)^2 + (y2 - B)^2
        grad_y = [2*(y1-A), 2*(y2-B)]
        """
        x1, x2 = ul_vars[0], ul_vars[1]
        y1, y2 = ll_vars[0], ll_vars[1]

        A = x1**2 - x2
        B = -x1 + x2**2

        df_dy1 = 2 * (y1 - A)
        df_dy2 = 2 * (y2 - B)
        return np.array([df_dy1, df_dy2])


class SMD2(BilevelProblem):
    def __init__(self):
        super().__init__(ul_dim=2, ll_dim=2, ul_bounds=(-5, 10), ll_bounds=(-5, 10))

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x1, x2 = ul_vars[0], ul_vars[1]
        y1, y2 = ll_vars[0], ll_vars[1]
        ul_objective = (x1 - 1) ** 2 + (y1 - 1) ** 2
        term1_ll = y1 - (x1**2 - 2 * x2)
        term2_ll = y2 - (x1 * np.sin(np.pi * x2) + x2 * np.cos(np.pi * x1))
        ll_objective = term1_ll**2 + term2_ll**2
        return ul_objective, ll_objective


class SMD3(BilevelProblem):
    def __init__(self):
        super().__init__(ul_dim=2, ll_dim=2, ul_bounds=(-5, 10), ll_bounds=(-5, 10))

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x1, x2, y1, y2 = *ul_vars, *ll_vars
        p, r = 2.0, 0.1
        ul_objective = (x1 - 1) ** 2 + (x2 - 1) ** 2 + (y1 - 1) ** 2
        term1_ll = (y1 - x1**2 + x2) ** 2 + (y2 - (-x1 + x2**2)) ** 2
        term2_ll = (
            (y1 - x1) ** p + (y2 - x2) ** p - r * np.cos(2 * np.pi * ((y1 - x1) ** p + (y2 - x2) ** p))
        )
        ll_objective = term1_ll * term2_ll
        return ul_objective, ll_objective


class SMD4(BilevelProblem):
    def __init__(self):
        super().__init__(ul_dim=2, ll_dim=2, ul_bounds=(-5, 10), ll_bounds=(-5, 10))

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x1, x2, y1, y2 = *ul_vars, *ll_vars
        ul_objective = (x1 - 1) ** 2 + (x2 - 1) ** 2 + (y1 - 1) ** 2 + np.exp((y2 - 1) ** 2)
        ll_objective = -np.exp(-((y1 - x1**2) ** 2 + (y2 - x2**2) ** 2)) + 0.01 * (
            (y1 - x1) ** 2 + (y2 - x2) ** 2
        )
        return ul_objective, ll_objective


class SMD5(BilevelProblem):
    def __init__(self):
        super().__init__(ul_dim=2, ll_dim=2, ul_bounds=(-5, 10), ll_bounds=(-5, 10))

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x1, x2, y1, y2 = *ul_vars, *ll_vars
        ul_objective = (x1 - 1) ** 2 + (x2 - 1) ** 2 + (y1 - 1) ** 2 + (y2 - 1) ** 2
        ll_objective = ((y1 - x1) ** 2 + (y2 - x2) ** 2) * ((y1 - x1**2) ** 2 + (y2 - x2**2) ** 2 + 1e-4)
        return ul_objective, ll_objective


class SMD6(BilevelProblem):
    def __init__(self):
        super().__init__(ul_dim=1, ll_dim=1, ul_bounds=(1, 10), ll_bounds=(1, 10))
        self.num_ll_constraints = 1

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x, y = ul_vars[0], ll_vars[0]
        ul_objective = (x - 1) ** 2 + (y - 1) ** 2
        ll_objective = (y - (1 + 0.1 * x**2)) ** 2

        if add_penalty:
            C = (x - 4) ** 2 + (y - 4) ** 2 - 9  # UL constraint
            c = (x - 5) ** 2 + y**2 - 25  # LL constraint
            penalty_param = 1e6
            if C > 0:
                ul_objective += penalty_param * C
            if c > 0:
                ll_objective += penalty_param * c
        return ul_objective, ll_objective

    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        x, y = ul_vars[0], ll_vars[0]
        df_dy = 2 * (y - (1 + 0.1 * x**2))
        return np.array([df_dy])

    def evaluate_ll_constraints(self, ul_vars, ll_vars):
        x, y = ul_vars[0], ll_vars[0]
        return np.array([(x - 5) ** 2 + y**2 - 25])

    def evaluate_ll_constraint_gradient(self, ul_vars, ll_vars):
        x, y = ul_vars[0], ll_vars[0]
        dc_dy = 2 * y
        return np.array([[dc_dy]])


# NEWLY ADDED PROBLEMS
class SMD7(BilevelProblem):
    """Interdependent constraints at both levels."""

    def __init__(self):
        super().__init__(ul_dim=2, ll_dim=2, ul_bounds=(-5, 10), ll_bounds=(-5, 10))
        self.num_ll_constraints = 2

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x1, x2, y1, y2 = *ul_vars, *ll_vars
        ul_objective = (x1 - 1) ** 2 + (x2 - 1) ** 2 + (y1 - 1) ** 2 + (y2 - 1) ** 2
        ll_objective = (y1 - x1) ** 2 + (y2 - x2) ** 2
        if add_penalty:
            C1 = 2 * x1 - y1 - y2
            C2 = 2 * x2 - y1 - y2
            c1 = (y1 - x1) ** 2 + (y2 - x2) ** 2 - 1
            c2 = y1 + y2 - 2
            if C1 < 0:
                ul_objective += 1e6 * abs(C1)
            if C2 < 0:
                ul_objective += 1e6 * abs(C2)
            if c1 > 0:
                ll_objective += 1e6 * c1
            if c2 > 0:
                ll_objective += 1e6 * c2
        return ul_objective, ll_objective

    def evaluate_ll_constraints(self, ul_vars, ll_vars):
        x1, x2, y1, y2 = *ul_vars, *ll_vars
        return np.array([(y1 - x1) ** 2 + (y2 - x2) ** 2 - 1, y1 + y2 - 2])


class SMD8(BilevelProblem):
    """Vanishing feasible region at the lower level."""

    def __init__(self):
        super().__init__(ul_dim=2, ll_dim=2, ul_bounds=(-5, 10), ll_bounds=(-5, 10))
        self.num_ll_constraints = 1

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x1, x2, y1, y2 = *ul_vars, *ll_vars
        ul_objective = x1**2 + x2**2 + y1**2 + y2**2 + (y1 - y2) ** 2
        ll_objective = (y1 - 1) ** 2 + (y2 - 1) ** 2
        if add_penalty:
            c1 = (y1 - x1) ** 2 + (y2 - x2) ** 2 - np.cos(x1 + x2) - 1
            if c1 > 0:
                ll_objective += 1e6 * c1
        return ul_objective, ll_objective

    def evaluate_ll_constraints(self, ul_vars, ll_vars):
        x1, x2, y1, y2 = *ul_vars, *ll_vars
        return np.array([(y1 - x1) ** 2 + (y2 - x2) ** 2 - np.cos(x1 + x2) - 1])


def get_smd_problem(name: str):
    problem_map = {
        "smd1": SMD1,
        "smd2": SMD2,
        "smd3": SMD3,
        "smd4": SMD4,
        "smd5": SMD5,
        "smd6": SMD6,
        "smd7": SMD7,
        "smd8": SMD8,
    }
    problem_class = problem_map.get(name.lower())
    if problem_class:
        return problem_class()
    raise ValueError(f"Problem '{name}' not found in SMD suite.")


# =============================================================================
# Test Module
# =============================================================================
if __name__ == "__main__":
    print("--- Verifying smd_suite.py (with Gradients) ---")

    # Test SMD1 Gradient
    print("\n[Test] Verifying SMD1 gradient...")
    smd1 = get_smd_problem("SMD1")
    ul_vars = np.array([1.0, 1.0])
    ll_vars = np.array([0.0, 0.0])  # At the LL optimum for this UL
    grad = smd1.evaluate_ll_gradient(ul_vars, ll_vars)
    print(f"  Gradient at x=(1,1), y=(0,0): {grad} (Expected: [0, 0])")
    assert np.allclose(grad, [0, 0]), "SMD1 gradient is incorrect!"
    print("  Assertion PASSED.")

    # Test SMD6 Gradient and Constraints
    print("\n[Test] Verifying SMD6 gradients and constraints...")
    smd6 = get_smd_problem("SMD6")
    ul_vars_test = np.array([5.0])
    ll_vars_test = np.array([1.0])

    # Constraint value
    c_val = smd6.evaluate_ll_constraints(ul_vars_test, ll_vars_test)
    expected_c = (5.0 - 5.0) ** 2 + 1.0**2 - 25.0  # Should be -24.0
    print(f"  LL constraint value at x=5, y=1: {c_val[0]:.2f} (Expected: {expected_c:.2f})")
    assert np.isclose(c_val[0], expected_c), "SMD6 constraint value is incorrect."

    # Constraint gradient
    c_grad = smd6.evaluate_ll_constraint_gradient(ul_vars_test, ll_vars_test)
    expected_c_grad = 2.0 * 1.0  # Should be 2.0
    print(f"  LL constraint gradient at x=5, y=1: {c_grad[0][0]:.2f} (Expected: {expected_c_grad:.2f})")
    assert np.isclose(c_grad[0][0], expected_c_grad), "SMD6 constraint gradient is incorrect."
    print("  Assertions PASSED.")

    # Test for unimplemented gradient
    print("\n[Test] Verifying error for unimplemented gradient (SMD2)...")
    smd2 = get_smd_problem("SMD2")
    try:
        smd2.evaluate_ll_gradient(np.array([1, 1]), np.array([1, 1]))
    except NotImplementedError as e:
        print(f"  Successfully caught expected error: {e}")
        print("  Assertion PASSED.")

    print("\n--- Verification Complete ---")
