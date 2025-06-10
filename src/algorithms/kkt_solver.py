#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:42:36 2025

@author: sanup
"""

# sace_project/src/algorithms/kkt_solver.py

import numpy as np
from scipy.optimize import minimize
from .base_optimizer import BaseOptimizer

# For testing, we'll need a problem definition. We'll assume it's updated with gradients.
from ..problems.smd_suite import get_smd_problem


class KKTSolver(BaseOptimizer):
    """
    Implements a bilevel optimization solver based on KKT reformulation.

    This approach replaces the lower-level optimization problem with its KKT
    optimality conditions, creating a single-level Mathematical Program with
    Equilibrium Constraints (MPEC), which is then solved using a standard
    Nonlinear Programming (NLP) solver (scipy's SLSQP).

    NOTE: This solver requires the problem object to provide methods for
    evaluating gradients of the lower-level objective and constraints.
    """

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self.max_iter = self.config.get("max_iter", 200)
        self.method = "SLSQP"

    def solve(self):
        """
        Sets up and solves the MPEC.
        """
        # The variables for the single-level problem are a concatenation of
        # upper-level vars (x), lower-level vars (y), and Lagrange multipliers (lambda).
        # We start from a random initial guess.
        initial_x = np.random.uniform(
            self.problem.ul_bounds[0], self.problem.ul_bounds[1], self.problem.ul_dim
        )
        initial_y = np.random.uniform(
            self.problem.ll_bounds[0], self.problem.ll_bounds[1], self.problem.ll_dim
        )

        # Check if the problem's lower level is constrained
        has_ll_constraints = (
            hasattr(self.problem, "evaluate_ll_constraints") and self.problem.num_ll_constraints > 0
        )

        if has_ll_constraints:
            # Case with lower-level constraints
            num_lambda = self.problem.num_ll_constraints
            initial_lambda = np.random.rand(num_lambda)
            initial_guess = np.concatenate([initial_x, initial_y, initial_lambda])
        else:
            # Case without lower-level constraints (KKT conditions simplify)
            num_lambda = 0
            initial_guess = np.concatenate([initial_x, initial_y])

        # Define bounds for all variables (x, y, and lambda >= 0)
        bounds = [(self.problem.ul_bounds[0], self.problem.ul_bounds[1])] * self.problem.ul_dim + [
            (self.problem.ll_bounds[0], self.problem.ll_bounds[1])
        ] * self.problem.ll_dim
        if has_ll_constraints:
            bounds += [(0, None)] * num_lambda  # Lagrange multipliers must be non-negative

        # Define the objective function for the NLP solver (which is the UL objective)
        def objective(z):
            x = z[: self.problem.ul_dim]
            y = z[self.problem.ul_dim : self.problem.ul_dim + self.problem.ll_dim]
            self.ul_nfe += 1  # Count each objective call as an evaluation
            return self.problem.evaluate(x, y)[0]

        # Define the KKT conditions as equality and inequality constraints
        constraints = []

        # 1. Stationarity condition: grad_y(L) = 0
        def stationarity_constraint(z):
            x = z[: self.problem.ul_dim]
            y = z[self.problem.ul_dim : self.problem.ul_dim + self.problem.ll_dim]
            grad_f = self.problem.evaluate_ll_gradient(x, y)  # Requires new problem method
            self.ll_nfe += 1  # Count each gradient call as an evaluation

            if has_ll_constraints:
                lmbda = z[self.problem.ul_dim + self.problem.ll_dim :]
                grad_g = self.problem.evaluate_ll_constraint_gradient(x, y)  # Requires new method
                return grad_f + np.dot(lmbda, grad_g)
            else:
                return grad_f  # Simplified for unconstrained LL

        constraints.append({"type": "eq", "fun": stationarity_constraint})

        if has_ll_constraints:
            # 2. Primal feasibility: g(x, y) <= 0
            def primal_feasibility(z):
                x = z[: self.problem.ul_dim]
                y = z[self.problem.ul_dim : self.problem.ul_dim + self.problem.ll_dim]
                return -self.problem.evaluate_ll_constraints(x, y)  # SLSQP expects g(x) >= 0

            constraints.append({"type": "ineq", "fun": primal_feasibility})

            # 3. Complementary slackness: lambda * g(x, y) = 0
            def complementary_slackness(z):
                x = z[: self.problem.ul_dim]
                y = z[self.problem.ul_dim : self.problem.ul_dim + self.problem.ll_dim]
                lmbda = z[self.problem.ul_dim + self.problem.ll_dim :]
                g = self.problem.evaluate_ll_constraints(x, y)
                return np.dot(lmbda, g)

            constraints.append({"type": "eq", "fun": complementary_slackness})

        # Solve the reformulated problem
        result = minimize(
            objective,
            initial_guess,
            method=self.method,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter, "disp": False},
        )

        # Extract results
        if result.success:
            final_x = result.x[: self.problem.ul_dim]
            final_y = result.x[self.problem.ul_dim : self.problem.ul_dim + self.problem.ll_dim]
            final_fitness = result.fun
        else:
            # If solver fails, return poor results
            final_x, final_y, final_fitness = initial_x, initial_y, np.inf

        final_results = {
            "final_ul_fitness": final_fitness,
            "total_ul_nfe": self.ul_nfe,
            "total_ll_nfe": self.ll_nfe,
            "best_ul_solution": final_x,
            "corresponding_ll_solution": final_y,
        }
        return final_results


# =============================================================================
# Test Module
# =============================================================================
if __name__ == "__main__":
    print("--- Verifying KKTSolver Algorithm Implementation ---")

    # To test this, we need a problem that provides gradients.
    # We will create a mock version of SMD1 with the required methods.
    class MockSMD1WithGrad(get_smd_problem("SMD1")):
        def evaluate_ll_gradient(self, ul_vars, ll_vars):
            x1, x2 = ul_vars[0], ul_vars[1]
            y1, y2 = ll_vars[0], ll_vars[1]
            # Gradient of f = (y1 - (x1^2 - x2))^2 + (y2 - (-x1 + x2^2))^2 w.r.t y
            df_dy1 = 2 * (y1 - (x1**2 - x2))
            df_dy2 = 2 * (y2 - (-x1 + x2**2))
            return np.array([df_dy1, df_dy2])

        @property
        def num_ll_constraints(self):
            return 0  # SMD1 is unconstrained at the lower level

    mock_problem = MockSMD1WithGrad()
    test_config = {"max_iter": 1000}

    print(f"\nProblem: MockSMD1WithGrad")
    print(f"Config: {test_config}")

    try:
        optimizer = KKTSolver(mock_problem, test_config)
        results = optimizer.solve()

        print("\n--- Optimization Finished ---")
        print(f"Solver Succeeded: {results['final_ul_fitness'] != np.inf}")
        print(f"Final UL Fitness: {results['final_ul_fitness']:.4f}")
        print(f"Best UL Solution: {np.round(results['best_ul_solution'], 3)}")
        print(f"Corresponding LL Solution: {np.round(results['corresponding_ll_solution'], 3)}")
        print("Expected UL solution for SMD1 is near [1, 1]")
        print("\n--- Verification Complete ---")

    except Exception as e:
        print(f"\nAn error occurred during the test run: {e}")
