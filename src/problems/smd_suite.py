#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:52:47 2025

@author: sanup
"""


# sace_project/src/problems/smd_suite.py

# smd_benchmark_suite_v1.0.py

import numpy as np


class BilevelProblem:
    """
    Base class for the Sinha, Malo, Deb (SMD) bilevel optimization problem suite.
    This implementation adheres strictly to the definitions in KanGAL Report 2013002.
    """

    def __init__(self, p, q, r, s, name):
        self.p = p  # Dimension of x_u1
        self.q = q  # Dimension of x_l1
        self.r = r  # Dimension of x_u2 and x_l2
        self.s = s  # Additional dimension for x_l1 in SMD6
        self.name = name
        self.ul_dim = p + r
        self.ll_dim = q + s + r

    def _split_vars(self, ul_vars, ll_vars):
        """Helper to split variables according to the paper's formulation."""
        x_u1 = ul_vars[: self.p]
        x_u2 = ul_vars[self.p :]
        x_l1 = ll_vars[: self.q + self.s]
        x_l2 = ll_vars[self.q + self.s :]
        return x_u1, x_u2, x_l1, x_l2

    def evaluate(self, ul_vars, ll_vars):
        """Evaluates the raw upper and lower level objectives."""
        raise NotImplementedError("This method must be implemented by a subclass.")

    def evaluate_ul_constraints(self, ul_vars, ll_vars):
        """Evaluates upper-level constraints. Positive value means violation."""
        return np.array([])

    def evaluate_ll_constraints(self, ul_vars, ll_vars):
        """Evaluates lower-level constraints. Positive value means violation."""
        return np.array([])

    def __repr__(self):
        return f"{self.name}(UL_dim={self.ul_dim}, LL_dim={self.ll_dim})"


class SMD1(BilevelProblem):
    """Cooperative, convex problem. Ref: Eq. (12)."""

    def __init__(self, p=1, q=2, r=1, s=0):
        super().__init__(p, q, r, s, "SMD1")

    def evaluate(self, ul_vars, ll_vars):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)

        F1 = np.sum(x_u1**2)
        F2 = np.sum(x_l1**2)
        F3 = np.sum(x_u2**2) + np.sum((x_u2 - np.tan(x_l2)) ** 2)
        ul_objective = F1 + F2 + F3

        f2 = np.sum(x_l1**2)
        f3 = np.sum((x_u2 - np.tan(x_l2)) ** 2)
        ll_objective = f2 + f3

        return ul_objective, ll_objective


class SMD2(BilevelProblem):
    """Conflicting, convex problem. Ref: Eq. (15)."""

    def __init__(self, p=1, q=2, r=1, s=0):
        super().__init__(p, q, r, s, "SMD2")

    def evaluate(self, ul_vars, ll_vars):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)

        F1 = np.sum(x_u1**2)
        F2 = -np.sum(x_l1**2)
        F3 = np.sum(x_u2**2) - np.sum((x_u2 - np.log(x_l2)) ** 2)
        ul_objective = F1 + F2 + F3

        f2 = np.sum(x_l1**2)
        f3 = np.sum((x_u2 - np.log(x_l2)) ** 2)
        ll_objective = f2 + f3

        return ul_objective, ll_objective


class SMD3(BilevelProblem):
    """Cooperative, LL is multi-modal (Rastrigin). Ref: Eq. (18)."""

    def __init__(self, p=1, q=2, r=1, s=0):
        super().__init__(p, q, r, s, "SMD3")

    def evaluate(self, ul_vars, ll_vars):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)

        F1 = np.sum(x_u1**2)
        F2 = np.sum(x_l1**2)
        F3 = np.sum(x_u2**2) + np.sum((x_u2**2 - np.tan(x_l2)) ** 2)
        ul_objective = F1 + F2 + F3

        f2 = self.q + np.sum(x_l1**2 - np.cos(2 * np.pi * x_l1))
        f3 = np.sum((x_u2**2 - np.tan(x_l2)) ** 2)
        ll_objective = f2 + f3

        return ul_objective, ll_objective


class SMD4(BilevelProblem):
    """Conflicting, LL is multi-modal (Rastrigin). Ref: Eq. (21)."""

    def __init__(self, p=1, q=2, r=1, s=0):
        super().__init__(p, q, r, s, "SMD4")

    def evaluate(self, ul_vars, ll_vars):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)

        F1 = np.sum(x_u1**2)
        F2 = -np.sum(x_l1**2)
        F3 = np.sum(x_u2**2) - np.sum((np.abs(x_u2) - np.log(1 + x_l2)) ** 2)
        ul_objective = F1 + F2 + F3

        f2 = self.q + np.sum(x_l1**2 - np.cos(2 * np.pi * x_l1))
        f3 = np.sum((np.abs(x_u2) - np.log(1 + x_l2)) ** 2)
        ll_objective = f2 + f3

        return ul_objective, ll_objective


class SMD5(BilevelProblem):
    """Conflicting, LL is difficult to converge (Rosenbrock). Ref: Eq. (24)."""

    def __init__(self, p=1, q=2, r=1, s=0):
        super().__init__(p, q, r, s, "SMD5")
        if q < 1:
            raise ValueError("SMD5 requires q >= 1 for Rosenbrock function.")

    def evaluate(self, ul_vars, ll_vars):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)

        rosenbrock_f2 = np.sum((x_l1[1:] - x_l1[:-1] ** 2) ** 2 + (x_l1[:-1] - 1) ** 2)

        F1 = np.sum(x_u1**2)
        F2 = -rosenbrock_f2
        F3 = np.sum(x_u2**2) - np.sum((np.abs(x_u2) - x_l2**2) ** 2)
        ul_objective = F1 + F2 + F3

        f2 = rosenbrock_f2
        f3 = np.sum((np.abs(x_u2) - x_l2**2) ** 2)
        ll_objective = f2 + f3

        return ul_objective, ll_objective


class SMD6(BilevelProblem):
    """Conflicting, LL has multiple global solutions. Ref: Eq. (27)."""

    def __init__(self, p=1, q=0, r=1, s=2):
        super().__init__(p, q, r, s, "SMD6")
        if s < 2 or s % 2 != 0:
            raise ValueError("SMD6 requires s to be an even number >= 2.")

    def evaluate(self, ul_vars, ll_vars):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)

        f2_term2 = 0
        for i in range(self.q, self.q + self.s - 1, 2):
            f2_term2 += (x_l1[i + 1] - x_l1[i]) ** 2

        F1 = np.sum(x_u1**2)
        F2 = -np.sum(x_l1[: self.q] ** 2) + np.sum(x_l1[self.q :] ** 2)
        F3 = np.sum(x_u2**2) - np.sum((x_u2 - x_l2) ** 2)
        ul_objective = F1 + F2 + F3

        f2 = np.sum(x_l1[: self.q] ** 2) + f2_term2
        f3 = np.sum((x_u2 - x_l2) ** 2)
        ll_objective = f2 + f3

        return ul_objective, ll_objective


class SMD7(BilevelProblem):
    """Conflicting, UL is multi-modal (Griewank). Ref: Eq. (30)."""

    def __init__(self, p=1, q=2, r=1, s=0):
        super().__init__(p, q, r, s, "SMD7")

    def evaluate(self, ul_vars, ll_vars):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)

        cos_prod = np.prod([np.cos(x_u1[i] / np.sqrt(i + 1)) for i in range(self.p)])
        F1 = 1 + (1 / 400) * np.sum(x_u1**2) - cos_prod
        F2 = -np.sum(x_l1**2)
        F3 = np.sum(x_u2**2) - np.sum((x_u2 - np.log(x_l2)) ** 2)
        ul_objective = F1 + F2 + F3

        f2 = np.sum(x_l1**2)
        f3 = np.sum((x_u2 - np.log(x_l2)) ** 2)
        ll_objective = f2 + f3

        return ul_objective, ll_objective


class SMD8(BilevelProblem):
    """Conflicting, UL (Ackley) and LL (Rosenbrock) are multi-modal. Ref: Eq. (33)."""

    def __init__(self, p=1, q=2, r=1, s=0):
        super().__init__(p, q, r, s, "SMD8")
        if q < 1:
            raise ValueError("SMD8 requires q >= 1 for Rosenbrock function.")

    def evaluate(self, ul_vars, ll_vars):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)

        rosenbrock_f2 = np.sum((x_l1[1:] - x_l1[:-1] ** 2) ** 2 + (x_l1[:-1] - 1) ** 2)

        term1_F1 = -0.2 * np.sqrt((1 / self.p) * np.sum(x_u1**2))
        term2_F1 = (1 / self.p) * np.sum(np.cos(2 * np.pi * x_u1))
        F1 = 20 + np.e - 20 * np.exp(term1_F1) - np.exp(term2_F1)
        F2 = -rosenbrock_f2
        F3 = np.sum(x_u2**2) - np.sum((x_u2 - x_l2**3) ** 2)
        ul_objective = F1 + F2 + F3

        f2 = rosenbrock_f2
        f3 = np.sum((x_u2 - x_l2**3) ** 2)
        ll_objective = f2 + f3

        return ul_objective, ll_objective


class SMD9(BilevelProblem):
    """Constrained, conflicting, inactive constraints at optimum. Ref: Eq. (36-37)."""

    def __init__(self, p=1, q=2, r=1, s=0, a=1.0, b=1.0):
        super().__init__(p, q, r, s, "SMD9")
        self.a = a
        self.b = b

    def evaluate(self, ul_vars, ll_vars):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)

        F1 = np.sum(x_u1**2)
        F2 = -np.sum(x_l1**2)
        F3 = np.sum(x_u2**2) - np.sum((x_u2 - np.log(1 + x_l2)) ** 2)
        ul_objective = F1 + F2 + F3

        f2 = np.sum(x_l1**2)
        f3 = np.sum((x_u2 - np.log(1 + x_l2)) ** 2)
        ll_objective = f2 + f3

        return ul_objective, ll_objective

    def evaluate_ul_constraints(self, ul_vars, ll_vars):
        term = np.sum(ul_vars**2) / self.a
        # Constraint is G >= 0, so violation is -G if G < 0
        G1 = term - np.floor(term + 0.5 / self.b)
        return np.array([-G1 if G1 < 0 else 0])

    def evaluate_ll_constraints(self, ul_vars, ll_vars):
        term = np.sum(ll_vars**2) / self.a
        # Constraint is g >= 0, so violation is -g if g < 0
        g1 = term - np.floor(term + 0.5 / self.b)
        return np.array([-g1 if g1 < 0 else 0])


class SMD10(BilevelProblem):
    """Constrained, cooperative, active constraints at optimum. Ref: Eq. (40-41)."""

    def __init__(self, p=1, q=2, r=1, s=0):
        super().__init__(p, q, r, s, "SMD10")

    def evaluate(self, ul_vars, ll_vars):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)

        F1 = np.sum((x_u1 - 2) ** 2)
        F2 = np.sum(x_l1**2)  # Note: paper says sum(x_l1^2) not sum((x_l1-2)^2)
        F3 = np.sum((x_u2 - 2) ** 2) + np.sum((x_u2 - np.tan(x_l2)) ** 2)
        ul_objective = F1 + F2 + F3

        f2 = np.sum((x_l1 - 2) ** 2)
        f3 = np.sum((x_u2 - np.tan(x_l2)) ** 2)
        ll_objective = f2 + f3

        return ul_objective, ll_objective

    def evaluate_ul_constraints(self, ul_vars, ll_vars):
        x_u1, x_u2, _, _ = self._split_vars(ul_vars, ll_vars)
        constraints = []
        for j in range(self.p):
            sum_xu1_cubed = np.sum(np.delete(x_u1, j) ** 3)
            sum_xu2_cubed = np.sum(x_u2**3)
            G = x_u1[j] - sum_xu1_cubed - sum_xu2_cubed
            constraints.append(-G if G < 0 else 0)
        for j in range(self.r):
            sum_xu2_cubed = np.sum(np.delete(x_u2, j) ** 3)
            sum_xu1_cubed = np.sum(x_u1**3)
            G = x_u2[j] - sum_xu2_cubed - sum_xu1_cubed
            constraints.append(-G if G < 0 else 0)
        return np.array(constraints)

    def evaluate_ll_constraints(self, ul_vars, ll_vars):
        _, _, x_l1, _ = self._split_vars(ul_vars, ll_vars)
        constraints = []
        for j in range(self.q):
            sum_xl1_cubed = np.sum(np.delete(x_l1, j) ** 3)
            g = x_l1[j] - sum_xl1_cubed
            constraints.append(-g if g < 0 else 0)
        return np.array(constraints)


class SMD11(BilevelProblem):
    """Constrained, conflicting, functions of both x_u and x_l. Ref: Eq. (44-46)."""

    def __init__(self, p=1, q=2, r=1, s=0):
        super().__init__(p, q, r, s, "SMD11")

    def evaluate(self, ul_vars, ll_vars):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)

        F1 = np.sum(x_u1**2)
        F2 = -np.sum(x_l1**2)
        F3 = np.sum(x_u2**2) - np.sum((x_u2 - np.log(x_l2)) ** 2)
        ul_objective = F1 + F2 + F3

        f2 = np.sum(x_l1**2)
        f3 = np.sum((x_u2 - np.log(x_l2)) ** 2)
        ll_objective = f2 + f3

        return ul_objective, ll_objective

    def evaluate_ul_constraints(self, ul_vars, ll_vars):
        _, x_u2, _, x_l2 = self._split_vars(ul_vars, ll_vars)
        constraints = []
        for j in range(self.r):
            G = x_u2[j] - (1 / np.sqrt(self.r)) - np.log(x_l2[j])
            constraints.append(-G if G < 0 else 0)
        return np.array(constraints)

    def evaluate_ll_constraints(self, ul_vars, ll_vars):
        _, x_u2, _, x_l2 = self._split_vars(ul_vars, ll_vars)
        g1 = 1 - np.sum((x_u2 - np.log(x_l2)) ** 2)
        return np.array([-g1 if g1 < 0 else 0])


class SMD12(BilevelProblem):
    """Constrained, mixed difficulties. Ref: Eq. (48-50)."""

    def __init__(self, p=1, q=2, r=1, s=0):
        super().__init__(p, q, r, s, "SMD12")

    def evaluate(self, ul_vars, ll_vars):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)

        F1 = np.sum((x_u1 - 2) ** 2)
        F2 = np.sum(x_l1**2)
        F3 = np.sum((x_u2 - 2) ** 2) + np.sum(np.tan(np.abs(x_l2))) - np.sum((x_u2 - np.tan(x_l2)) ** 2)
        ul_objective = F1 + F2 + F3

        f2 = np.sum((x_l1 - 2) ** 2)
        f3 = np.sum((x_u2 - np.tan(x_l2)) ** 2)
        ll_objective = f2 + f3

        return ul_objective, ll_objective

    def evaluate_ul_constraints(self, ul_vars, ll_vars):
        x_u1, x_u2, _, x_l2 = self._split_vars(ul_vars, ll_vars)
        constraints = []
        # G_j for j in {1..r}
        for i in range(self.r):
            G = x_u2[i] - np.tan(x_l2[i])
            constraints.append(-G if G < 0 else 0)
        # G_j for j in {p}
        for j in range(self.p):
            sum_xu1_cubed = np.sum(np.delete(x_u1, j) ** 3)
            sum_xu2_cubed = np.sum(x_u2**3)
            G = x_u1[j] - sum_xu1_cubed - sum_xu2_cubed
            constraints.append(-G if G < 0 else 0)
        # G_j for j in {r}
        for j in range(self.r):
            sum_xu2_cubed = np.sum(np.delete(x_u2, j) ** 3)
            sum_xu1_cubed = np.sum(x_u1**3)
            G = x_u2[j] - sum_xu2_cubed - sum_xu1_cubed
            constraints.append(-G if G < 0 else 0)
        return np.array(constraints)

    def evaluate_ll_constraints(self, ul_vars, ll_vars):
        _, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        constraints = []
        # g1
        g = 1 - np.sum((x_u2 - np.tan(x_l2)) ** 2)
        constraints.append(-g if g < 0 else 0)
        # g_j for j in {q}
        for j in range(self.q):
            g = x_l1[j] - np.sum(np.delete(x_l1, j) ** 3)  # Paper has p here, assume typo for q
            constraints.append(-g if g < 0 else 0)
        return np.array(constraints)


def get_smd_problem(name: str, **kwargs):
    """Factory function to get an SMD problem instance."""
    problem_map = {
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
    }
    problem_class = problem_map.get(name.lower())
    if problem_class:
        return problem_class(**kwargs)
    raise ValueError(f"Problem '{name}' not found in SMD suite.")


# Example of how to use it
if __name__ == "__main__":
    print("--- Verifying smd_benchmark_suite_v1.0.py ---")

    # Get an instance of SMD1 with default dimensions
    smd1 = get_smd_problem("SMD1")
    print(f"Instantiated: {smd1}")

    # Get an instance of SMD10 with 10 variables (p=3, q=3, r=2)
    smd10 = get_smd_problem("SMD10", p=3, q=3, r=2)
    print(f"Instantiated: {smd10}")

    # Test evaluation of a problem (using zeros as example inputs)
    ul_vars_test = np.zeros(smd1.ul_dim)
    ll_vars_test = np.ones(smd1.ll_dim) * 0.1  # Avoid tan(0) issues if any
    ul_obj, ll_obj = smd1.evaluate(ul_vars_test, ll_vars_test)
    print(f"\nTesting SMD1 with dummy inputs:")
    print(f"  UL Objective: {ul_obj:.4f}")
    print(f"  LL Objective: {ll_obj:.4f}")

    # Test a constrained problem
    smd9 = get_smd_problem("smd9")
    ul_vars_c = np.ones(smd9.ul_dim) * 2
    ll_vars_c = np.ones(smd9.ll_dim) * 2
    ul_obj_c, ll_obj_c = smd9.evaluate(ul_vars_c, ll_vars_c)
    ul_con_v = smd9.evaluate_ul_constraints(ul_vars_c, ll_vars_c)
    ll_con_v = smd9.evaluate_ll_constraints(ul_vars_c, ll_vars_c)

    print(f"\nTesting SMD9 with dummy inputs:")
    print(f"  UL Objective: {ul_obj_c:.4f}, UL Constraint Violation: {ul_con_v}")
    print(f"  LL Objective: {ll_obj_c:.4f}, LL Constraint Violation: {ll_con_v}")

    print("\n--- Verification Complete ---")
