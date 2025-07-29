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
    Base class for the Sinha, Malo, Deb (SMD) bilevel optimization problem suite.
    This implementation is fully updated to be compatible with all algorithms.
    - Constructor now sets ul_bounds and ll_bounds.
    - evaluate() now accepts an 'add_penalty' flag.
    - Added evaluate_ll_gradient() method.
    """

    def __init__(self, p, q, r, s, name, ul_bounds, ll_bounds):
        self.p = p
        self.q = q
        self.r = r
        self.s = s
        self.name = name
        self.ul_dim = p + r
        self.ll_dim = q + s + r
        self.ul_bounds = np.array(ul_bounds)
        self.ll_bounds = np.array(ll_bounds)
        self.num_ul_constraints = 0
        self.num_ll_constraints = 0

    def _split_vars(self, ul_vars, ll_vars):
        x_u1 = ul_vars[: self.p]
        x_u2 = ul_vars[self.p :]
        x_l1 = ll_vars[: self.q + self.s]
        x_l2 = ll_vars[self.q + self.s :]
        return x_u1, x_u2, x_l1, x_l2

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        raise NotImplementedError("evaluate must be implemented.")

    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        raise NotImplementedError(f"LL gradient not implemented for {self.name}.")

    def evaluate_ul_constraints(self, ul_vars, ll_vars):
        return np.array([])

    def evaluate_ll_constraints(self, ul_vars, ll_vars):
        return np.array([])

    def __repr__(self):
        return f"{self.name}(UL={self.ul_dim}, LL={self.ll_dim})"


class SMD1(BilevelProblem):
    def __init__(self, p=1, q=2, r=1, s=0):
        ul_b = [[-5, 10]] * (p + r)
        ll_b = [[-5, 10]] * q + [[-1.57, 1.57]] * r
        super().__init__(p, q, r, s, "SMD1", ul_b, ll_b)

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        term_f3 = (x_u2 - np.tan(x_l2)) ** 2
        ul_objective = np.sum(x_u1**2) + np.sum(x_l1**2) + np.sum(x_u2**2) + np.sum(term_f3)
        ll_objective = np.sum(x_l1**2) + np.sum(term_f3)
        return ul_objective, ll_objective

    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        _, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        grad_l1 = 2 * x_l1
        grad_l2 = 2 * (np.tan(x_l2) - x_u2) * (1 / np.cos(x_l2) ** 2)
        return np.concatenate([grad_l1, grad_l2])


class SMD2(BilevelProblem):
    def __init__(self, p=1, q=2, r=1, s=0):
        ul_b = [[-5, 10]] * p + [[-5, 1]] * r
        ll_b = [[-5, 10]] * q + [[1e-6, np.e]] * r
        super().__init__(p, q, r, s, "SMD2", ul_b, ll_b)

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        term_f3 = (x_u2 - np.log(x_l2)) ** 2
        ul_objective = np.sum(x_u1**2) - np.sum(x_l1**2) + np.sum(x_u2**2) - np.sum(term_f3)
        ll_objective = np.sum(x_l1**2) + np.sum(term_f3)
        return ul_objective, ll_objective

    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        _, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        grad_l1 = 2 * x_l1
        grad_l2 = 2 * (np.log(x_l2) - x_u2) / x_l2
        return np.concatenate([grad_l1, grad_l2])


class SMD3(BilevelProblem):
    def __init__(self, p=1, q=2, r=1, s=0):
        ul_b = [[-5, 10]] * (p + r)
        ll_b = [[-5, 10]] * q + [[-1.57, 1.57]] * r
        super().__init__(p, q, r, s, "SMD3", ul_b, ll_b)

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        term_f3 = (x_u2**2 - np.tan(x_l2)) ** 2
        ul_objective = np.sum(x_u1**2) + np.sum(x_l1**2) + np.sum(x_u2**2) + np.sum(term_f3)
        ll_objective = self.q + np.sum(x_l1**2 - np.cos(2 * np.pi * x_l1)) + np.sum(term_f3)
        return ul_objective, ll_objective

    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        _, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        grad_l1 = 2 * x_l1 + 2 * np.pi * np.sin(2 * np.pi * x_l1)
        grad_l2 = 2 * (np.tan(x_l2) - x_u2**2) * (1 / np.cos(x_l2) ** 2)
        return np.concatenate([grad_l1, grad_l2])


class SMD4(BilevelProblem):
    def __init__(self, p=1, q=2, r=1, s=0):
        ul_b = [[-5, 10]] * p + [[-1, 1]] * r
        ll_b = [[-5, 10]] * q + [[0, np.e]] * r
        super().__init__(p, q, r, s, "SMD4", ul_b, ll_b)

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        term_f3 = (np.abs(x_u2) - np.log(1 + x_l2)) ** 2
        ul_objective = np.sum(x_u1**2) - np.sum(x_l1**2) + np.sum(x_u2**2) - np.sum(term_f3)
        ll_objective = self.q + np.sum(x_l1**2 - np.cos(2 * np.pi * x_l1)) + np.sum(term_f3)
        return ul_objective, ll_objective

    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        _, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        grad_l1 = 2 * x_l1 + 2 * np.pi * np.sin(2 * np.pi * x_l1)
        grad_l2 = 2 * (np.log(1 + x_l2) - np.abs(x_u2)) / (1 + x_l2)
        return np.concatenate([grad_l1, grad_l2])


class SMD5(BilevelProblem):
    def __init__(self, p=1, q=2, r=1, s=0):
        ul_b = [[-5, 10]] * (p + r)
        ll_b = [[-5, 10]] * (q + r)
        super().__init__(p, q, r, s, "SMD5", ul_b, ll_b)
        if q < 2:
            raise ValueError("SMD5 requires q >= 2 for Rosenbrock.")

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        rosenbrock = np.sum(100 * (x_l1[1:] - x_l1[:-1] ** 2) ** 2 + (x_l1[:-1] - 1) ** 2)
        term_f3 = (np.abs(x_u2) - x_l2**2) ** 2
        ul_objective = np.sum(x_u1**2) - rosenbrock + np.sum(x_u2**2) - np.sum(term_f3)
        ll_objective = rosenbrock + np.sum(term_f3)
        return ul_objective, ll_objective

    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        _, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        grad_l1 = np.zeros_like(x_l1)
        grad_l1[:-1] = -400 * (x_l1[1:] - x_l1[:-1] ** 2) * x_l1[:-1] + 2 * (x_l1[:-1] - 1)
        grad_l1[1:] += 200 * (x_l1[1:] - x_l1[:-1] ** 2)
        grad_l2 = 4 * (x_l2**2 - np.abs(x_u2)) * x_l2
        return np.concatenate([grad_l1, grad_l2])


class SMD6(BilevelProblem):
    def __init__(self, p=1, q=0, r=1, s=2):
        ul_b = [[-5, 10]] * (p + r)
        ll_b = [[-5, 10]] * (q + s + r)
        super().__init__(p, q, r, s, "SMD6", ul_b, ll_b)
        if s < 2 or s % 2 != 0:
            raise ValueError("SMD6 requires s to be an even number >= 2.")

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        f2_term2 = np.sum([(x_l1[i + 1] - x_l1[i]) ** 2 for i in range(self.q, self.q + self.s - 1, 2)])
        term_f3 = (x_u2 - x_l2) ** 2
        ul_objective = (
            np.sum(x_u1**2)
            - np.sum(x_l1[: self.q] ** 2)
            + np.sum(x_l1[self.q :] ** 2)
            + np.sum(x_u2**2)
            - np.sum(term_f3)
        )
        ll_objective = np.sum(x_l1[: self.q] ** 2) + f2_term2 + np.sum(term_f3)
        return ul_objective, ll_objective

    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        _, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        grad_l1 = np.zeros_like(x_l1)
        grad_l1[: self.q] = 2 * x_l1[: self.q]
        for i in range(self.q, self.q + self.s - 1, 2):
            diff = x_l1[i + 1] - x_l1[i]
            grad_l1[i] -= 2 * diff
            grad_l1[i + 1] += 2 * diff
        grad_l2 = 2 * (x_l2 - x_u2)
        return np.concatenate([grad_l1, grad_l2])


class SMD7(BilevelProblem):
    def __init__(self, p=1, q=2, r=1, s=0):
        ul_b = [[-5, 10]] * p + [[-5, 1]] * r
        ll_b = [[-5, 10]] * q + [[1e-6, np.e]] * r
        super().__init__(p, q, r, s, "SMD7", ul_b, ll_b)

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        cos_prod = np.prod([np.cos(x_u1[i] / np.sqrt(i + 1)) for i in range(self.p)])
        F1 = 1 + (1 / 4000) * np.sum(x_u1**2) - cos_prod
        term_f3 = (x_u2 - np.log(x_l2)) ** 2
        ul_objective = F1 - np.sum(x_l1**2) + np.sum(x_u2**2) - np.sum(term_f3)
        ll_objective = np.sum(x_l1**2) + np.sum(term_f3)
        return ul_objective, ll_objective

    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        _, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        grad_l1 = 2 * x_l1
        grad_l2 = 2 * (np.log(x_l2) - x_u2) / x_l2
        return np.concatenate([grad_l1, grad_l2])


class SMD8(BilevelProblem):
    def __init__(self, p=1, q=2, r=1, s=0):
        ul_b = [[-5, 10]] * (p + r)
        ll_b = [[-5, 10]] * (q + r)
        super().__init__(p, q, r, s, "SMD8", ul_b, ll_b)
        if q < 2:
            raise ValueError("SMD8 requires q >= 2 for Rosenbrock.")

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        rosenbrock = np.sum(100 * (x_l1[1:] - x_l1[:-1] ** 2) ** 2 + (x_l1[:-1] - 1) ** 2)
        term_f3 = (x_u2 - x_l2**3) ** 2
        term1_F1 = -0.2 * np.sqrt((1 / self.p) * np.sum(x_u1**2))
        term2_F1 = (1 / self.p) * np.sum(np.cos(2 * np.pi * x_u1))
        F1 = 20 + np.e - 20 * np.exp(term1_F1) - np.exp(term2_F1)
        ul_objective = F1 - rosenbrock + np.sum(x_u2**2) - np.sum(term_f3)
        ll_objective = rosenbrock + np.sum(term_f3)
        return ul_objective, ll_objective

    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        _, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        grad_l1 = np.zeros_like(x_l1)
        grad_l1[:-1] = -400 * (x_l1[1:] - x_l1[:-1] ** 2) * x_l1[:-1] + 2 * (x_l1[:-1] - 1)
        grad_l1[1:] += 200 * (x_l1[1:] - x_l1[:-1] ** 2)
        grad_l2 = 6 * (x_l2**3 - x_u2) * x_l2**2
        return np.concatenate([grad_l1, grad_l2])


class SMD9(BilevelProblem):
    def __init__(self, p=1, q=2, r=1, s=0):
        ul_b = [[-5, 10]] * p + [[-5, 1]] * r
        ll_b = [[-5, 10]] * q + [[-1 + 1e-6, -1 + np.e]] * r
        super().__init__(p, q, r, s, "SMD9", ul_b, ll_b)
        self.num_ul_constraints = 1
        self.num_ll_constraints = 1
        self.a = 1.0
        self.b = 1.0

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        term_f3 = (x_u2 - np.log(1 + x_l2)) ** 2
        ul_objective = np.sum(x_u1**2) - np.sum(x_l1**2) + np.sum(x_u2**2) - np.sum(term_f3)
        ll_objective = np.sum(x_l1**2) + np.sum(term_f3)
        return ul_objective, ll_objective

    def evaluate_ul_constraints(self, ul_vars, ll_vars):
        term = np.sum(ul_vars**2) / self.a
        G1 = term - np.floor(term + 0.5 / self.b)
        return np.array([G1])

    def evaluate_ll_constraints(self, ul_vars, ll_vars):
        term = np.sum(ll_vars**2) / self.a
        g1 = term - np.floor(term + 0.5 / self.b)
        return np.array([g1])

    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        _, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        grad_l1 = 2 * x_l1
        grad_l2 = 2 * (np.log(1 + x_l2) - x_u2) / (1 + x_l2)
        return np.concatenate([grad_l1, grad_l2])


class SMD10(BilevelProblem):
    def __init__(self, p=1, q=2, r=1, s=0):
        ul_b = [[-5, 10]] * (p + r)
        ll_b = [[-5, 10]] * q + [[-1.57, 1.57]] * r
        super().__init__(p, q, r, s, "SMD10", ul_b, ll_b)
        self.num_ul_constraints = p + r
        self.num_ll_constraints = q

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        term_f3 = (x_u2 - np.tan(x_l2)) ** 2
        ul_objective = np.sum((x_u1 - 2) ** 2) + np.sum(x_l1**2) + np.sum((x_u2 - 2) ** 2) - np.sum(term_f3)
        ll_objective = np.sum((x_l1 - 2) ** 2) + np.sum(term_f3)
        return ul_objective, ll_objective

    def evaluate_ul_constraints(self, ul_vars, ll_vars):
        x_u1, x_u2, _, _ = self._split_vars(ul_vars, ll_vars)
        constraints = []
        for j in range(self.p):
            G = x_u1[j] - np.sum(np.delete(x_u1, j) ** 3) - np.sum(x_u2**3)
            constraints.append(G)
        for j in range(self.r):
            G = x_u2[j] - np.sum(np.delete(x_u2, j) ** 3) - np.sum(x_u1**3)
            constraints.append(G)
        return np.array(constraints)

    def evaluate_ll_constraints(self, ul_vars, ll_vars):
        _, _, x_l1, _ = self._split_vars(ul_vars, ll_vars)
        constraints = [x_l1[j] - np.sum(np.delete(x_l1, j) ** 3) for j in range(self.q)]
        return np.array(constraints)

    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        _, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        grad_l1 = 2 * (x_l1 - 2)
        grad_l2 = 2 * (np.tan(x_l2) - x_u2) * (1 / np.cos(x_l2) ** 2)
        return np.concatenate([grad_l1, grad_l2])


class SMD11(BilevelProblem):
    def __init__(self, p=1, q=2, r=1, s=0):
        ul_b = [[-5, 10]] * p + [[-1, 1]] * r
        ll_b = [[-5, 10]] * q + [[1 / np.e, np.e]] * r
        super().__init__(p, q, r, s, "SMD11", ul_b, ll_b)
        self.num_ul_constraints = r
        self.num_ll_constraints = 1

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        term_f3 = (x_u2 - np.log(x_l2)) ** 2
        ul_objective = np.sum(x_u1**2) - np.sum(x_l1**2) + np.sum(x_u2**2) - np.sum(term_f3)
        ll_objective = np.sum(x_l1**2) + np.sum(term_f3)
        return ul_objective, ll_objective

    def evaluate_ul_constraints(self, ul_vars, ll_vars):
        _, x_u2, _, x_l2 = self._split_vars(ul_vars, ll_vars)
        return x_u2 - (1 / np.sqrt(self.r)) - np.log(x_l2)

    def evaluate_ll_constraints(self, ul_vars, ll_vars):
        _, x_u2, _, x_l2 = self._split_vars(ul_vars, ll_vars)
        g1 = np.sum((x_u2 - np.log(x_l2)) ** 2) - 1
        return np.array([g1])

    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        _, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        grad_l1 = 2 * x_l1
        grad_l2 = 2 * (np.log(x_l2) - x_u2) / x_l2
        return np.concatenate([grad_l1, grad_l2])


class SMD12(BilevelProblem):
    def __init__(self, p=1, q=2, r=1, s=0):
        ul_b = [[-5, 10]] * p + [[-14.1, 14.1]] * r
        ll_b = [[-5, 10]] * q + [[-1.5, 1.5]] * r
        super().__init__(p, q, r, s, "SMD12", ul_b, ll_b)
        self.num_ul_constraints = p + 2 * r
        self.num_ll_constraints = q + 1

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        x_u1, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        term_f3 = (x_u2 - np.tan(x_l2)) ** 2
        ul_objective = (
            np.sum((x_u1 - 2) ** 2)
            + np.sum(x_l1**2)
            + np.sum((x_u2 - 2) ** 2)
            + np.sum(np.tan(np.abs(x_l2)))
            - np.sum(term_f3)
        )
        ll_objective = np.sum((x_l1 - 2) ** 2) + np.sum(term_f3)
        return ul_objective, ll_objective

    def evaluate_ul_constraints(self, ul_vars, ll_vars):
        x_u1, x_u2, _, x_l2 = self._split_vars(ul_vars, ll_vars)
        g1 = x_u2 - np.tan(x_l2)
        g2 = [x_u1[j] - np.sum(np.delete(x_u1, j) ** 3) - np.sum(x_u2**3) for j in range(self.p)]
        g3 = [x_u2[j] - np.sum(np.delete(x_u2, j) ** 3) - np.sum(x_u1**3) for j in range(self.r)]
        return np.concatenate([g1, g2, g3])

    def evaluate_ll_constraints(self, ul_vars, ll_vars):
        _, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        g1 = np.sum((x_u2 - np.tan(x_l2)) ** 2) - 1
        g2 = [x_l1[j] - np.sum(np.delete(x_l1, j) ** 3) for j in range(self.q)]
        return np.concatenate([[g1], g2])

    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        _, x_u2, x_l1, x_l2 = self._split_vars(ul_vars, ll_vars)
        grad_l1 = 2 * (x_l1 - 2)
        grad_l2 = 2 * (np.tan(x_l2) - x_u2) * (1 / np.cos(x_l2) ** 2)
        return np.concatenate([grad_l1, grad_l2])


def get_smd_problem(name: str, params: dict):
    """Factory function to get an SMD problem instance by its name."""
    smd_map = {
        "SMD1": SMD1,
        "SMD2": SMD2,
        "SMD3": SMD3,
        "SMD4": SMD4,
        "SMD5": SMD5,
        "SMD6": SMD6,
        "SMD7": SMD7,
        "SMD8": SMD8,
        "SMD9": SMD9,
        "SMD10": SMD10,
        "SMD11": SMD11,
        "SMD12": SMD12,
    }
    problem_class = smd_map.get(name.upper())
    if not problem_class:
        raise ValueError(f"Problem '{name}' not found in the SMD suite.")
    return problem_class(**params)


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
    smd9 = get_smd_problem("SMD9")
    ul_vars_c = np.ones(smd9.ul_dim) * 2
    ll_vars_c = np.ones(smd9.ll_dim) * 2
    ul_obj_c, ll_obj_c = smd9.evaluate(ul_vars_c, ll_vars_c)
    ul_con_v = smd9.evaluate_ul_constraints(ul_vars_c, ll_vars_c)
    ll_con_v = smd9.evaluate_ll_constraints(ul_vars_c, ll_vars_c)

    print(f"\nTesting SMD9 with dummy inputs:")
    print(f"  UL Objective: {ul_obj_c:.4f}, UL Constraint Violation: {ul_con_v}")
    print(f"  LL Objective: {ll_obj_c:.4f}, LL Constraint Violation: {ll_con_v}")

    print("\n--- Verification Complete ---")
