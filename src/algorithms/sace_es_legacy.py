#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:20:55 2025

@author: sanup
"""


# sace_project/src/algorithms/sace_es.py

import numpy as np
from .base_optimizer import BaseOptimizer
from ..surrogates.gaussian_process import GaussianProcessSurrogate
from ..problems.smd_suite import get_smd_problem


class SACE_ES(BaseOptimizer):
    """
    Implements the Surrogate-Assisted Co-evolutionary Evolutionary Strategy (SACE-ES).

    VERSION 4: The Constraint-Aware Model
    - This version uses a multi-surrogate approach to explicitly handle constraints.
    - One surrogate (M_y) models the mapping from leader to follower solution (x -> y_opt).
    - A second surrogate (M_c) models the lower-level constraint violation c(x, y_opt).
    - Infill selection and evolution are guided by constraint-domination principles.
    """

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self.ul_pop_size = self.config.get("ul_pop_size", 50)
        self.ll_pop_size = self.config.get("ll_pop_size", 30)
        self.generations = self.config.get("generations", 100)
        self.ll_generations = self.config.get("ll_generations", 20)
        self.initial_samples = self.config.get("initial_samples", 20)

        # Surrogates for the LL solution mapping
        self.surrogates_y = [GaussianProcessSurrogate() for _ in range(self.problem.ll_dim)]

        # Surrogates for the LL constraint functions
        self.has_ll_constraints = self.problem.num_ll_constraints > 0
        if self.has_ll_constraints:
            self.surrogates_c = [GaussianProcessSurrogate() for _ in range(self.problem.num_ll_constraints)]

        # Archives for training data
        self.archive_ul = []
        self.archive_ll_sols = []
        self.archive_ll_cons = []  # For constraint values

    def _run_lower_level_es(self, ul_individual):
        """
        Runs ES to find the follower's optimal response and returns the solution
        and its corresponding constraint violation values.
        """
        ul_ind_safe = np.atleast_1d(ul_individual)
        ll_pop = np.random.uniform(
            self.problem.ll_bounds[0],
            self.problem.ll_bounds[1],
            size=(self.ll_pop_size, self.problem.ll_dim),
        )

        fitness_ll = np.array(
            [self.problem.evaluate(ul_ind_safe, np.atleast_1d(p), add_penalty=True)[1] for p in ll_pop]
        )
        self.ll_nfe += self.ll_pop_size

        for _ in range(self.ll_generations):
            offspring_pop = ll_pop + np.random.normal(0, 0.2, size=ll_pop.shape)
            offspring_pop = np.clip(offspring_pop, self.problem.ll_bounds[0], self.problem.ll_bounds[1])
            offspring_fitness = np.array(
                [
                    self.problem.evaluate(ul_ind_safe, np.atleast_1d(p), add_penalty=True)[1]
                    for p in offspring_pop
                ]
            )
            self.ll_nfe += self.ll_pop_size

            combined_pop = np.vstack([ll_pop, offspring_pop])
            combined_fitness = np.concatenate([fitness_ll, offspring_fitness])
            best_indices = np.argsort(combined_fitness)[: self.ll_pop_size]
            ll_pop = combined_pop[best_indices]
            fitness_ll = combined_fitness[best_indices]

        best_ll_sol = ll_pop[0]

        ll_cons_vals = self.problem.evaluate_ll_constraints(ul_ind_safe, np.atleast_1d(best_ll_sol))

        return best_ll_sol, ll_cons_vals

    def _get_surrogate_predictions(self, ul_pop):
        """Gets predicted LL solutions and constraint violations from surrogates."""
        if not all(s.is_trained for s in self.surrogates_y):
            return None, None

        pred_ll_sols = np.array([s.predict(ul_pop)[0] for s in self.surrogates_y]).T

        pred_cons_violations = np.zeros((len(ul_pop), self.problem.num_ll_constraints))
        if self.has_ll_constraints and all(s.is_trained for s in self.surrogates_c):
            pred_cons_vals = np.array([s.predict(ul_pop)[0] for s in self.surrogates_c]).T
            pred_cons_violations = np.maximum(0, pred_cons_vals)

        return pred_ll_sols, np.sum(pred_cons_violations, axis=1)

    def _constrained_dominance_selection(self, combined_pop, pred_ul_fitness, pred_cons_violations):
        """Performs tournament selection using Deb's constraint handling rules."""
        new_pop = []
        for _ in range(self.ul_pop_size):
            p1_idx, p2_idx = np.random.choice(len(combined_pop), 2, replace=False)

            v1 = pred_cons_violations[p1_idx]
            v2 = pred_cons_violations[p2_idx]
            f1 = pred_ul_fitness[p1_idx]
            f2 = pred_ul_fitness[p2_idx]

            if v1 < 1e-6 and v2 < 1e-6:  # Both feasible
                winner_idx = p1_idx if f1 < f2 else p2_idx
            elif v1 < v2:  # p1 is less infeasible
                winner_idx = p1_idx
            else:  # p2 is less or equally infeasible
                winner_idx = p2_idx
            new_pop.append(combined_pop[winner_idx])
        return np.array(new_pop)

    def solve(self):
        # Phase 1: Initial Sampling
        print("Phase 1: Initial sampling for constraint-aware surrogate models (v4)...")
        initial_ul_samples = np.random.uniform(
            self.problem.ul_bounds[0],
            self.problem.ul_bounds[1],
            size=(self.initial_samples, self.problem.ul_dim),
        )

        for ul_sample in initial_ul_samples:
            ll_opt, ll_cons = self._run_lower_level_es(ul_sample)
            self.archive_ul.append(ul_sample)
            self.archive_ll_sols.append(ll_opt)
            if self.has_ll_constraints:
                self.archive_ll_cons.append(ll_cons)

        self.archive_ul = np.array(self.archive_ul)
        self.archive_ll_sols = np.array(self.archive_ll_sols)

        for i in range(self.problem.ll_dim):
            self.surrogates_y[i].train(self.archive_ul, self.archive_ll_sols[:, i])
        if self.has_ll_constraints:
            self.archive_ll_cons = np.array(self.archive_ll_cons)
            for i in range(self.problem.num_ll_constraints):
                self.surrogates_c[i].train(self.archive_ul, self.archive_ll_cons[:, i])
        print("Surrogate models trained.")

        # Phase 2: Main Evolutionary Loop
        print("Phase 2: Starting constraint-aware optimization (v4)...")
        ul_pop = np.random.uniform(
            self.problem.ul_bounds[0],
            self.problem.ul_bounds[1],
            size=(self.ul_pop_size, self.problem.ul_dim),
        )

        for gen in range(self.generations):
            offspring_ul_pop = ul_pop + np.random.normal(0, 0.5, size=ul_pop.shape)
            offspring_ul_pop = np.clip(
                offspring_ul_pop, self.problem.ul_bounds[0], self.problem.ul_bounds[1]
            )

            infill_idx = np.random.randint(0, len(offspring_ul_pop))
            promising_ul = offspring_ul_pop[infill_idx]

            exact_ll_sol, exact_ll_cons = self._run_lower_level_es(promising_ul)

            self.archive_ul = np.vstack([self.archive_ul, promising_ul])
            self.archive_ll_sols = np.vstack([self.archive_ll_sols, exact_ll_sol])
            for i in range(self.problem.ll_dim):
                self.surrogates_y[i].train(self.archive_ul, self.archive_ll_sols[:, i])
            if self.has_ll_constraints:
                self.archive_ll_cons = np.vstack([self.archive_ll_cons, exact_ll_cons])
                for i in range(self.problem.num_ll_constraints):
                    self.surrogates_c[i].train(self.archive_ul, self.archive_ll_cons[:, i])

            combined_pop = np.vstack([ul_pop, offspring_ul_pop])
            pred_ll_sols, pred_cons_violations = self._get_surrogate_predictions(combined_pop)
            pred_ul_fitness = np.array(
                [
                    self.problem.evaluate(
                        np.atleast_1d(combined_pop[i]), np.atleast_1d(pred_ll_sols[i]), add_penalty=False
                    )[0]
                    for i in range(len(combined_pop))
                ]
            )
            self.ul_nfe += len(combined_pop)

            ul_pop = self._constrained_dominance_selection(
                combined_pop, pred_ul_fitness, pred_cons_violations
            )

            archive_true_fitness_raw = np.array(
                [
                    self.problem.evaluate(self.archive_ul[i], self.archive_ll_sols[i], add_penalty=False)[0]
                    for i in range(len(self.archive_ul))
                ]
            )
            if self.has_ll_constraints:
                feasible_mask = np.all(self.archive_ll_cons <= 1e-6, axis=1)
                best_fitness_overall = (
                    np.min(archive_true_fitness_raw[feasible_mask]) if np.any(feasible_mask) else np.inf
                )
            else:
                best_fitness_overall = np.min(archive_true_fitness_raw)
            self.log_generation(gen, best_fitness_overall, 0)
            if gen % 10 == 0:
                # print(
                #     f"Gen {gen}: Best True Feasible Fitness = {best_fitness_overall:.4f}, Archive Size = {len(self.archive_ul)}"
                # )
                print(
                    f"Gen {gen}: Best UL Fitness = {best_fitness_overall:.4f}, Avg NFE (UL/LL) = {self.ul_nfe}/{self.ll_nfe}"
                )

        # Final result from archive
        archive_true_fitness_raw = np.array(
            [
                self.problem.evaluate(self.archive_ul[i], self.archive_ll_sols[i], add_penalty=False)[0]
                for i in range(len(self.archive_ul))
            ]
        )
        if self.has_ll_constraints:
            feasible_mask = np.all(self.archive_ll_cons <= 1e-6, axis=1)
            if not np.any(feasible_mask):
                best_archive_idx = np.argmin(np.sum(np.maximum(0, self.archive_ll_cons), axis=1))
            else:
                archive_true_fitness_raw[~feasible_mask] = np.inf
                best_archive_idx = np.argmin(archive_true_fitness_raw)
        else:
            best_archive_idx = np.argmin(archive_true_fitness_raw)

        best_ul_solution = self.archive_ul[best_archive_idx]
        final_ll_solution = self.archive_ll_sols[best_archive_idx]
        final_results = {
            "final_ul_fitness": self.problem.evaluate(
                np.atleast_1d(best_ul_solution), np.atleast_1d(final_ll_solution), add_penalty=False
            )[0],
            "total_ul_nfe": self.ul_nfe,
            "total_ll_nfe": self.ll_nfe,
            "best_ul_solution": best_ul_solution,
            "corresponding_ll_solution": final_ll_solution,
        }
        return final_results


# =============================================================================
# Test Module
# =============================================================================
if __name__ == "__main__":
    print("--- Verifying SACE-ES Algorithm Implementation ---")

    test_config = {
        "ul_pop_size": 20,
        "ll_pop_size": 15,
        "generations": 30,
        "ll_generations": 15,
        "initial_samples": 10,
    }

    smd1_problem = get_smd_problem("SMD1")

    print(f"\nProblem: {smd1_problem}")
    print(f"Config: {test_config}")

    try:
        optimizer = SACE_ES(smd1_problem, test_config)
        results = optimizer.solve()

        print("\n--- Optimization Finished ---")
        print(f"Final UL Fitness: {results['final_ul_fitness']:.4f}")
        print(f"Best UL Solution: {results['best_ul_solution']}")
        print(f"Corresponding LL Solution: {results['corresponding_ll_solution']}")
        print(f"Total UL NFE: {results['total_ul_nfe']}")
        print(f"Total LL NFE: {results['total_ll_nfe']}")
        print("\n--- Verification Complete ---")

    except Exception as e:
        print(f"\nAn error occurred during the test run: {e}")
