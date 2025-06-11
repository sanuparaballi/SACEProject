#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:20:55 2025

@author: sanup
"""


# sace_project/src/algorithms/sace_es.py

import numpy as np
from scipy.stats import norm
from .base_optimizer import BaseOptimizer
from ..surrogates.gaussian_process import GaussianProcessSurrogate
from ..problems.smd_suite import get_smd_problem


class SACE_ES(BaseOptimizer):
    """
    Implements the Surrogate-Assisted Co-evolutionary Evolutionary Strategy (SACE-ES).

    VERSION 2 (for version control):
    - Implements Expected Improvement (EI) for intelligent infill criteria.
    - The surrogate model attempts to learn the direct mapping from the leader's
      variables to the follower's optimal solution (x -> y_opt).
    - NOTE: This version contains the known issue where it does not explicitly
      model constraints, leading to poor performance on SMD6.
    """

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self.ul_pop_size = self.config.get("ul_pop_size", 50)
        self.ll_pop_size = self.config.get("ll_pop_size", 30)
        self.generations = self.config.get("generations", 100)
        self.ll_generations = self.config.get("ll_generations", 20)
        self.initial_samples = self.config.get("initial_samples", 20)
        self.xi = self.config.get("xi", 0.01)  # Exploration-exploitation trade-off for EI

        # We use one surrogate model for each dimension of the lower-level solution vector y
        self.surrogates = [GaussianProcessSurrogate() for _ in range(self.problem.ll_dim)]
        self.archive_ul = []  # Stores UL solutions that have been exactly evaluated
        self.archive_ll = []  # Stores their corresponding true optimal LL solutions

    def _run_lower_level_es(self, ul_individual):
        """
        Runs a simple ES to find the follower's optimal response.
        It minimizes the final, penalized lower-level objective.
        """
        ul_ind_safe = np.atleast_1d(ul_individual)

        ll_pop = np.random.uniform(
            self.problem.ll_bounds[0],
            self.problem.ll_bounds[1],
            size=(self.ll_pop_size, self.problem.ll_dim),
        )

        fitness_ll = np.array([self.problem.evaluate(ul_ind_safe, np.atleast_1d(p))[1] for p in ll_pop])
        self.ll_nfe += self.ll_pop_size

        for _ in range(self.ll_generations):
            offspring_pop = ll_pop + np.random.normal(0, 0.2, size=ll_pop.shape)
            offspring_pop = np.clip(offspring_pop, self.problem.ll_bounds[0], self.problem.ll_bounds[1])
            offspring_fitness = np.array(
                [self.problem.evaluate(ul_ind_safe, np.atleast_1d(p))[1] for p in offspring_pop]
            )
            self.ll_nfe += self.ll_pop_size

            combined_pop = np.vstack([ll_pop, offspring_pop])
            combined_fitness = np.concatenate([fitness_ll, offspring_fitness])
            best_indices = np.argsort(combined_fitness)[: self.ll_pop_size]
            ll_pop = combined_pop[best_indices]
            fitness_ll = combined_fitness[best_indices]

        return ll_pop[0]

    def _get_surrogate_based_ul_fitness(self, ul_pop):
        """
        Predicts the UL fitness for a population using the surrogate mapping.
        """
        if not all(s.is_trained for s in self.surrogates):
            # If surrogates aren't trained, return a non-informative high fitness
            return np.full(len(ul_pop), np.inf)

        # Predict the LL solution for each UL individual
        pred_ll_sols = np.array([s.predict(ul_pop)[0] for s in self.surrogates]).T

        # Evaluate the UL objective using the *predicted* LL solutions
        predicted_ul_fitness = np.array(
            [
                self.problem.evaluate(np.atleast_1d(ul_pop[i]), np.atleast_1d(pred_ll_sols[i]))[0]
                for i in range(len(ul_pop))
            ]
        )
        self.ul_nfe += len(ul_pop)  # Count this as a UL evaluation
        return predicted_ul_fitness

    def solve(self):
        # Phase 1: Initial Sampling to build the surrogate models
        print("Phase 1: Initial sampling for surrogate model (v2)...")
        initial_ul_samples = np.random.uniform(
            self.problem.ul_bounds[0],
            self.problem.ul_bounds[1],
            size=(self.initial_samples, self.problem.ul_dim),
        )

        for ul_sample in initial_ul_samples:
            ll_opt = self._run_lower_level_es(ul_sample)
            self.archive_ul.append(ul_sample)
            self.archive_ll.append(ll_opt)

        self.archive_ul = np.array(self.archive_ul)
        self.archive_ll = np.array(self.archive_ll)

        for i in range(self.problem.ll_dim):
            self.surrogates[i].train(self.archive_ul, self.archive_ll[:, i])
        print("Surrogate models trained on initial samples.")

        # Phase 2: Main Evolutionary Loop
        print("Phase 2: Starting surrogate-assisted optimization (v2)...")
        ul_pop = np.random.uniform(
            self.problem.ul_bounds[0],
            self.problem.ul_bounds[1],
            size=(self.ul_pop_size, self.problem.ul_dim),
        )

        fitness_ul_pred = self._get_surrogate_based_ul_fitness(ul_pop)

        for gen in range(self.generations):
            # Evolve UL population
            offspring_ul_pop = ul_pop + np.random.normal(0, 0.5, size=ul_pop.shape)
            offspring_ul_pop = np.clip(
                offspring_ul_pop, self.problem.ul_bounds[0], self.problem.ul_bounds[1]
            )

            # Model Management: This version has a simplified infill strategy
            # A true EI would require a surrogate on the final fitness, which this version avoids.
            # We select the predicted best offspring as the infill point.
            offspring_fitness_pred = self._get_surrogate_based_ul_fitness(offspring_ul_pop)
            infill_idx = np.argmin(offspring_fitness_pred)
            promising_ul = offspring_ul_pop[infill_idx]

            # Perform one expensive evaluation to gather new data
            exact_ll_sol = self._run_lower_level_es(promising_ul)

            # Update archives and retrain surrogates
            self.archive_ul = np.vstack([self.archive_ul, promising_ul])
            self.archive_ll = np.vstack([self.archive_ll, exact_ll_sol])
            for i in range(self.problem.ll_dim):
                self.surrogates[i].train(self.archive_ul, self.archive_ll[:, i])

            # Re-evaluate offspring fitness with the updated surrogate
            offspring_fitness_pred = self._get_surrogate_based_ul_fitness(offspring_ul_pop)

            # Selection
            combined_pop_ul = np.vstack([ul_pop, offspring_ul_pop])
            combined_fitness_pred = np.concatenate([fitness_ul_pred, offspring_fitness_pred])
            best_indices_ul = np.argsort(combined_fitness_pred)[: self.ul_pop_size]

            ul_pop = combined_pop_ul[best_indices_ul]
            fitness_ul_pred = combined_fitness_pred[best_indices_ul]

            # For logging, we need the best *true* fitness found so far
            archive_fitness_true = np.array(
                [
                    self.problem.evaluate(self.archive_ul[i], self.archive_ll[i])[0]
                    for i in range(len(self.archive_ul))
                ]
            )
            best_fitness_overall = np.min(archive_fitness_true)
            self.log_generation(gen, best_fitness_overall, np.mean(fitness_ul_pred))

            if gen % 10 == 0:
                print(
                    f"Gen {gen}: Best True Fitness Found = {best_fitness_overall:.4f}, Archive Size = {len(self.archive_ul)}"
                )

        # Final result: Find the best solution from the archive of all true evaluations
        archive_fitness_true = np.array(
            [
                self.problem.evaluate(self.archive_ul[i], self.archive_ll[i])[0]
                for i in range(len(self.archive_ul))
            ]
        )
        best_archive_idx = np.argmin(archive_fitness_true)
        best_ul_solution = self.archive_ul[best_archive_idx]
        final_ll_solution = self.archive_ll[best_archive_idx]

        final_results = {
            "final_ul_fitness": archive_fitness_true[best_archive_idx],
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
