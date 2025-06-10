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
    Implements the proposed Surrogate-Assisted Co-evolutionary Evolutionary
    Strategy (SACE-ES) for bilevel optimization.
    """

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self.ul_pop_size = self.config.get("ul_pop_size", 50)
        self.ll_pop_size = self.config.get("ll_pop_size", 30)
        self.generations = self.config.get("generations", 100)
        self.ll_generations = self.config.get("ll_generations", 20)

        # Number of initial samples to build the first surrogate
        self.initial_samples = self.config.get("initial_samples", 20)

        # Surrogate model for the lower-level optimal solution mapping: x_u -> y_opt
        self.surrogate = GaussianProcessSurrogate()
        self.archive_ul = []  # To store UL individuals for training
        self.archive_ll = []  # To store corresponding LL optimal solutions

    def _run_lower_level_es(self, ul_individual):
        """
        Runs a simple ES to find the follower's optimal response.

        Args:
            ul_individual (np.array): The upper-level solution.

        Returns:
            np.array: The best found lower-level solution.
        """
        # Initialize population and strategy parameters (mutation strengths)
        ll_pop = np.random.uniform(
            self.problem.ll_bounds[0],
            self.problem.ll_bounds[1],
            size=(self.ll_pop_size, self.problem.ll_dim),
        )
        ll_sigmas = np.full((self.ll_pop_size, self.problem.ll_dim), 0.5)

        # Evaluate initial population
        fitness_ll = np.array([self.problem.evaluate(ul_individual, p)[1] for p in ll_pop])
        self.ll_nfe += self.ll_pop_size

        for _ in range(self.ll_generations):
            # Generate offspring
            offspring_pop = []
            offspring_sigmas = []
            for i in range(self.ll_pop_size):
                # Self-adaptive mutation
                tau = 1 / np.sqrt(2 * self.problem.ll_dim)
                tau_prime = 1 / np.sqrt(2 * np.sqrt(self.problem.ll_dim))
                new_sigma = ll_sigmas[i] * np.exp(
                    tau_prime * np.random.randn() + tau * np.random.randn(self.problem.ll_dim)
                )

                mutant = ll_pop[i] + new_sigma * np.random.randn(self.problem.ll_dim)
                mutant = np.clip(mutant, self.problem.ll_bounds[0], self.problem.ll_bounds[1])
                offspring_pop.append(mutant)
                offspring_sigmas.append(new_sigma)

            offspring_pop = np.array(offspring_pop)
            offspring_sigmas = np.array(offspring_sigmas)

            # Evaluate offspring
            offspring_fitness = np.array([self.problem.evaluate(ul_individual, p)[1] for p in offspring_pop])
            self.ll_nfe += self.ll_pop_size

            # (mu + lambda) selection
            combined_pop = np.vstack([ll_pop, offspring_pop])
            combined_sigmas = np.vstack([ll_sigmas, offspring_sigmas])
            combined_fitness = np.concatenate([fitness_ll, offspring_fitness])

            best_indices = np.argsort(combined_fitness)[: self.ll_pop_size]
            ll_pop = combined_pop[best_indices]
            ll_sigmas = combined_sigmas[best_indices]
            fitness_ll = combined_fitness[best_indices]

        return ll_pop[0]  # Return the best individual

    def solve(self):
        # 1. Initial Sampling to build the first surrogate model
        print("Phase 1: Initial sampling for surrogate model...")
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
        self.surrogate.train(self.archive_ul, self.archive_ll)
        print("Surrogate model trained on initial samples.")

        # 2. Initialize UL population and evaluate using the surrogate
        print("Phase 2: Starting co-evolutionary optimization...")
        ul_pop = np.random.uniform(
            self.problem.ul_bounds[0],
            self.problem.ul_bounds[1],
            size=(self.ul_pop_size, self.problem.ul_dim),
        )
        ul_sigmas = np.full((self.ul_pop_size, self.problem.ul_dim), 0.5)

        ll_solutions_pred, _ = self.surrogate.predict(ul_pop)
        fitness_ul = np.array(
            [self.problem.evaluate(ul_pop[i], ll_solutions_pred[i])[0] for i in range(self.ul_pop_size)]
        )
        self.ul_nfe += self.ul_pop_size  # We count surrogate evals as UL NFE

        best_fitness_overall = np.min(fitness_ul)

        # 3. Main Co-evolutionary Loop
        for gen in range(self.generations):
            # Generate UL offspring
            offspring_ul_pop = []
            for i in range(self.ul_pop_size):
                tau = 1.0 / np.sqrt(2 * self.problem.ul_dim)
                mutant_sigma = ul_sigmas[i] * np.exp(tau * np.random.randn(self.problem.ul_dim))
                mutant = ul_pop[i] + mutant_sigma * np.random.randn(self.problem.ul_dim)
                mutant = np.clip(mutant, self.problem.ul_bounds[0], self.problem.ul_bounds[1])
                offspring_ul_pop.append(mutant)
            offspring_ul_pop = np.array(offspring_ul_pop)

            # Use surrogate to get predicted LL solutions and fitness for offspring
            offspring_ll_sols_pred, _ = self.surrogate.predict(offspring_ul_pop)
            offspring_fitness_ul = np.array(
                [
                    self.problem.evaluate(offspring_ul_pop[i], offspring_ll_sols_pred[i])[0]
                    for i in range(self.ul_pop_size)
                ]
            )
            self.ul_nfe += self.ul_pop_size

            # Model Management: Choose one individual to re-evaluate exactly
            # We use a simple strategy: pick the most promising offspring
            most_promising_idx = np.argmin(offspring_fitness_ul)
            promising_ul = offspring_ul_pop[most_promising_idx]

            # Perform expensive evaluation
            exact_ll_sol = self._run_lower_level_es(promising_ul)

            # Update archives and retrain surrogate
            self.archive_ul = np.vstack([self.archive_ul, promising_ul])
            self.archive_ll = np.vstack([self.archive_ll, exact_ll_sol])
            self.surrogate.train(self.archive_ul, self.archive_ll)

            # Update the fitness of the re-evaluated individual
            offspring_fitness_ul[most_promising_idx] = self.problem.evaluate(promising_ul, exact_ll_sol)[0]

            # (mu + lambda) selection for UL
            combined_pop_ul = np.vstack([ul_pop, offspring_ul_pop])
            combined_fitness_ul = np.concatenate([fitness_ul, offspring_fitness_ul])
            best_indices_ul = np.argsort(combined_fitness_ul)[: self.ul_pop_size]

            ul_pop = combined_pop_ul[best_indices_ul]
            fitness_ul = combined_fitness_ul[best_indices_ul]

            best_fitness_overall = np.min(fitness_ul)
            self.log_generation(gen, best_fitness_overall, np.mean(fitness_ul))

            if gen % 10 == 0:
                print(
                    f"Gen {gen}: Best UL Fitness = {best_fitness_overall:.4f}, NFE (UL/LL) = {self.ul_nfe}/{self.ll_nfe}"
                )

        # Final result
        best_idx = np.argmin(fitness_ul)
        best_ul_solution = ul_pop[best_idx]
        # Get the final, most accurate LL solution for the best UL one
        final_ll_solution = self._run_lower_level_es(best_ul_solution)

        final_results = {
            "final_ul_fitness": self.problem.evaluate(best_ul_solution, final_ll_solution)[0],
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
