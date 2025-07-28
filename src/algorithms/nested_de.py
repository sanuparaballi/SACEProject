#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:03:25 2025

@author: sanup
"""

# sace_project/src/algorithms/nested_de.py

# import numpy as np
# from .base_optimizer import BaseOptimizer

# To test this file standalone, we will need to import a problem
from ..problems.smd_suite import get_smd_problem


# class NestedDE(BaseOptimizer):
#     """
#     Implements a classic Nested Differential Evolution (DE) algorithm for
#     bilevel optimization.

#     In this approach, for each individual in the upper-level population, a full
#     DE optimization is run on the lower-level problem to find its optimal response.
#     """

#     def __init__(self, problem, config):
#         super().__init__(problem, config)
#         # DE specific parameters from config
#         self.ul_pop_size = self.config.get("ul_pop_size", 50)
#         self.ll_pop_size = self.config.get("ll_pop_size", 30)
#         self.ul_generations = self.config.get("generations", 100)
#         self.ll_generations = self.config.get("ll_generations", 50)  # Generations for the inner loop
#         self.F = self.config.get("F", 0.5)  # Mutation factor
#         self.Cr = self.config.get("Cr", 0.9)  # Crossover rate

#     def _run_lower_level_de(self, ul_individual):
#         """
#         Runs a full DE optimization for the lower-level problem.

#         Args:
#             ul_individual (np.array): The upper-level solution for which we need to
#                                       find the follower's optimal response.

#         Returns:
#             np.array: The best found lower-level solution.
#         """
#         # 1. Initialize LL population
#         ll_pop = np.random.uniform(
#             self.problem.ll_bounds[0],
#             self.problem.ll_bounds[1],
#             size=(self.ll_pop_size, self.problem.ll_dim),
#         )

#         # 2. Evaluate initial LL population
#         fitness_ll = np.array([self.problem.evaluate(ul_individual, ll_ind)[1] for ll_ind in ll_pop])
#         self.ll_nfe += self.ll_pop_size

#         # 3. LL DE evolution loop
#         for _ in range(self.ll_generations):
#             for i in range(self.ll_pop_size):
#                 # a. Mutation
#                 idxs = [idx for idx in range(self.ll_pop_size) if idx != i]
#                 a, b, c = ll_pop[np.random.choice(idxs, 3, replace=False)]
#                 mutant = a + self.F * (b - c)
#                 mutant = np.clip(mutant, self.problem.ll_bounds[0], self.problem.ll_bounds[1])

#                 # b. Crossover
#                 cross_points = np.random.rand(self.problem.ll_dim) < self.Cr
#                 # Ensure at least one gene is from the mutant
#                 if not np.any(cross_points):
#                     cross_points[np.random.randint(0, self.problem.ll_dim)] = True
#                 trial = np.where(cross_points, mutant, ll_pop[i])

#                 # c. Selection
#                 trial_fitness = self.problem.evaluate(ul_individual, trial)[1]
#                 self.ll_nfe += 1
#                 if trial_fitness < fitness_ll[i]:
#                     ll_pop[i] = trial
#                     fitness_ll[i] = trial_fitness

#         # Return the best lower-level solution found
#         best_ll_idx = np.argmin(fitness_ll)
#         return ll_pop[best_ll_idx]

#     def solve(self):
#         """
#         Executes the main nested DE optimization loop.
#         """
#         # 1. Initialize UL population
#         ul_pop = np.random.uniform(
#             self.problem.ul_bounds[0],
#             self.problem.ul_bounds[1],
#             size=(self.ul_pop_size, self.problem.ul_dim),
#         )

#         # 2. Evaluate initial UL population (requires running LL DE for each)
#         print("Evaluating initial UL population (this may take a while)...")
#         ll_solutions = np.array([self._run_lower_level_de(ul_ind) for ul_ind in ul_pop])
#         fitness_ul = np.array(
#             [self.problem.evaluate(ul_pop[i], ll_solutions[i])[0] for i in range(self.ul_pop_size)]
#         )
#         self.ul_nfe += self.ul_pop_size

#         best_fitness_overall = np.min(fitness_ul)

#         # 3. UL DE evolution loop
#         for gen in range(self.ul_generations):
#             for i in range(self.ul_pop_size):
#                 # a. Mutation
#                 idxs = [idx for idx in range(self.ul_pop_size) if idx != i]
#                 a, b, c = ul_pop[np.random.choice(idxs, 3, replace=False)]
#                 mutant = a + self.F * (b - c)
#                 mutant = np.clip(mutant, self.problem.ul_bounds[0], self.problem.ul_bounds[1])

#                 # b. Crossover
#                 cross_points = np.random.rand(self.problem.ul_dim) < self.Cr
#                 if not np.any(cross_points):
#                     cross_points[np.random.randint(0, self.problem.ul_dim)] = True
#                 trial_ul = np.where(cross_points, mutant, ul_pop[i])

#                 # c. Selection (requires expensive evaluation)
#                 trial_ll_sol = self._run_lower_level_de(trial_ul)
#                 trial_ul_fitness = self.problem.evaluate(trial_ul, trial_ll_sol)[0]
#                 self.ul_nfe += 1

#                 if trial_ul_fitness < fitness_ul[i]:
#                     ul_pop[i] = trial_ul
#                     ll_solutions[i] = trial_ll_sol
#                     fitness_ul[i] = trial_ul_fitness

#             best_gen_fitness = np.min(fitness_ul)
#             if best_gen_fitness < best_fitness_overall:
#                 best_fitness_overall = best_gen_fitness

#             # Log progress for this generation
#             self.log_generation(gen, best_fitness_overall, np.mean(fitness_ul))
#             if gen % 10 == 0:
#                 print(
#                     f"Gen {gen}: Best UL Fitness = {best_fitness_overall:.4f}, Avg NFE (UL/LL) = {self.ul_nfe}/{self.ll_nfe}"
#                 )

#         # Final result
#         best_idx = np.argmin(fitness_ul)
#         final_results = {
#             "final_ul_fitness": fitness_ul[best_idx],
#             "total_ul_nfe": self.ul_nfe,
#             "total_ll_nfe": self.ll_nfe,
#             "best_ul_solution": ul_pop[best_idx],
#             "corresponding_ll_solution": ll_solutions[best_idx],
#         }
#         return final_results


# Updated Implementation
import numpy as np
from .base_optimizer import BaseOptimizer


class NestedDE(BaseOptimizer):
    """
    Implements a classic Nested Differential Evolution algorithm.
    It uses DE for both the upper-level and lower-level optimization tasks.
    """

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self.ul_pop_size = self.config.get("ul_pop_size", 50)
        self.ll_pop_size = self.config.get("ll_pop_size", 30)
        self.generations = self.config.get("generations", 100)
        self.ll_generations = self.config.get("ll_generations", 25)
        self.F = self.config.get("F", 0.5)  # DE mutation factor
        self.Cr = self.config.get("Cr", 0.9)  # DE crossover rate

    def _run_ll_de(self, ul_vars):
        """Runs a full DE optimization on the lower level."""
        pop = np.random.uniform(
            self.problem.ll_bounds[0],
            self.problem.ll_bounds[1],
            size=(self.ll_pop_size, self.problem.ll_dim),
        )

        fitness = np.array([self.problem.evaluate(ul_vars, p, add_penalty=True)[1] for p in pop])
        self.ll_nfe += self.ll_pop_size

        for _ in range(self.ll_generations):
            for i in range(self.ll_pop_size):
                idxs = [idx for idx in range(self.ll_pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                # Mutation
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.problem.ll_bounds[0], self.problem.ll_bounds[1])

                # Crossover
                cross_points = np.random.rand(self.problem.ll_dim) <= self.Cr
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fitness = self.problem.evaluate(ul_vars, trial, add_penalty=True)[1]
                self.ll_nfe += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

        best_idx = np.argmin(fitness)
        return pop[best_idx]

    def solve(self):
        # Initialize UL population
        ul_pop = np.random.uniform(
            self.problem.ul_bounds[0],
            self.problem.ul_bounds[1],
            size=(self.ul_pop_size, self.problem.ul_dim),
        )

        # Evaluate initial population (requires full LL DE for each)
        ll_pop = np.array([self._run_ll_de(ul_ind) for ul_ind in ul_pop])
        fitness = np.array([self.problem.evaluate(ul_pop[i], ll_pop[i])[0] for i in range(self.ul_pop_size)])
        self.ul_nfe += self.ul_pop_size

        for gen in range(self.generations):
            for i in range(self.ul_pop_size):
                idxs = [idx for idx in range(self.ul_pop_size) if idx != i]
                a, b, c = ul_pop[np.random.choice(idxs, 3, replace=False)]

                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.problem.ul_bounds[0], self.problem.ul_bounds[1])

                cross_points = np.random.rand(self.problem.ul_dim) <= self.Cr
                trial_ul = np.where(cross_points, mutant, ul_pop[i])

                # Run full LL DE for the trial vector
                trial_ll = self._run_ll_de(trial_ul)

                trial_fitness = self.problem.evaluate(trial_ul, trial_ll)[0]
                self.ul_nfe += 1

                if trial_fitness < fitness[i]:
                    ul_pop[i] = trial_ul
                    ll_pop[i] = trial_ll
                    fitness[i] = trial_fitness

            self.log_generation(gen, np.min(fitness), self.ul_nfe)

        best_idx = np.argmin(fitness)
        return {
            "final_ul_fitness": fitness[best_idx],
            "total_ul_nfe": self.ul_nfe,
            "total_ll_nfe": self.ll_nfe,
            "best_ul_solution": ul_pop[best_idx],
            "corresponding_ll_solution": ll_pop[best_idx],
        }


# =============================================================================
# Test Module
# =============================================================================
if __name__ == "__main__":
    print("--- Verifying NestedDE Algorithm Implementation ---")

    # Define a simple configuration for the test
    test_config = {
        "ul_pop_size": 20,  # Smaller for a quick test
        "ll_pop_size": 15,
        "generations": 30,
        "ll_generations": 20,
        "F": 0.5,
        "Cr": 0.9,
    }

    # Get a problem to solve
    smd1_problem = get_smd_problem("SMD1")

    print(f"\nProblem: {smd1_problem}")
    print(f"Config: {test_config}")

    # Instantiate and run the optimizer
    try:
        optimizer = NestedDE(smd1_problem, test_config)
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
