#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:09:36 2025

@author: sanup
"""

# sace_project/src/algorithms/biga.py

import numpy as np
from .base_optimizer import BaseOptimizer
from ..problems.smd_suite import get_smd_problem


class BiGA(BaseOptimizer):
    """
    Implements a co-evolutionary Bilevel Genetic Algorithm (BiGA).

    This approach maintains two interacting populations, one for the upper level
    and one for the lower level, which evolve simultaneously.
    """

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self.ul_pop_size = self.config.get("ul_pop_size", 50)
        self.ll_pop_size = self.config.get("ll_pop_size", 50)
        self.generations = self.config.get("generations", 100)
        # Generations to refine the LL pop periodically
        self.ll_refinement_gens = self.config.get("ll_refinement_gens", 5)

        # GA-specific parameters
        self.crossover_prob = self.config.get("crossover_prob", 0.9)
        self.mutation_prob = self.config.get("mutation_prob", 1.0 / self.problem.ul_dim)
        self.eta = self.config.get("eta", 20)  # Distribution index for SBX and Polynomial Mutation

    def _sbx_crossover(self, parent1, parent2):
        """Simulated Binary Crossover (SBX)."""
        child1, child2 = parent1.copy(), parent2.copy()
        for i in range(len(parent1)):
            if np.random.rand() < self.crossover_prob:
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (self.eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (self.eta + 1))

                c1 = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                c2 = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                child1[i], child2[i] = c1, c2
        return child1, child2

    def _polynomial_mutation(self, individual, bounds):
        """Polynomial Mutation."""
        mutated_ind = individual.copy()
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_prob:
                delta1 = (individual[i] - bounds[0]) / (bounds[1] - bounds[0])
                delta2 = (bounds[1] - individual[i]) / (bounds[1] - bounds[0])
                u = np.random.rand()

                if u <= 0.5:
                    deltaq = ((2 * u) + (1 - 2 * u) * (1 - delta1) ** (self.eta + 1)) ** (
                        1 / (self.eta + 1)
                    ) - 1
                else:
                    deltaq = 1 - (2 * (1 - u) + 2 * u * (1 - delta2) ** (self.eta + 1)) ** (
                        1 / (self.eta + 1)
                    )

                mutated_ind[i] += deltaq * (bounds[1] - bounds[0])

        mutated_ind = np.clip(mutated_ind, bounds[0], bounds[1])
        return mutated_ind

    def _evolve_population(self, pop, fitness, bounds, fixed_vars=None, level="ul"):
        """Evolves a population for one generation using GA operators."""
        new_pop = []
        for _ in range(len(pop) // 2):
            # Tournament selection
            p1_idx = np.random.randint(0, len(pop))
            p2_idx = np.random.randint(0, len(pop))
            parent1 = pop[p1_idx] if fitness[p1_idx] < fitness[p2_idx] else pop[p2_idx]

            p3_idx = np.random.randint(0, len(pop))
            p4_idx = np.random.randint(0, len(pop))
            parent2 = pop[p3_idx] if fitness[p3_idx] < fitness[p4_idx] else pop[p4_idx]

            # Crossover and Mutation
            child1, child2 = self._sbx_crossover(parent1, parent2)
            child1 = self._polynomial_mutation(child1, bounds)
            child2 = self._polynomial_mutation(child2, bounds)
            new_pop.extend([child1, child2])

        # Evaluate new population
        new_fitness = []
        if level == "ul":
            best_ll_sol = fixed_vars  # fixed_vars is the best ll solution
            for ul_ind in new_pop:
                fit, _ = self.problem.evaluate(ul_ind, best_ll_sol)
                new_fitness.append(fit)
            self.ul_nfe += len(new_pop)
        else:  # level == 'll'
            ul_sol = fixed_vars  # fixed_vars is the best ul solution
            for ll_ind in new_pop:
                _, fit = self.problem.evaluate(ul_sol, ll_ind)
                new_fitness.append(fit)
            self.ll_nfe += len(new_pop)

        # Elitist replacement
        combined_pop = np.vstack([pop, new_pop])
        combined_fitness = np.concatenate([fitness, new_fitness])
        best_indices = np.argsort(combined_fitness)[: len(pop)]

        return combined_pop[best_indices], combined_fitness[best_indices]

    def solve(self):
        # 1. Initialize UL and LL populations
        ul_pop = np.random.uniform(
            self.problem.ul_bounds[0],
            self.problem.ul_bounds[1],
            size=(self.ul_pop_size, self.problem.ul_dim),
        )
        ll_pop = np.random.uniform(
            self.problem.ll_bounds[0],
            self.problem.ll_bounds[1],
            size=(self.ll_pop_size, self.problem.ll_dim),
        )

        # 2. Initial evaluation
        # Evaluate LL population against a random UL individual
        fitness_ll = np.array([self.problem.evaluate(ul_pop[0], ll_ind)[1] for ll_ind in ll_pop])
        self.ll_nfe += self.ll_pop_size

        # Evaluate UL population against the best LL individual
        best_ll_idx = np.argmin(fitness_ll)
        best_ll_sol = ll_pop[best_ll_idx]
        fitness_ul = np.array([self.problem.evaluate(ul_ind, best_ll_sol)[0] for ul_ind in ul_pop])
        self.ul_nfe += self.ul_pop_size

        best_fitness_overall = np.min(fitness_ul)

        # 3. Main co-evolutionary loop
        for gen in range(self.generations):
            # Evolve UL population based on the current best follower
            ul_pop, fitness_ul = self._evolve_population(
                ul_pop, fitness_ul, self.problem.ul_bounds, fixed_vars=best_ll_sol, level="ul"
            )

            # Identify current best leader
            best_ul_idx = np.argmin(fitness_ul)
            best_ul_sol = ul_pop[best_ul_idx]

            # Periodically refine the LL population based on the best leader
            if gen % self.ll_refinement_gens == 0:
                # Re-evaluate the LL population with the new best leader
                fitness_ll = np.array([self.problem.evaluate(best_ul_sol, ll_ind)[1] for ll_ind in ll_pop])
                self.ll_nfe += self.ll_pop_size
                # Evolve LL population for a few generations
                for _ in range(self.ll_refinement_gens):
                    ll_pop, fitness_ll = self._evolve_population(
                        ll_pop, fitness_ll, self.problem.ll_bounds, fixed_vars=best_ul_sol, level="ll"
                    )

            # Update the best follower for the next UL generation
            best_ll_idx = np.argmin(fitness_ll)
            best_ll_sol = ll_pop[best_ll_idx]
            best_fitness_overall = fitness_ul[best_ul_idx]

            self.log_generation(gen, best_fitness_overall, np.mean(fitness_ul))
            if gen % 10 == 0:
                print(
                    f"Gen {gen}: Best UL Fitness = {best_fitness_overall:.4f}, NFE (UL/LL) = {self.ul_nfe}/{self.ll_nfe}"
                )

        # Final result
        best_ul_idx = np.argmin(fitness_ul)
        final_results = {
            "final_ul_fitness": fitness_ul[best_ul_idx],
            "total_ul_nfe": self.ul_nfe,
            "total_ll_nfe": self.ll_nfe,
            "best_ul_solution": ul_pop[best_ul_idx],
            "corresponding_ll_solution": best_ll_sol,  # The best LL sol corresponding to the final UL pop
        }
        return final_results


# =============================================================================
# Test Module
# =============================================================================
if __name__ == "__main__":
    print("--- Verifying BiGA Algorithm Implementation ---")

    test_config = {
        "ul_pop_size": 30,  # Smaller for a quick test
        "ll_pop_size": 30,
        "generations": 50,
        "ll_refinement_gens": 5,
        "crossover_prob": 0.9,
        "mutation_prob": 0.1,
        "eta": 20,
    }

    smd1_problem = get_smd_problem("SMD1")

    print(f"\nProblem: {smd1_problem}")
    print(f"Config: {test_config}")

    try:
        optimizer = BiGA(smd1_problem, test_config)
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
