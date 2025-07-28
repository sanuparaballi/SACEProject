#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 11:00:17 2025

@author: sanup
"""

# sace_project/src/algorithms/bboa_kkt.py

import numpy as np
from .base_optimizer import BaseOptimizer


class BBOA_KKT(BaseOptimizer):
    """
    Implements a Bilevel Biobjective Optimization Algorithm using KKT conditions.
    This approach reformulates the bilevel problem into a single-level,
    two-objective problem.
    """

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self.pop_size = self.config.get("pop_size", 100)
        self.generations = self.config.get("generations", 100)

        # Check if the problem supports gradient evaluation
        if not hasattr(self.problem, "evaluate_ll_gradient"):
            raise NotImplementedError(
                f"BBOA-KKT requires problem '{problem.name}' to have a 'evaluate_ll_gradient' method."
            )

    def _non_dominated_sort(self, fitnesses):
        """Performs non-dominated sorting and returns ranks."""
        n_points = fitnesses.shape[0]
        ranks = np.zeros(n_points, dtype=int)

        # This is a simplified implementation of non-dominated sorting
        for i in range(n_points):
            is_dominated = np.any(np.all(fitnesses < fitnesses[i], axis=1))
            if not is_dominated:
                ranks[i] = 0  # Rank 0 for non-dominated set
            else:
                # In a full NSGA-II, we'd find the count of dominating solutions
                ranks[i] = 1  # Simplified: all others are rank 1
        return ranks

    def solve(self):
        # Initialize a random population of (ul_vars, ll_vars)
        ul_pop = np.random.uniform(
            self.problem.ul_bounds[0], self.problem.ul_bounds[1], size=(self.pop_size, self.problem.ul_dim)
        )
        ll_pop = np.random.uniform(
            self.problem.ll_bounds[0], self.problem.ll_bounds[1], size=(self.pop_size, self.problem.ll_dim)
        )
        pop = np.hstack([ul_pop, ll_pop])

        for gen in range(self.generations):
            # Evaluate the two objectives for each individual
            ul_fitnesses = np.zeros(self.pop_size)
            kkt_violations = np.zeros(self.pop_size)

            for i, ind in enumerate(pop):
                ul_vars, ll_vars = ind[: self.problem.ul_dim], ind[self.problem.ul_dim :]
                ul_fitnesses[i], _ = self.problem.evaluate(ul_vars, ll_vars, add_penalty=False)

                grad = self.problem.evaluate_ll_gradient(ul_vars, ll_vars)
                kkt_violations[i] = np.linalg.norm(grad)

            fitnesses_2d = np.vstack([ul_fitnesses, kkt_violations]).T

            # Perform non-dominated sorting to find the Pareto front
            ranks = self._non_dominated_sort(fitnesses_2d)

            # Select parents from the best ranks (simplified tournament)
            parent_indices = np.random.choice(np.where(ranks == 0)[0], self.pop_size)
            parents = pop[parent_indices]

            # Create offspring (simplified mutation)
            offspring = parents + np.random.normal(0, 0.1, size=parents.shape)
            pop = offspring  # The new generation

            self.ul_nfe += self.pop_size
            self.ll_nfe += self.pop_size  # Each eval counts as one for LL gradient

        # Find the best solution from the final population
        # The best is the one on the front with the lowest KKT violation
        final_ul_fit = np.zeros(self.pop_size)
        final_kkt_v = np.zeros(self.pop_size)
        for i, ind in enumerate(pop):
            ul_vars, ll_vars = ind[: self.problem.ul_dim], ind[self.problem.ul_dim :]
            final_ul_fit[i], _ = self.problem.evaluate(ul_vars, ll_vars, add_penalty=False)
            final_kkt_v[i] = np.linalg.norm(self.problem.evaluate_ll_gradient(ul_vars, ll_vars))

        feasible_mask = final_kkt_v < 1e-4
        if np.any(feasible_mask):
            best_idx = np.argmin(np.where(feasible_mask, final_ul_fit, np.inf))
        else:
            best_idx = np.argmin(final_kkt_v)  # Fallback to least infeasible

        best_solution = pop[best_idx]
        best_ul_vars = best_solution[: self.problem.ul_dim]
        best_ll_vars = best_solution[self.problem.ul_dim :]

        return {
            "final_ul_fitness": self.problem.evaluate(best_ul_vars, best_ll_vars)[0],
            "total_ul_nfe": self.ul_nfe,
            "total_ll_nfe": self.ll_nfe,
        }


# Updated implementation. Testing the original first
# class BBOA_KKT(BaseOptimizer):
#     """
#     Implements a Bilevel Biobjective Optimization Algorithm using KKT conditions.
#     This approach reformulates the bilevel problem into a single-level,
#     two-objective problem. This is a complete implementation.
#     """
#     def __init__(self, problem, config):
#         super().__init__(problem, config)
#         self.pop_size = self.config.get("pop_size", 100)
#         self.generations = self.config.get("generations", 100)

#         if not hasattr(self.problem, 'evaluate_ll_gradient'):
#             raise NotImplementedError(
#                 f"BBOA-KKT requires problem '{problem.name}' to have a 'evaluate_ll_gradient' method."
#             )

#     def _non_dominated_sort(self, objectives):
#         n = objectives.shape[0]
#         fronts = []
#         domination_counts = np.zeros(n, dtype=int)
#         dominated_solutions = [[] for _ in range(n)]

#         for i in range(n):
#             for j in range(i + 1, n):
#                 # Check for dominance
#                 p, q = objectives[i], objectives[j]
#                 if np.all(p <= q) and np.any(p < q): # p dominates q
#                     dominated_solutions[i].append(j)
#                     domination_counts[j] += 1
#                 elif np.all(q <= p) and np.any(q < p): # q dominates p
#                     dominated_solutions[j].append(i)
#                     domination_counts[i] += 1

#         current_front = np.where(domination_counts == 0)[0].tolist()
#         front_idx = 0
#         while current_front:
#             fronts.append(current_front)
#             next_front = []
#             for i in current_front:
#                 for j in dominated_solutions[i]:
#                     domination_counts[j] -= 1
#                     if domination_counts[j] == 0:
#                         next_front.append(j)
#             front_idx += 1
#             current_front = next_front

#         ranks = np.zeros(n, dtype=int)
#         for i, front in enumerate(fronts):
#             ranks[front] = i
#         return ranks

#     def solve(self):
#         ul_pop = np.random.uniform(self.problem.ul_bounds[0], self.problem.ul_bounds[1],
#                                    size=(self.pop_size, self.problem.ul_dim))
#         ll_pop = np.random.uniform(self.problem.ll_bounds[0], self.problem.ll_bounds[1],
#                                    size=(self.pop_size, self.problem.ll_dim))
#         pop = np.hstack([ul_pop, ll_pop])

#         for gen in range(self.generations):
#             # Evaluate the two objectives
#             ul_fitnesses = np.array([self.problem.evaluate(p[:self.problem.ul_dim], p[self.problem.ul_dim:])[0] for p in pop])
#             kkt_violations = np.array([np.linalg.norm(self.problem.evaluate_ll_gradient(p[:self.problem.ul_dim], p[self.problem.ul_dim:])) for p in pop])
#             self.ul_nfe += self.pop_size
#             self.ll_nfe += self.pop_size # Gradient eval counts as one LL NFE

#             objectives = np.vstack([ul_fitnesses, kkt_violations]).T
#             ranks = self._non_dominated_sort(objectives)

#             # Binary tournament selection based on rank
#             new_pop = []
#             for _ in range(self.pop_size):
#                 p1_idx, p2_idx = np.random.choice(len(pop), 2, replace=False)
#                 winner_idx = p1_idx if ranks[p1_idx] < ranks[p2_idx] else p2_idx
#                 new_pop.append(pop[winner_idx])

#             # Simple mutation for next generation
#             pop = np.array(new_pop) + np.random.normal(0, 0.1, size=pop.shape)

#         # Final selection from the last population
#         final_ul_fit = np.array([self.problem.evaluate(p[:self.problem.ul_dim], p[self.problem.ul_dim:])[0] for p in pop])
#         final_kkt_v = np.array([np.linalg.norm(self.problem.evaluate_ll_gradient(p[:self.problem.ul_dim], p[self.problem.ul_dim:])) for p in pop])

#         feasible_mask = final_kkt_v < 1e-4
#         if np.any(feasible_mask):
#             best_idx = np.argmin(np.where(feasible_mask, final_ul_fit, np.inf))
#         else:
#             best_idx = np.argmin(final_kkt_v)

#         best_ul, best_ll = pop[best_idx][:self.problem.ul_dim], pop[best_idx][self.problem.ul_dim:]
#         return {"final_ul_fitness": final_ul_fit[best_idx], "total_ul_nfe": self.ul_nfe, "total_ll_nfe": self.ll_nfe}
