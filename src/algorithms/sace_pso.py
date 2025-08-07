#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 16:04:02 2025

@author: sanup
"""

# sace_project/src/algorithms/sace_pso.py

import numpy as np
from .base_optimizer import BaseOptimizer

# The SACE_PSO will reuse the same surrogate strategies as SACE_ES
from .sace_es import IndependentGPs, MOGP_GPy, HeteroscedasticGPs


class SACE_PSO(BaseOptimizer):
    """
    Implements a Surrogate-Assisted Co-evolutionary Particle Swarm Optimization (SACE-PSO).

    This is a new variant of the SACE framework. It uses:
    - An Adaptive PSO as the upper-level global optimizer.
    - The same surrogate modeling strategies (Independent, MOGP, Heteroscedastic).
    - The same LCB infill criterion to select candidates for expensive evaluation.
    """

    def __init__(self, problem, config):
        super().__init__(problem, config)
        # PSO specific parameters
        self.ul_pop_size = self.config.get("ul_pop_size", 50)
        self.w_max = self.config.get("w_max", 0.9)
        self.w_min = self.config.get("w_min", 0.4)
        self.c1 = self.config.get("c1", 2.0)
        self.c2 = self.config.get("c2", 2.0)

        # Shared SACE parameters
        self.generations = self.config.get("generations", 100)
        self.initial_samples = self.config.get("initial_samples", 20)
        self.kappa = self.config.get("lcb_kappa", 2.0)
        self.ll_pop_size = self.config.get("ll_pop_size", 30)
        self.ll_generations = self.config.get("ll_generations", 25)

        # Surrogate Strategy Selection
        surrogate_config = self.config.get("surrogate_config", {"strategy": "independent"})
        strategy = surrogate_config["strategy"]
        surrogate_map = {
            "independent": IndependentGPs,
            "mogp": MOGP_GPy,
            "heteroscedastic": HeteroscedasticGPs,
        }
        self.surrogates_y = surrogate_map[strategy](self.problem.ll_dim)
        self.archive_ul, self.archive_ll_sols = [], []

    def _run_lower_level_es(self, ul_individual):
        """Finds the follower's response. Re-using the ES solver from SACE_ES."""
        ll_pop = np.random.uniform(
            self.problem.ll_bounds[:, 0],
            self.problem.ll_bounds[:, 1],
            size=(self.ll_pop_size, self.problem.ll_dim),
        )
        fitness_ll = np.array([self.problem.evaluate(ul_individual, p, add_penalty=True)[1] for p in ll_pop])
        self.ll_nfe += self.ll_pop_size
        for _ in range(self.ll_generations):
            parents = ll_pop[np.random.choice(len(ll_pop), self.ll_pop_size, replace=True)]
            offspring = np.clip(
                parents + np.random.normal(0, 0.2, size=parents.shape),
                self.problem.ll_bounds[:, 0],
                self.problem.ll_bounds[:, 1],
            )
            offspring_fitness = np.array(
                [self.problem.evaluate(ul_individual, o, add_penalty=True)[1] for o in offspring]
            )
            self.ll_nfe += self.ll_pop_size
            combined_pop = np.vstack([ll_pop, offspring])
            combined_fitness = np.concatenate([fitness_ll, offspring_fitness])
            best_idxs = np.argsort(combined_fitness)[: self.ll_pop_size]
            ll_pop, fitness_ll = combined_pop[best_idxs], combined_fitness[best_idxs]
        return ll_pop[0], self.problem.evaluate_ll_constraints(ul_individual, ll_pop[0])

    def solve(self):
        # Phase 1: Initial Sampling (same as SACE_ES)
        print("Phase 1: Initial sampling...")
        initial_ul_samples = np.random.uniform(
            self.problem.ul_bounds[:, 0],
            self.problem.ul_bounds[:, 1],
            size=(self.initial_samples, self.problem.ul_dim),
        )
        for ul_sample in initial_ul_samples:
            ll_opt, _ = self._run_lower_level_es(ul_sample)
            self.archive_ul.append(ul_sample)
            self.archive_ll_sols.append(ll_opt)
        self.archive_ul = np.array(self.archive_ul)
        self.archive_ll_sols = np.array(self.archive_ll_sols)
        self.surrogates_y.train(self.archive_ul, self.archive_ll_sols)
        print("Surrogate models trained.")

        # Phase 2: Main PSO Evolutionary Loop
        print("Phase 2: Starting optimization with SACE-PSO...")
        # Initialize PSO particles
        positions = np.random.uniform(
            self.problem.ul_bounds[:, 0],
            self.problem.ul_bounds[:, 1],
            size=(self.ul_pop_size, self.problem.ul_dim),
        )
        velocities = np.zeros_like(positions)
        pbest_positions = np.copy(positions)

        # Evaluate initial pbest on surrogate
        pred_ll_sols, _ = self.surrogates_y.predict(pbest_positions)
        pbest_fitness = np.array(
            [self.problem.evaluate(pbest_positions[i], pred_ll_sols[i])[0] for i in range(self.ul_pop_size)]
        )

        gbest_idx = np.argmin(pbest_fitness)
        gbest_position = np.copy(pbest_positions[gbest_idx])

        for gen in range(self.generations):
            # Adaptive inertia weight
            w = self.w_max - (self.w_max - self.w_min) * (gen / self.generations)

            # Update velocities and positions
            r1, r2 = np.random.rand(2, self.ul_pop_size, self.problem.ul_dim)
            velocities = (
                w * velocities
                + self.c1 * r1 * (pbest_positions - positions)
                + self.c2 * r2 * (gbest_position - positions)
            )
            positions += velocities
            positions = np.clip(positions, self.problem.ul_bounds[:, 0], self.problem.ul_bounds[:, 1])

            # Use LCB to find the most promising infill point from the current swarm
            pred_ll_sols, pred_ll_vars = self.surrogates_y.predict(positions)
            pred_ul_fitness = np.array(
                [self.problem.evaluate(positions[i], pred_ll_sols[i])[0] for i in range(self.ul_pop_size)]
            )
            lcb_scores = pred_ul_fitness - self.kappa * np.mean(np.sqrt(pred_ll_vars + 1e-9), axis=1)

            infill_idx = np.argmin(lcb_scores)
            promising_ul = positions[infill_idx]

            # Exact evaluation of the promising point
            exact_ll_sol, _ = self._run_lower_level_es(promising_ul)
            self.archive_ul = np.vstack([self.archive_ul, promising_ul])
            self.archive_ll_sols = np.vstack([self.archive_ll_sols, exact_ll_sol])

            # Retrain surrogates with new data
            self.surrogates_y.train(self.archive_ul, self.archive_ll_sols)

            # Update pbest and gbest using the CHEAP surrogate model for the whole swarm
            fitness_surrogate = np.array(
                [self.problem.evaluate(positions[i], pred_ll_sols[i])[0] for i in range(self.ul_pop_size)]
            )

            update_mask = fitness_surrogate < pbest_fitness
            pbest_positions[update_mask] = positions[update_mask]
            pbest_fitness[update_mask] = fitness_surrogate[update_mask]

            current_best_idx = np.argmin(pbest_fitness)
            gbest_position = np.copy(pbest_positions[current_best_idx])

            # Logging
            best_archive_fitness = np.min(
                [
                    self.problem.evaluate(self.archive_ul[i], self.archive_ll_sols[i])[0]
                    for i in range(len(self.archive_ul))
                ]
            )
            self.log_generation(gen, best_archive_fitness, np.mean(fitness_surrogate))

        # Final result is the best found in the archive of exactly evaluated solutions
        final_fitness_values = np.array(
            [
                self.problem.evaluate(self.archive_ul[i], self.archive_ll_sols[i])[0]
                for i in range(len(self.archive_ul))
            ]
        )
        best_archive_idx = np.argmin(final_fitness_values)
        best_ul_solution = self.archive_ul[best_archive_idx]
        corresponding_ll_solution = self.archive_ll_sols[best_archive_idx]

        return {
            "final_ul_fitness": final_fitness_values[best_archive_idx],
            "total_ul_nfe": self.ul_nfe,
            "total_ll_nfe": self.ll_nfe,
            "best_ul_solution": best_ul_solution,
            "corresponding_ll_solution": corresponding_ll_solution,
        }
