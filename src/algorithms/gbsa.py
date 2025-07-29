#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 11:00:56 2025

@author: sanup
"""

# sace_project/src/algorithms/gbsa.py

import numpy as np
from .base_optimizer import BaseOptimizer


class GBSA(BaseOptimizer):
    """
    Implements a complete Generation-Based Step-size Approximation algorithm.
    This method uses a few steps of a local optimizer (gradient descent)
    to approximate the lower-level solution for offspring.
    """

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self.pop_size = self.config.get("pop_size", 50)
        self.generations = self.config.get("generations", 100)
        self.ll_refinement_steps = self.config.get("ll_refinement_steps", 5)
        self.ll_step_size = self.config.get("ll_step_size", 0.05)

        if not hasattr(self.problem, "evaluate_ll_gradient"):
            raise NotImplementedError(
                f"GBSA requires problem '{problem.name}' to have a 'evaluate_ll_gradient' method for its local search."
            )

    def _full_ll_solve(self, ul_vars):
        """A placeholder for a full LL solve (like in NestedDE) for initialization."""
        ll_vars = np.random.uniform(
            self.problem.ll_bounds[:, 0], self.problem.ll_bounds[:, 1], size=self.problem.ll_dim
        )
        for _ in range(50):  # More steps for initial solve
            grad = self.problem.evaluate_ll_gradient(ul_vars, ll_vars)
            ll_vars -= self.ll_step_size * grad
            ll_vars = np.clip(ll_vars, self.problem.ll_bounds[:, 0], self.problem.ll_bounds[:, 1])
            self.ll_nfe += 1
        return ll_vars

    def _approximate_ll_solve(self, ul_vars, ll_initial_guess):
        ll_vars = np.copy(ll_initial_guess)
        for _ in range(self.ll_refinement_steps):
            grad = self.problem.evaluate_ll_gradient(ul_vars, ll_vars)
            ll_vars -= self.ll_step_size * grad
            ll_vars = np.clip(ll_vars, self.problem.ll_bounds[:, 0], self.problem.ll_bounds[:, 1])
            self.ll_nfe += 1
        return ll_vars

    def solve(self):
        # Initialize population
        ul_pop = np.random.uniform(
            self.problem.ul_bounds[:, 0],
            self.problem.ul_bounds[:, 1],
            size=(self.pop_size, self.problem.ul_dim),
        )
        ll_pop = np.array([self._full_ll_solve(ul_ind) for ul_ind in ul_pop])
        fitness = np.array([self.problem.evaluate(ul_pop[i], ll_pop[i])[0] for i in range(self.pop_size)])
        self.ul_nfe += self.pop_size

        for gen in range(self.generations):
            # Tournament selection
            parent_indices = np.random.randint(0, self.pop_size, size=self.pop_size * 2)
            tourn_fitness = fitness[parent_indices]
            winners = np.argmin(tourn_fitness.reshape(-1, 2), axis=1)
            parent_pop_ul = ul_pop[parent_indices.reshape(-1, 2)[np.arange(self.pop_size), winners]]
            parent_pop_ll = ll_pop[parent_indices.reshape(-1, 2)[np.arange(self.pop_size), winners]]

            # Crossover and Mutation for UL
            offspring_ul = parent_pop_ul + np.random.normal(0, 0.1, size=ul_pop.shape)
            offspring_ul = np.clip(offspring_ul, self.problem.ul_bounds[:, 0], self.problem.ul_bounds[:, 1])

            # Approximate LL solutions for offspring
            offspring_ll = np.array(
                [self._approximate_ll_solve(offspring_ul[i], parent_pop_ll[i]) for i in range(self.pop_size)]
            )
            offspring_fitness = np.array(
                [self.problem.evaluate(offspring_ul[i], offspring_ll[i])[0] for i in range(self.pop_size)]
            )
            self.ul_nfe += self.pop_size

            # Generational replacement
            ul_pop, ll_pop, fitness = offspring_ul, offspring_ll, offspring_fitness

            self.log_generation(gen, np.min(fitness), self.ul_nfe)

        best_idx = np.argmin(fitness)
        # return {
        #     "final_ul_fitness": fitness[best_idx],
        #     "total_ul_nfe": self.ul_nfe,
        #     "total_ll_nfe": self.ll_nfe,
        # }

        return {
            "final_ul_fitness": fitness[best_idx],
            "total_ul_nfe": self.ul_nfe,
            "total_ll_nfe": self.ll_nfe,
            "best_ul_solution": ul_pop[best_idx],
            "corresponding_ll_solution": ll_pop[best_idx],
        }
