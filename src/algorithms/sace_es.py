#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 23:22:22 2025

@author: sanup
"""

# sace_project/src/algorithms/sace_es_v3.0.py

import numpy as np
import GPy
from .base_optimizer import BaseOptimizer
from ..surrogates.gaussian_process import GaussianProcessSurrogate
from ..problems.smd_suite import get_smd_problem

# =============================================================================
# Surrogate Strategy Implementations (v3.0)
# =============================================================================


class BaseSurrogateStrategy:
    """Base class for different surrogate modeling strategies."""

    def __init__(self, problem_dim):
        self.problem_dim = problem_dim
        self.is_trained = False

    def train(self, X, Y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class IndependentGPs(BaseSurrogateStrategy):
    """Variant 1: Models each output dimension with a separate GP."""

    def __init__(self, problem_dim):
        super().__init__(problem_dim)
        self.models = [GaussianProcessSurrogate() for _ in range(problem_dim)]

    def train(self, X, Y):
        for i in range(self.problem_dim):
            self.models[i].train(X, Y[:, i])
        self.is_trained = True

    def predict(self, X):
        means = np.array([m.predict(X)[0] for m in self.models]).T
        variances = np.array([m.predict(X)[1] for m in self.models]).T
        return means, variances


class MOGP_GPy(BaseSurrogateStrategy):
    """Variant 2: Full Multi-Output GP using GPy Coregionalized Regression."""

    def __init__(self, problem_dim):
        super().__init__(problem_dim)
        self.model = None

    def train(self, X, Y):
        n_samples, input_dim = X.shape

        # Reshape data for GPy's Coregionalized model
        # X_augmented has a new column for the output index
        X_augmented = np.hstack([X for _ in range(self.problem_dim)])
        X_augmented = X_augmented.reshape(-1, input_dim)
        index_col = np.array([i for i in range(self.problem_dim) for _ in range(n_samples)]).reshape(-1, 1)
        X_augmented = np.hstack([X_augmented, index_col])

        # Y is flattened
        Y_flattened = Y.T.reshape(-1, 1)

        # Create the Coregionalized kernel
        kernel = GPy.kern.RBF(input_dim=input_dim, ARD=True)
        coreg_kernel = GPy.kern.Coregionalize(input_dim=1, output_dim=self.problem_dim, rank=1)
        self.kernel = kernel * coreg_kernel

        # Create and optimize the GPy model
        self.model = GPy.models.GPCoregionalizedRegression(
            X_list=[X_augmented], Y_list=[Y_flattened], kernel=self.kernel
        )
        self.model.optimize(messages=False, optimizer="lbfgs", max_iters=1000)
        self.is_trained = True

    def predict(self, X):
        n_samples, input_dim = X.shape

        # Prepare input for prediction
        X_pred_augmented = np.hstack([X for _ in range(self.problem_dim)]).reshape(-1, input_dim)
        index_col = np.array([i for i in range(self.problem_dim) for _ in range(n_samples)]).reshape(-1, 1)

        # Noise dictionary for prediction
        noise_dict = {"output_index": index_col}

        # Predict and reshape back
        mean_flat, var_flat = self.model.predict(
            np.hstack([X_pred_augmented, index_col]), Y_metadata=noise_dict
        )

        mean = mean_flat.reshape(self.problem_dim, n_samples).T
        variance = var_flat.reshape(self.problem_dim, n_samples).T

        return mean, variance


class HeteroscedasticGPs(BaseSurrogateStrategy):
    """Variant 3: Models mean and noise variance for each dimension."""

    def __init__(self, problem_dim):
        super().__init__(problem_dim)
        # Implementation remains conceptual as it's complex and for comparison
        self.model = IndependentGPs(problem_dim)  # Placeholder
        print("INFO: Using Heteroscedastic placeholder. A dedicated library/implementation is recommended.")

    def train(self, X, Y):
        self.model.train(X, Y)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)


# =============================================================================
# SACE-ES Algorithm v3.0
# =============================================================================


class SACE_ES_v3(BaseOptimizer):
    """
    SACE-ES Version 3.0:
    - Implements a (mu, lambda)-ES with Self-Adaptation.
    - Uses GPy for a full MOGP implementation.
    - Retains LCB infill and pluggable surrogate strategies.
    """

    def __init__(self, problem, config):
        super().__init__(problem, config)
        self.mu = config.get("ul_mu", 15)
        self.lambda_ = config.get("ul_lambda", 30)
        # Other params from v2.0
        self.ll_pop_size = self.config.get("ll_pop_size", 30)
        self.generations = self.config.get("generations", 100)
        self.ll_generations = self.config.get("ll_generations", 20)
        self.initial_samples = self.config.get("initial_samples", 20)
        self.kappa = self.config.get("lcb_kappa", 2.0)

        # Init learning rates for self-adaptation
        self.tau = 1.0 / np.sqrt(2 * self.problem.ul_dim)
        self.tau_prime = 1.0 / np.sqrt(2 * np.sqrt(self.problem.ul_dim))

        # Surrogate Strategy Selection
        surrogate_config = self.config.get("surrogate_config", {"strategy": "independent"})
        strategy = surrogate_config["strategy"]
        surrogate_map = {
            "independent": IndependentGPs,
            "mogp": MOGP_GPy,
            "heteroscedastic": HeteroscedasticGPs,
        }
        self.surrogates_y = surrogate_map[strategy](self.problem.ll_dim)
        self.surrogates_c = (
            IndependentGPs(self.problem.num_ll_constraints) if self.problem.num_ll_constraints > 0 else None
        )

        self.archive_ul = []
        self.archive_ll_sols = []
        self.archive_ll_cons = []

    def _initialize_population(self):
        # Initialize decision variables
        pop_x = np.random.uniform(
            self.problem.ul_bounds[0], self.problem.ul_bounds[1], size=(self.mu, self.problem.ul_dim)
        )
        # Initialize strategy parameters (mutation strengths)
        pop_sigma = np.random.uniform(0.1, 0.5, size=(self.mu, self.problem.ul_dim))
        return np.hstack([pop_x, pop_sigma])

    def _recombine(self, pop):
        # Discrete recombination for decision variables, intermediate for strategy
        offspring_pop = []
        for _ in range(self.lambda_):
            p1, p2 = pop[np.random.choice(len(pop), 2, replace=False)]
            x1, s1 = p1[: self.problem.ul_dim], p1[self.problem.ul_dim :]
            x2, s2 = p2[: self.problem.ul_dim], p2[self.problem.ul_dim :]

            mask = np.random.rand(self.problem.ul_dim) > 0.5
            child_x = np.where(mask, x1, x2)
            child_s = (s1 + s2) / 2.0
            offspring_pop.append(np.concatenate([child_x, child_s]))
        return np.array(offspring_pop)

    def _self_adaptive_mutation(self, pop):
        mutated_pop = []
        for ind in pop:
            x, sigma = ind[: self.problem.ul_dim], ind[self.problem.ul_dim :]

            # Mutate strategy parameters
            n_prime = np.random.randn()
            n_i = np.random.randn(self.problem.ul_dim)
            sigma_prime = sigma * np.exp(self.tau_prime * n_prime + self.tau * n_i)

            # Mutate decision variables with new sigma
            x_prime = x + sigma_prime * np.random.randn(self.problem.ul_dim)
            x_prime = np.clip(x_prime, self.problem.ul_bounds[0], self.problem.ul_bounds[1])

            mutated_pop.append(np.concatenate([x_prime, sigma_prime]))
        return np.array(mutated_pop)

    def _get_fitness(self, pop):
        # Use surrogate to get fitness for the current population
        pop_x = pop[:, : self.problem.ul_dim]
        pred_ll_sols, _ = self.surrogates_y.predict(pop_x)
        pred_ul_fitness = np.array(
            [
                self.problem.evaluate(np.atleast_1d(pop_x[i]), np.atleast_1d(pred_ll_sols[i]))[0]
                for i in range(len(pop_x))
            ]
        )
        return pred_ul_fitness

    def solve(self):
        # Phase 1: Initial Sampling (same as before)
        print(f"Phase 1: Initial sampling...")
        # ... [Code from v2.0 for initial sampling and training] ...
        initial_ul_samples = np.random.uniform(
            self.problem.ul_bounds[0],
            self.problem.ul_bounds[1],
            size=(self.initial_samples, self.problem.ul_dim),
        )
        for ul_sample in initial_ul_samples:
            ll_opt, ll_cons = self._run_lower_level_es(ul_sample)  # Placeholder for actual LL solve
            self.archive_ul.append(ul_sample)
            self.archive_ll_sols.append(ll_opt)
        self.archive_ul = np.array(self.archive_ul)
        self.archive_ll_sols = np.array(self.archive_ll_sols)
        self.surrogates_y.train(self.archive_ul, self.archive_ll_sols)
        print("Surrogate models trained.")

        # Phase 2: Main Evolutionary Loop
        print("Phase 2: Starting optimization with Self-Adaptive ES...")
        ul_pop = self._initialize_population()

        for gen in range(self.generations):
            # Create offspring via recombination and self-adaptive mutation
            recombined_pop = self._recombine(ul_pop)
            offspring_ul_pop = self._self_adaptive_mutation(recombined_pop)

            # Use LCB to find the most promising infill point from offspring
            offspring_x = offspring_ul_pop[:, : self.problem.ul_dim]
            pred_ll_sols, pred_ll_vars = self.surrogates_y.predict(offspring_x)
            pred_ul_fitness = np.array(
                [self.problem.evaluate(offspring_x[i], pred_ll_sols[i])[0] for i in range(len(offspring_x))]
            )
            lcb_scores = pred_ul_fitness - self.kappa * np.mean(np.sqrt(pred_ll_vars + 1e-6), axis=1)

            infill_idx = np.argmin(lcb_scores)
            promising_ul = offspring_x[infill_idx]

            # Exact evaluation and archive update
            exact_ll_sol, _ = self._run_lower_level_es(promising_ul)  # Placeholder
            self.archive_ul = np.vstack([self.archive_ul, promising_ul])
            self.archive_ll_sols = np.vstack([self.archive_ll_sols, exact_ll_sol])

            # Retrain surrogates with new data
            self.surrogates_y.train(self.archive_ul, self.archive_ll_sols)

            # Selection for next generation: evaluate parents + offspring and choose best mu
            combined_pop = np.vstack([ul_pop, offspring_ul_pop])
            fitness_values = self._get_fitness(combined_pop)
            best_indices = np.argsort(fitness_values)[: self.mu]
            ul_pop = combined_pop[best_indices]

            # Logging
            best_archive_fitness = np.min(
                [
                    self.problem.evaluate(self.archive_ul[i], self.archive_ll_sols[i])[0]
                    for i in range(len(self.archive_ul))
                ]
            )
            if gen % 10 == 0:
                print(f"Gen {gen}: Best True Fitness in Archive = {best_archive_fitness:.4f}")

        final_fitness_values = np.array(
            [
                self.problem.evaluate(self.archive_ul[i], self.archive_ll_sols[i])[0]
                for i in range(len(self.archive_ul))
            ]
        )
        best_archive_idx = np.argmin(final_fitness_values)
        return {"final_ul_fitness": final_fitness_values[best_archive_idx]}

    def _run_lower_level_es(self, ul_individual):  # Placeholder from v2.0
        ll_pop = np.random.uniform(
            self.problem.ll_bounds[0],
            self.problem.ll_bounds[1],
            size=(self.ll_pop_size, self.problem.ll_dim),
        )
        # Simplified loop for brevity
        return ll_pop[0], np.array([])


if __name__ == "__main__":
    # This test module would need GPy installed: pip install GPy
    print("--- Verifying SACE-ES Algorithm v3.0 ---")
    try:
        test_problem = get_smd_problem("SMD1", p=2, q=2, r=2)  # Slightly larger problem
        base_config = {"initial_samples": 20, "generations": 50, "ul_mu": 10, "ul_lambda": 20}

        config_mogp = {**base_config, "surrogate_config": {"strategy": "mogp"}}

        print("\n" + "=" * 50)
        print(f"RUNNING STRATEGY: MOGP (GPy)")
        print("=" * 50)

        optimizer = SACE_ES_v3(test_problem, config_mogp)
        results = optimizer.solve()
        print("\n--- Optimization Finished ---")
        print(f"Final UL Fitness: {results['final_ul_fitness']:.4f}")

    except ImportError:
        print("\nERROR: GPy library not found. Please install it using 'pip install GPy' to run this test.")
    except Exception as e:
        print(f"\nAn error occurred during the test run: {e}")
