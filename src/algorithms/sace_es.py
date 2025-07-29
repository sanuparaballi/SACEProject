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

        # The most robust way to initialize GPCoregionalizedRegression is to
        # format the Y data as a list of column vectors. Each element in the
        # list represents the observations for one output dimension.
        # This correctly informs the model of the number of outputs.
        Y_list = [Y[:, i : i + 1] for i in range(self.problem_dim)]

        # Reshape data for GPy's Coregionalized model
        # X_augmented has a new column for the output index
        # X_augmented = np.hstack([X for _ in range(self.problem_dim)])
        # X_augmented = X_augmented.reshape(-1, input_dim)
        # index_col = np.array([i for i in range(self.problem_dim) for _ in range(n_samples)]).reshape(-1, 1)
        # X_augmented = np.hstack([X_augmented, index_col])

        # # Y is flattened
        # Y_flattened = Y.T.reshape(-1, 1)

        # Explicitly tell each kernel which columns of the augmented data to use.
        # The RBF kernel should only use the decision variable columns (0 to input_dim-1).
        # The Coregionalize kernel should only use the final index column (input_dim).
        k_rbf = GPy.kern.RBF(input_dim=input_dim, active_dims=list(range(input_dim)), ARD=True)
        k_coreg = GPy.kern.Coregionalize(
            input_dim=1, output_dim=self.problem_dim, rank=1, active_dims=[input_dim]
        )

        # Constrain kappa to be positive to avoid numerical issues
        k_coreg.kappa.constrain_positive()

        self.kernel = k_rbf * k_coreg

        # Create and optimize the GPy model
        self.model = GPy.models.GPCoregionalizedRegression(
            X_list=[X] * self.problem_dim, Y_list=Y_list, kernel=self.kernel
        )
        # self.model = GPy.models.GPCoregionalizedRegression(
        #     X_list=[X_augmented], Y_list=[Y_flattened], kernel=self.kernel, output_dim=self.problem_dim
        # )
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
    """
    Variant 3: Full implementation for a Heteroscedastic GP model.
    This model learns a separate GP for the mean and for the noise variance
    of each output dimension.
    """

    def __init__(self, problem_dim):
        super().__init__(problem_dim)
        self.mean_models = [GaussianProcessSurrogate() for _ in range(problem_dim)]
        self.noise_models = [GaussianProcessSurrogate() for _ in range(problem_dim)]

    def train(self, X, Y):
        # Stage 1: Train the mean models on the original data
        for i in range(self.problem_dim):
            self.mean_models[i].train(X, Y[:, i])

        # Stage 2: Model the noise (variance)
        # Predict the mean for the training data to find residuals
        mean_preds, _ = self._predict_mean(X)

        # Calculate squared residuals (proxy for noise variance)
        squared_residuals = (Y - mean_preds) ** 2

        # Train the noise models on the log of the squared residuals for stability
        for i in range(self.problem_dim):
            # Adding a small epsilon to avoid log(0)
            self.noise_models[i].train(X, np.log(squared_residuals[:, i] + 1e-9))

        self.is_trained = True

    def _predict_mean(self, X):
        """Internal method to predict only from the mean models."""
        means = np.array([m.predict(X)[0] for m in self.mean_models]).T
        variances = np.array([m.predict(X)[1] for m in self.mean_models]).T
        return means, variances

    def predict(self, X):
        """
        Predicts the mean and total variance. Total variance is the sum of
        the model's uncertainty and the predicted data noise.
        """
        if not self.is_trained:
            # Fallback for initial state before training
            return np.zeros((X.shape[0], self.problem_dim)), np.ones((X.shape[0], self.problem_dim))

        # Predict mean and the GP model's own uncertainty
        pred_mean, model_variance = self._predict_mean(X)

        # Predict the log of the data noise variance
        pred_log_noise_var = np.array([m.predict(X)[0] for m in self.noise_models]).T

        # Convert back to linear scale
        pred_noise_variance = np.exp(pred_log_noise_var)

        # Total variance = model uncertainty + predicted data noise
        total_variance = model_variance + pred_noise_variance

        return pred_mean, total_variance


# =============================================================================
# SACE-ES Algorithm v3.0
# =============================================================================


class SACE_ES(BaseOptimizer):
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
        self.ll_pop_size = self.config.get("ll_pop_size", 30)
        self.generations = self.config.get("generations", 100)
        self.ll_generations = self.config.get("ll_generations", 20)
        self.initial_samples = self.config.get("initial_samples", 20)
        self.kappa = self.config.get("lcb_kappa", 2.0)
        self.tau = 1.0 / np.sqrt(2 * self.problem.ul_dim)
        self.tau_prime = 1.0 / np.sqrt(2 * np.sqrt(self.problem.ul_dim))

        surrogate_config = self.config.get("surrogate_config", {"strategy": "independent"})
        self.strategy = surrogate_config["strategy"]
        surrogate_map = {
            "independent": IndependentGPs,
            "mogp": MOGP_GPy,
            "heteroscedastic": HeteroscedasticGPs,
        }
        self.surrogates_y = surrogate_map[self.strategy](self.problem.ll_dim)
        self.surrogates_c = (
            IndependentGPs(self.problem.num_ll_constraints) if self.problem.num_ll_constraints > 0 else None
        )
        self.archive_ul, self.archive_ll_sols, self.archive_ll_cons = [], [], []

    def _initialize_population(self):
        pop_x = np.random.uniform(
            self.problem.ul_bounds[:, 0], self.problem.ul_bounds[:, 1], size=(self.mu, self.problem.ul_dim)
        )
        pop_sigma = np.random.uniform(0.1, 0.5, size=(self.mu, self.problem.ul_dim))
        return np.hstack([pop_x, pop_sigma])

    def _recombine(self, pop):
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
            n_prime = np.random.randn()
            n_i = np.random.randn(self.problem.ul_dim)
            sigma_prime = sigma * np.exp(self.tau_prime * n_prime + self.tau * n_i)
            x_prime = x + sigma_prime * np.random.randn(self.problem.ul_dim)
            x_prime = np.clip(x_prime, self.problem.ul_bounds[:, 0], self.problem.ul_bounds[:, 1])
            mutated_pop.append(np.concatenate([x_prime, sigma_prime]))
        return np.array(mutated_pop)

    def _get_surrogate_fitness(self, pop):
        pop_x = pop[:, : self.problem.ul_dim]
        pred_ll_sols, _ = self.surrogates_y.predict(pop_x)
        return np.array(
            [
                self.problem.evaluate(pop_x[i], pred_ll_sols[i], add_penalty=False)[0]
                for i in range(len(pop_x))
            ]
        )

    def _run_lower_level_es(self, ul_individual):
        """
        Full implementation of the lower-level Evolutionary Strategy.
        This function finds the optimal follower response y* for a given leader's decision x.
        """
        # 1. Initialize LL population
        ll_pop = np.random.uniform(
            self.problem.ll_bounds[:, 0],
            self.problem.ll_bounds[:, 1],
            size=(self.ll_pop_size, self.problem.ll_dim),
        )

        # 2. Evaluate initial LL population fitness (with penalties for constraints)
        fitness_ll = np.array(
            [self.problem.evaluate(ul_individual, ll_ind, add_penalty=True)[1] for ll_ind in ll_pop]
        )
        self.ll_nfe += self.ll_pop_size

        # 3. Main ES loop for the lower level
        for _ in range(self.ll_generations):
            # a. Create offspring: Simple (μ,λ) selection and mutation
            parents = ll_pop[np.random.choice(len(ll_pop), self.ll_pop_size, replace=True)]
            offspring_ll = parents + np.random.normal(0, 0.2, size=parents.shape)  # Simple mutation
            offspring_ll = np.clip(offspring_ll, self.problem.ll_bounds[:, 0], self.problem.ll_bounds[:, 1])

            # b. Evaluate offspring
            offspring_fitness = np.array(
                [self.problem.evaluate(ul_individual, off, add_penalty=True)[1] for off in offspring_ll]
            )
            self.ll_nfe += self.ll_pop_size

            # c. Selection: Combine parents and offspring and select the best
            combined_pop = np.vstack([ll_pop, offspring_ll])
            combined_fitness = np.concatenate([fitness_ll, offspring_fitness])

            best_indices = np.argsort(combined_fitness)[: self.ll_pop_size]
            ll_pop = combined_pop[best_indices]
            fitness_ll = combined_fitness[best_indices]

        # 4. Return the best solution found and its true constraint violation
        best_ll_solution = ll_pop[0]
        ll_constraint_violations = self.problem.evaluate_ll_constraints(ul_individual, best_ll_solution)

        return best_ll_solution, ll_constraint_violations

    def solve(self):
        print(f"Phase 1: Initial sampling for {self.strategy}")
        initial_ul_samples = np.random.uniform(
            self.problem.ul_bounds[:, 0],
            self.problem.ul_bounds[:, 1],
            size=(self.initial_samples, self.problem.ul_dim),
        )
        for ul_sample in initial_ul_samples:
            ll_opt, ll_cons = self._run_lower_level_es(ul_sample)
            self.archive_ul.append(ul_sample)
            self.archive_ll_sols.append(ll_opt)
            if self.surrogates_c:
                self.archive_ll_cons.append(ll_cons)

        self.archive_ul = np.array(self.archive_ul)
        self.archive_ll_sols = np.array(self.archive_ll_sols)
        self.surrogates_y.train(self.archive_ul, self.archive_ll_sols)
        if self.surrogates_c:
            self.archive_ll_cons = np.array(self.archive_ll_cons)
            self.surrogates_c.train(self.archive_ul, self.archive_ll_cons)
        print("Surrogate models trained.")

        print("Phase 2: Starting optimization with Self-Adaptive ES...")
        ul_pop = self._initialize_population()

        for gen in range(self.generations):
            recombined_pop = self._recombine(ul_pop)
            offspring_ul_pop = self._self_adaptive_mutation(recombined_pop)

            offspring_x = offspring_ul_pop[:, : self.problem.ul_dim]
            pred_ll_sols, pred_ll_vars = self.surrogates_y.predict(offspring_x)
            pred_ul_fitness = np.array(
                [self.problem.evaluate(offspring_x[i], pred_ll_sols[i])[0] for i in range(len(offspring_x))]
            )
            self.ul_nfe += len(offspring_x)
            lcb_scores = pred_ul_fitness - self.kappa * np.mean(np.sqrt(pred_ll_vars + 1e-6), axis=1)

            infill_idx = np.argmin(lcb_scores)
            promising_ul = offspring_x[infill_idx]

            exact_ll_sol, exact_ll_cons = self._run_lower_level_es(promising_ul)
            self.archive_ul = np.vstack([self.archive_ul, promising_ul])
            self.archive_ll_sols = np.vstack([self.archive_ll_sols, exact_ll_sol])
            if self.surrogates_c:
                self.archive_ll_cons = np.vstack([self.archive_ll_cons, exact_ll_cons])

            self.surrogates_y.train(self.archive_ul, self.archive_ll_sols)
            if self.surrogates_c:
                self.surrogates_c.train(self.archive_ul, self.archive_ll_cons)

            combined_pop = np.vstack([ul_pop, offspring_ul_pop])
            fitness_values = self._get_surrogate_fitness(combined_pop)
            best_indices = np.argsort(fitness_values)[: self.mu]
            ul_pop = combined_pop[best_indices]

            best_archive_fitness = np.min(
                [
                    self.problem.evaluate(self.archive_ul[i], self.archive_ll_sols[i])[0]
                    for i in range(len(self.archive_ul))
                ]
            )
            self.ul_nfe += len(self.archive_ul)
            self.log_generation(gen, best_archive_fitness, np.mean(fitness_values))

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

        optimizer = SACE_ES(test_problem, config_mogp)
        results = optimizer.solve()
        print("\n--- Optimization Finished ---")
        print(f"Final UL Fitness: {results['final_ul_fitness']:.4f}")

    except ImportError:
        print("\nERROR: GPy library not found. Please install it using 'pip install GPy' to run this test.")
    except Exception as e:
        print(f"\nAn error occurred during the test run: {e}")
