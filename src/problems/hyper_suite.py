#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 21:57:42 2025

@author: sanup
"""


# sace_project/src/problems/hyper_suite.py

import numpy as np
from .smd_suite import BilevelProblem


class HyperRepresentation(BilevelProblem):
    """
    Implements the Hyper-representation problem as described in the paper.
    https://openreview.net/pdf?id=W4AZQzNe8h
    PROVABLE FASTER ZEROTH-ORDER METHOD FOR BILEVEL OPTIMIZATION:
        ACHIEVING BEST-KNOWN DEPENDENCY ON ERROR AND DIMENSION
    This is a bilevel optimization problem for finding an optimal data representation.

    - Lower Level: Learns optimal regression weights 'w' for a given representation.
    - Upper Level: Learns the optimal representation matrix 'A' that leads to the
                   best performance on a separate validation set.
    """

    def __init__(self, n=100, m=10, d2=2):
        """
        Initializes the problem.
        Args:
            n (int): Number of data samples.
            m (int): Original feature dimension.
            d2 (int): Dimension of the embedding (follower variables). This is a simplification.
                      In the paper, d2 is the dimension of w, which is m.
                      The leader variable dimension d1 is the size of matrix A (m*d2).
                      Let's stick to the paper's setup: w is d2-dim, A is d1-dim.
                      Let d1 = m*d2. Let's set d2=m for a linear regressor.
        """
        self.n_train, self.n_val = n, n
        self.m_features = m

        d1 = m * m  # Dimension of the leader's variable (matrix A)
        d2 = m  # Dimension of the follower's variable (vector w)

        ul_dim = d1
        ll_dim = d2
        ul_bounds = (-1, 1)  # Bounds for elements of matrix A
        ll_bounds = (-5, 5)  # Bounds for elements of vector w
        super().__init__(ul_dim, ll_dim, ul_bounds, ll_bounds)

        # Generate synthetic data as per the paper's methodology
        np.random.seed(42)
        # We need training data (X2, Y2) and validation data (X1, Y1)

        self.X_train = np.random.randn(self.n_train, self.m_features)
        self.Y_train = np.random.randn(self.n_train)
        self.X_val = np.random.randn(self.n_val, self.m_features)
        self.Y_val = np.random.randn(self.n_val)

        self.lambda_reg = 0.1  # Regularization parameter from the paper

    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        """
        Computes the objective values.

        Args:
            ul_vars (np.array): The leader's variables, representing matrix A (flattened).
            ll_vars (np.array): The follower's variables, representing vector w.
        """
        try:
            # Reshape the flattened leader's variables back into a matrix
            A = ul_vars.reshape((self.m_features, self.m_features))
            w = ll_vars

            # Lower-Level Objective: Minimize regularized MSE on the training set
            # f_LL(w) = ||(X_train * A)w - Y_train||^2 + lambda * ||w||^2
            X_train_embedded = self.X_train @ A
            train_residuals = X_train_embedded @ w - self.Y_train
            ll_objective = np.sum(train_residuals**2) + self.lambda_reg * np.sum(w**2)

            # Upper-Level Objective: Minimize MSE on the validation set
            # F_UL(A) = ||(X_val * A)w* - Y_val||^2  (where w* is the optimal follower response)
            X_val_embedded = self.X_val @ A
            val_residuals = X_val_embedded @ w - self.Y_val
            ul_objective = np.sum(val_residuals**2)

            return ul_objective, ll_objective
        except ValueError as e:
            # Provide more context if reshape fails
            print(
                f"Reshape Error in HyperRepresentation: Could not reshape ul_vars of shape {ul_vars.shape} to ({self.m_features}, {self.m_features})"
            )
            raise e


def get_hyper_problem(name: str, params: dict):
    if name.lower() == "hyper_representation":
        return HyperRepresentation(
            n=params.get("n", 100), m=params.get("m", 10)  # This will determine d1 and d2
        )
    raise ValueError(f"Problem '{name}' not found in hyper suite.")
