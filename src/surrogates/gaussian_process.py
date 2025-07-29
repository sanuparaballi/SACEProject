#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:22:02 2025

@author: sanup
"""

# sace_project/src/surrogates/gaussian_process.py

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GaussianProcessSurrogate:
    """
    A surrogate model based on Gaussian Process Regression (GPR).

    This class wraps the scikit-learn GPR implementation to provide a simple
    interface for training the model, making predictions, and assessing
    prediction uncertainty, which is essential for acquisition functions like
    Expected Improvement.
    """

    def __init__(self, kernel=None):
        """
        Initializes the Gaussian Process surrogate model.

        Args:
            kernel: A scikit-learn kernel object. If None, a default RBF kernel
                    is used, which is a common choice for smooth functions.
        """
        if kernel is None:
            # Default kernel: Radial Basis Function (RBF) for smoothness.
            # The length_scale parameter determines the 'reach' of the influence
            # of each training point.
            self.kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-5, 1e3))
        else:
            self.kernel = kernel

        # The regressor object from scikit-learn.
        # `n_restarts_optimizer` helps in finding a better set of kernel
        # hyperparameters by restarting the optimization multiple times.
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=10,
            alpha=1e-2,  # A small value for regularization
            normalize_y=True,  # Normalizing the target values can improve stability
        )

        # Flag to check if the model has been trained
        self.is_trained = False

    def train(self, X, y):
        """
        Trains the Gaussian Process model on the given data.

        Args:
            X (np.array): An array of input samples, shape (n_samples, n_features).
            y (np.array): An array of target values, shape (n_samples,).
        """
        if len(X) == 0 or len(y) == 0:
            print("Warning: Attempted to train surrogate with no data.")
            return

        try:
            self.model.fit(X, y)
            self.is_trained = True
        except Exception as e:
            print(f"Error during Gaussian Process training: {e}")
            self.is_trained = False

    def predict(self, X):
        """
        Makes predictions for new input samples.

        Args:
            X (np.array): An array of input samples for which to predict,
                          shape (n_samples, n_features).

        Returns:
            A tuple (mean, std) where:
                - mean (np.array): The predicted mean values.
                - std (np.array): The standard deviation of the predictions,
                                  representing uncertainty.
        """
        if not self.is_trained:
            # If the model isn't trained, return non-informative predictions
            # Return a high value for mean and std to discourage selection.
            num_samples = X.shape[0] if len(X.shape) > 1 else 1
            return np.full(num_samples, np.inf), np.full(num_samples, np.inf)

        # Reshape if a single sample is passed
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        mean, std = self.model.predict(X, return_std=True)
        return mean, std


# =============================================================================
# Test Module
# =============================================================================
if __name__ == "__main__":
    print("--- Verifying GaussianProcessSurrogate Implementation ---")

    # 1. Create a simple 1D test function to approximate (e.g., a sine wave)
    def test_function(x):
        return x * np.sin(x)

    # 2. Generate some training data from the function
    X_train = np.array([1.0, 3.0, 5.0, 6.0, 7.0, 8.0]).reshape(-1, 1)
    y_train = test_function(X_train).ravel()

    print(f"\nTraining data generated for function f(x) = x*sin(x).")
    print(f"  Training X points: {X_train.ravel()}")
    print(f"  Training y values: {np.round(y_train, 2)}")

    # 3. Instantiate and train the surrogate model
    print("\nInstantiating and training the surrogate model...")
    surrogate = GaussianProcessSurrogate()
    surrogate.train(X_train, y_train)

    if surrogate.is_trained:
        print("  Model training successful.")

        # 4. Generate points for prediction and get model output
        X_test = np.atleast_2d(np.linspace(0, 10, 100)).T
        y_pred_mean, y_pred_std = surrogate.predict(X_test)

        # 5. Check if predictions are reasonable
        # A simple check: does the prediction at a training point match the training value?
        test_point_idx = 2
        test_point = X_train[test_point_idx].reshape(1, -1)
        pred_mean_at_train_point, _ = surrogate.predict(test_point)

        print("\nVerifying prediction at a known training point...")
        print(f"  Test point X: {test_point.item():.1f}")
        print(f"  True y value: {y_train[test_point_idx]:.4f}")
        print(f"  Predicted mean: {pred_mean_at_train_point.item():.4f}")

        assert np.isclose(
            y_train[test_point_idx], pred_mean_at_train_point, atol=1e-4
        ), "Prediction at training point is not accurate!"
        print("  Assertion PASSED: Prediction at a training point is accurate.")

        # This part is for visualization if needed, requires matplotlib
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(X_test, test_function(X_test), "r:", label="True Function: $f(x) = x\\sin(x)$")
            plt.plot(X_train, y_train, "r.", markersize=10, label="Observations")
            plt.plot(X_test, y_pred_mean, "b-", label="GP Prediction (Mean)")
            plt.fill(
                np.concatenate([X_test, X_test[::-1]]),
                np.concatenate([y_pred_mean - 1.96 * y_pred_std, (y_pred_mean + 1.96 * y_pred_std)[::-1]]),
                alpha=0.2,
                fc="b",
                ec="None",
                label="95% confidence interval",
            )
            plt.xlabel("$x$")
            plt.ylabel("$f(x)$")
            plt.legend(loc="upper left")
            plt.title("Gaussian Process Surrogate Model Verification")
            plt.grid(True)
            print("\nPlotting verification graph (requires matplotlib)...")
            # plt.show() # Commented out to prevent blocking test runs
            print("  Plot generated successfully (display disabled in script).")
        except ImportError:
            print("\nSkipping plot generation: matplotlib is not installed.")

    else:
        print("  Model training failed. Verification skipped.")

    print("\n--- Verification Complete ---")
