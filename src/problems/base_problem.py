#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 14:49:32 2025

@author: sanup
"""

# sace_project/src/problems/base_problem.py

from abc import ABC, abstractmethod
import numpy as np


class BaseBilevelProblem(ABC):
    """
    Universal Abstract Base Class for all bilevel optimization problems.

    This class defines the common interface that all problem suites must follow,
    ensuring they are compatible with all implemented algorithms.
    """

    def __init__(self, ul_dim, ll_dim, ul_bounds, ll_bounds, name):
        self.name = name
        self.ul_dim = ul_dim
        self.ll_dim = ll_dim
        self.ul_bounds = np.array(ul_bounds)
        self.ll_bounds = np.array(ll_bounds)
        self.num_ul_constraints = 0
        self.num_ll_constraints = 0

    @abstractmethod
    def evaluate(self, ul_vars, ll_vars, add_penalty=True):
        """
        Evaluates the upper and lower level objectives for a given set of variables.
        Must be implemented by all subclasses.
        """
        pass

    def evaluate_ll_gradient(self, ul_vars, ll_vars):
        """
        Evaluates the gradient of the lower-level objective.
        Optional, but required by some algorithms (e.g., BBOA_KKT, GBSA).
        """
        raise NotImplementedError(f"LL gradient not implemented for {self.name}.")

    def evaluate_ul_constraints(self, ul_vars, ll_vars):
        return np.array([])

    def evaluate_ll_constraints(self, ul_vars, ll_vars):
        return np.array([])

    def __repr__(self):
        return f"{self.name}(UL={self.ul_dim}, LL={self.ll_dim})"
