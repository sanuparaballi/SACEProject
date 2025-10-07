# SACE-ES: A Surrogate-Assisted Co-evolutionary Framework for Bilevel Optimization


This repository contains the official implementation for our paper:
    A Surrogate-Assisted Co-evolutionary Framework for Bilevel Optimization
    
    To be presented at the 17th International Joint Conference on Computational Intelligence (IJCCI), 2025. > Link to Presentation Details

**Overview**

Bilevel optimization, a class of hierarchical optimization problems, presents a significant research challenge due to its inherent NP-hard nature, especially in non-convex settings. In this work, we address the limitations of existing solvers. Classical gradient-based methods are often inapplicable to the non-convex and non-differentiable landscapes common in practice, while derivative-free methods like nested evolutionary algorithms are rendered intractable by a prohibitively high query complexity.

To this end, we propose a novel framework, the Surrogate-Assisted Co-evolutionary Evolutionary Strategy (SACE-ES), which synergizes the global search capabilities of evolutionary computation with the data-driven efficiency of surrogate modeling.

The core innovation of our framework is a multi-surrogate, constraint-aware architecture that decouples the complex bilevel problem. We use separate Gaussian Process (GP) models to approximate the lower-level optimal solution vector and its corresponding constraint violations. This allows our algorithm to make intelligent, cheap evaluations to guide the search, reserving expensive, true evaluations only for the most informative candidate solutions.
## Authors

Sanup Araballi, Venkata Gandikota, Pranay Sharma, Prashant Khanduri, and Chilukuri K Mohan
## Features

**Key Contributions**

    A Novel Framework: We propose SACE-ES, a constraint-aware, multi-surrogate evolutionary strategy for bilevel optimization that efficiently balances exploration and exploitation in nested decision spaces.

    High Efficiency: Our framework consistently finds high-quality solutions with a reduction in computational cost of up to 96% compared to exhaustive search methods like Nested Differential Evolution.

    Superior Performance on Constrained Problems: We provide an extensive empirical demonstration that SACE-ES discovers qualitatively superior solutions on complex constrained problems where other heuristic and classical methods fail.



**Methodology in Brief**

Our framework is built on three architectural pillars:

    A Co-evolutionary Structure: We maintain a single population of upper-level candidate solutions and surrogate models for the lower level, which evolve in tandem to avoid expensive nested loops.

    An Evolutionary Strategy (ES) Core: The optimization is driven by an ES, a powerful gradient-free search heuristic well-suited for continuous decision spaces.

    Surrogate-Assisted Intelligence: We use a multi-surrogate system of Gaussian Processes to build cheap, predictive models of the lower-level outcome. This allows the algorithm to explore the search space quickly and intelligently allocate its computational budget.

For a complete technical description of the algorithm, surrogate models (including Independent, Multi-Output, and Heteroscedastic GPs), and formulations, please refer to our detailed Methodology Document.




**Citing Our Work**

If you use this code or our framework in your research, please cite our IJCNN paper:

@inproceedings{araballi2025surrogate,
  title={A Surrogate-Assisted Co-Evolutionary Framework for Bilevel Optimization},
  author={Araballi, Sanup and Gandikota, Venkata and Sharma, Pranay and Khanduri, Prashant and Mohan, Chilukuri K},
  booktitle={Proceedings of the 17th International Joint Conference on Computational Intelligence (IJCCI)},
  year={2025},
  organization={INSTICC}
}


**License**

This project is licensed under the MIT License. See the LICENSE file for details.
