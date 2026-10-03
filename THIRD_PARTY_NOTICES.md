# Algorithm acknowledgments

NeuMiss follows the shared-weight missingness-dependent recursion introduced by M. Le Morvan, N. Prost, J. Josse, E. Scornet, and G. Varoquaux, “NeuMiss networks: differentiable programming for supervised learning with missing values,” NeurIPS 2020. Original reference implementation: https://github.com/marineLM/NeuMiss.

The NeuMISE-style implementation follows the imputation-like recursion in Rockenschaub, Patrick and Xian, Zhicong and Zamanian, Alireza and Piperno, Marta and Ciora, Octavia-Andreea and Pachl, Elisabeth and Ahmidi, Narges, “Robust prediction under missingness shifts” (2024), https://doi.org/10.48550/arXiv.2406.16484.

The project uses its own compact implementations with a common prediction head and fixed tuning budget. Ridge regression, extra trees, histogram gradient boosting, and iterative imputation are provided by scikit-learn. Neural networks and optimization use PyTorch. Dependencies retain their respective licenses.
