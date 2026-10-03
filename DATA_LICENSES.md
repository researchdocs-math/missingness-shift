# Dataset attribution

The public datasets are distributed by the UCI Machine Learning Repository under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/). The prepared numerical arrays are adaptations of these records. Original files and their SHA-256 hashes are listed in `data/manifest.json`.

| Dataset | Creators and year | Dataset DOI | Preparation |
|---|---|---|---|
| Wine Quality, red and white | P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis (2009) | https://doi.org/10.24432/C56S3T | Separate red and white tables; quality is the target. |
| Concrete Compressive Strength | I-Cheng Yeh (1998) | https://doi.org/10.24432/C5PK67 | Eight inputs; compressive strength is the target. |
| Energy Efficiency | A. Tsanas, A. Xifara (2012) | https://doi.org/10.24432/C51307 | Eight inputs; heating load is the target; cooling load is excluded. |
| Airfoil Self-Noise | T. Brooks, D. Pope, M. Marcolini (1989) | https://doi.org/10.24432/C5VW2C | Five inputs; scaled sound-pressure level is the target. |
| Abalone | W. Nash, T. Sellers, S. Talbot, A. Cawthorn, W. Ford (1994) | https://doi.org/10.24432/C55C7W | Seven numerical measurements; sex is excluded; rings is the target. |

All prepared features and responses are stored as floating-point arrays. Missingness is generated during experiments, not added to the archived complete input tables. The synthetic dataset and its generation code are original project material under the MIT License.
