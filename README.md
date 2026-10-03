# Missingness Shift

Reproducible experiments for **Neural Learning under Missingness Shift**, by Cemil Emre Yavas (2026).

This benchmark tests whether regression predictors and source-calibrated intervals remain reliable when the process determining which inputs are observed changes. Its paired intervention reverses outcome-dependent selection while holding each feature's missing count fixed. Two source policies distinguish training-policy effects from the intrinsic difficulty of a target intervention.

## Contents

- `src/`: data preparation, model fitting, metric aggregation, paired analysis, and clean reproduction.
- `data/`: six public regression tables, prepared numerical arrays, one synthetic dataset, and a provenance manifest.
- `config/protocol.json`: the fixed experimental settings and metric definitions.
- `results/`: 70 environment records, predictions and masks, 8,190 evaluation rows, selected hyperparameters, and summary statistics.
- `tests/test_protocol.py`: checks of mask counts, selection direction, network behavior, and the analytical risk identity.
- `licenses/` and `DATA_LICENSES.md`: data attribution and third-party notices.

## Installation

The reference runs used Python 3.12.1 on an Apple M3 Pro with macOS, CPU execution, and six independent workers. Create an isolated Python 3.12 environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the results

Check the protocol and validate the archived summaries:

```bash
python tests/test_protocol.py
python src/run_benchmark.py --aggregate-only
python src/analyze.py
```

Run a small reproduction test: one Energy dataset environment with all nine methods. This performs fresh fitting and compares all 132 stored arrays against the reference:

```bash
python src/reproduce.py --quick --workers 1 --output reproduced_quick
```

Run all 70 environments, including model selection, in six parallel CPU workers:

```bash
python src/reproduce.py --workers 6 --output reproduced_full
```

Each command requires a fresh output directory. Original predictions remain in `results/`; fresh predictions, metrics, and a comparison report are saved in the selected output directory. The full comparison checks 9,240 arrays. Masks and split indices must match exactly. Floating-point predictions use an absolute tolerance of 0.000001; arithmetic can vary across hardware and numerical-library versions. The pinned environment provides the reference for exact comparisons.

The data files are included. To download the public tables again and regenerate the deterministic synthetic sample:

```bash
python src/prepare_data.py
```

Verify archived file integrity before changing generated outputs:

```bash
python src/verify_checksums.py
```

## Experimental protocol

Seven datasets × five seeds × two source policies × nine selected pipelines × thirteen target interventions give 8,190 evaluation rows. The six public datasets form the primary macro-average; the synthetic experiment is summarized separately. The first numerical feature is always observed. Missingness rates refer to the remaining features.

Data are split into training (55%), model-selection validation (15%), interval calibration (15%), and test (the remainder). Input normalization is fitted to observed training cells. Hyperparameter selection and early stopping use only the source validation split. Source calibration responses determine the absolute-residual interval radius. Target responses are used by the controlled missingness simulator and for evaluation, not by a fitted predictor.

`MCAR` denotes independent source missingness. `Y` denotes logistic, outcome-dependent source missingness. Targets cover an in-distribution draw, three independent missingness rates, observed-covariate selection, self-selection, outcome selection, matched-count reversals, and joint block absence. The matched-count targets use common random numbers and fix counts per feature; they are stress tests and do not meet the independent-row assumptions of ordinary split-conformal coverage guarantees.

The methods are ridge regression with indicators, extra trees with indicators, native-missingness histogram gradient boosting, mean-imputed MLP, indicator-augmented MLP, additional-erasure MLP, iterative-imputation MLP, NeuMiss, and a NeuMISE-style recursion. The compact neural implementations share a two-layer, 64-unit prediction head. `NeuMISE` is the identifier in data files; the implementation is a controlled shared-weight, shared-BatchNorm adaptation, described as NeuMISE-style in interpretation. All methods use source-only selection. Iterative imputation has a fixed ten-cycle budget and may reach that budget without convergence.

## Output format and analysis

An environment file is named `dataset__source__seed.json`. It records selected settings, split sizes, a split hash, runtime, and the metrics. The corresponding `.npz` file contains:

- `METHOD__SCENARIO`: test predictions in training-target-standard-deviation units;
- `y_test` and `test_indices`: target values on that scale and original row indices;
- `mask__SCENARIO`: Boolean missingness arrays (`True` means absent).

`metrics.csv` reports normalized RMSE, coverage, interval width, and missingness fractions. `summary.csv` averages equally over the six public datasets and five seeds. `paired_contrasts.csv` reports the matched-count squared-error increase and the difference of that increase between the two source policies. Intervals resample datasets, then paired seeds within each sampled dataset (20,000 replicates, seed 6021). `dataset_summary.csv` retains each dataset separately.

## Main interpretation

Under outcome-dependent source training, the indicator MLP's public-dataset average normalized RMSE changes from 0.364 to 1.499 between matched-count selection and reversal; 90% interval coverage changes from 89.9% to 28.8%. All nine pipelines have a positive training-policy–reversal interaction on each public dataset. These results concern controlled interventions on complete public tables and do not establish a universal ranking for naturally incomplete deployments.

## Citation and license

Use `CITATION.cff` for machine-readable author and software metadata. The project implementation and synthetic data are released under the MIT License. Public datasets retain their CC BY 4.0 terms and attribution in `DATA_LICENSES.md`. Algorithm sources are acknowledged in `THIRD_PARTY_NOTICES.md`.
