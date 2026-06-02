# Contributing to DGSA

Thank you for your interest in contributing to this project. This document outlines the project architecture, coding style, testing requirements, and the pull request process.

---

## Project Architecture

```plaintext
src/dgsa/
├── computation/       # Core sensitivity analysis algorithms
│   ├── kmedoids.py                          # K-medoids clustering
│   ├── single_parameter_sensitivity.py      # Single-parameter sensitivity (l1norm and ASL)
│   └── conditional_parameter_sensitivity.py # Conditional (two-way) sensitivity
├── utils/
│   └── dgsa_save_load.py                    # Save and load DGSA results
└── visualization/     # Plotting functions
    ├── cluster_mds.py         # MDS cluster plot
    ├── single_cdf.py          # CDF plots for single sensitivity
    ├── single_pareto.py       # Pareto plots for single sensitivity
    ├── conditional_cdf.py     # CDF plots for conditional sensitivity
    ├── conditional_heatmap.py # Heatmap for conditional sensitivity
    └── conditional_pareto.py  # Pareto plots for conditional sensitivity
```

**Key design principles:**
- `computation/` functions return plain Python `dict` objects — no custom classes.
- `visualization/` functions take the output dicts from `computation/` directly as inputs.
- There are no dependencies between `visualization/` and `computation/` modules — they are kept separate intentionally.

---

## Coding Style

- **Python version:** >= 3.10
- **Type hints:** required for all function signatures using `NDArray[np.float64]`, `dict`, `int | None`, etc.
- **Docstrings:** NumPy docstring style with `Parameters`, `Returns` sections. See existing functions for reference.
- **Naming conventions:**
  - Functions and variables: `snake_case`
  - No abbreviations unless widely established (e.g. `ASL`, `MDS`, `CDF`)
- **Copyright header:** add the following two lines at the top of every new source file:
  ```python
  # Copyright (c) 2026 Stanford Mineral-X
  # Licensed under the MIT License — see LICENSE file in the root of this repository for details.
  ```
- **No unused imports.** Keep imports minimal and grouped (standard library → third-party).

---

## Testing Requirements

Tests live in `tests/` and use `pytest`.

- **Run all tests before submitting a PR:**
  ```bash
  pytest -v
  ```
- **All 13 tests must pass** with no new failures introduced.
- **Adding new functionality:** add a corresponding test in the appropriate file:
  - `tests/test_clustering.py` — clustering logic
  - `tests/test_single_sensitivity.py` — single parameter sensitivity
  - `tests/test_conditional_sensitivity.py` — conditional parameter sensitivity
- **Reference fixtures** are stored in `tests/fixtures/`. If you change output format or computation logic, update the fixtures accordingly and document why in your PR.
- **Tolerance:** numerical tests use `rtol=0.10` for sensitivity values due to bootstrap variance. Do not tighten this without justification.

---

## Pull Request Process

1. **Fork** the repository and create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes**, following the coding style above.
3. **Run tests** and confirm all pass:
   ```bash
   pytest -v
   ```
4. **Submit a PR** against the `main` branch with:
   - A clear title describing the change
   - A short description of what was changed and why
   - Reference to any related issues if applicable
5. **PRs will be reviewed** for correctness, style, and test coverage before merging.

---

## Questions

Open an issue or contact [jihuid@stanford.edu](mailto:jihuid@stanford.edu).
