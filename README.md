# KnowledgeProbe

KnowledgeProbe is a small Python/NumPy library for examining singular-value spectra of numeric matrices. It provides basic spectral summaries and comparisons; it does not load or inspect trained models or determine what they know.

## Status

The repository contains a compact implementation and a pytest suite. Treat it as an experimental numerical utility, not a validated model-evaluation or knowledge-measurement method. The project includes no benchmark results or evidence connecting its metrics to task performance.

## Implemented features

- `singular_spectrum(matrix)` converts a non-empty two-dimensional input to `float64` and returns its singular values in descending order.
- `fit_power_law(spectrum)` fits a straight line to `log(singular value)` against `log(rank)` after discarding non-positive values, returning `(slope, intercept, r_squared)`.
- `is_power_law(spectrum, min_r_squared=0.9)` returns whether that fit's R-squared meets the supplied threshold.
- `behavior_signature(matrix, bins=16)` computes a SHA-256 digest of an integer histogram of the matrix's positive singular values in log10 space. The histogram uses the fixed range `[-12, 0]`; this is a compact spectral fingerprint, not a unique identifier for a matrix or model.
- `spectral_distance(matrix_a, matrix_b, k=8)` returns the L2 distance between the base-10 logs of the two matrices' top `k` singular values, divided by `sqrt(k)`.

These functions accept numeric matrices or spectra as appropriate; the module does not extract activations, load model weights, or provide a model API or command-line interface.

## Requirements

- Python 3.9 or newer
- NumPy 1.24 or newer
- pytest 7.4 or newer to run the tests

Install the project and its runtime dependency from the repository root:

```bash
python -m pip install .
```

To install the test dependency as well:

```bash
python -m pip install -r requirements.txt
```

## Example

```python
import numpy as np
from knowledgeprobe import (
    behavior_signature,
    fit_power_law,
    is_power_law,
    singular_spectrum,
    spectral_distance,
)

matrix = np.array([[4.0, 0.0], [0.0, 1.0]])
spectrum = singular_spectrum(matrix)
print(spectrum)  # [4. 1.]

# fit_power_law requires at least three positive spectrum values.
print(behavior_signature(matrix))
print(spectral_distance(matrix, matrix))  # 0.0
```

For `fit_power_law` and `is_power_law`, supply a spectrum with at least three positive values (for example, one obtained from a sufficiently large matrix):

```python
spectrum = singular_spectrum(np.diag([8.0, 4.0, 2.0, 1.0]))
slope, intercept, r_squared = fit_power_law(spectrum)
print(slope, intercept, r_squared)
print(is_power_law(spectrum))
```

## Tests and CI

On the cloned repository commit, `python -m pytest -q` completed with **12 passed**. The repository's CI workflow also specifies pytest, byte-compilation, and an import/spectrum smoke check for Python 3.10, 3.11, and 3.12. The compile and smoke checks were run locally and passed; no hosted GitHub Actions run status is claimed here.

Run the tests locally with:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Limitations

- A matching spectrum or histogram does not imply that matrices, models, or their learned representations are equivalent. The histogram signature is lossy and can collide.
- `spectral_distance` compares only leading singular values and is a numerical matrix comparison, not a semantic or task-level measure. It is intended for compatible matrix shapes and may not work when the compared spectra have different lengths.
- The power-law fit is an ordinary log-log line fit. Its slope and R-squared are descriptive for the supplied spectrum; no statistical validation or relationship to model capability is established here.
- The implementation computes a dense singular-value decomposition and includes no approximation, scalability study, or performance benchmark.
- There is no activation extraction, model-loading/training integration, dataset evaluation, or command-line application.

## License

The repository's `LICENSE` file contains the MIT License (copyright notice: 2026 Kapildhinwa2010-dev). See [LICENSE](LICENSE) for the full terms.
