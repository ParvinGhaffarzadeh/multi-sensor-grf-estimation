# Contributing

Thank you for improving the multi-sensor GRF estimation project. Contributions are welcome for model architecture, training and evaluation, XAI workflows, visualization, documentation, packaging, and tests.

## Before opening an issue

- Search existing issues and pull requests.
- Use the most specific issue template available.
- Do not open public issues for security vulnerabilities, leaked secrets, or sensitive participant data concerns; follow [SECURITY.md](SECURITY.md).
- Include dataset sources, licenses, and references when proposing data, models, metrics, or experiments.

## Development setup

```bash
python -m venv .venv
```

On Windows, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with:

```bash
source .venv/bin/activate
```

Then install the project and development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Local checks

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
python -m unittest discover tests
python -m build
```

Use focused tests for small changes and broader tests when changing shared behavior. If you cannot run a check locally, mention that in the pull request.

## Code guidelines

- Keep public APIs backward compatible where practical.
- Add docstrings for public functions and include expected input types.
- Prefer established PyTorch, NumPy, pandas, SciPy, and scikit-learn APIs over custom parsing when the project needs them.
- Avoid committing generated checkpoints, full datasets, private participant data, or large result artifacts.
- Keep notebooks reproducible and move reusable logic into the package when possible.

## Data and model guidelines

- Include the original source, version, license, and citation for datasets.
- Do not add personal, sensitive, confidential, or unlawfully obtained participant data.
- Document preprocessing assumptions, sensor channels, sampling rates, trial filtering, and train/test split strategy.
- Keep clinical and biomechanical claims aligned with validated results and documented limitations.

## Pull requests

- Keep pull requests focused on one topic.
- Link related issues.
- Update documentation and tests with behavior changes.
- Describe validation commands and any known limitations.
