## Testing & Automation Guide

This document explains the end-to-end testing and automation setup for `AI_Hallucination_Detection_and_Correction` from an SDET perspective.

### Overview
- **Unit tests**: Validate functions in `retrieval`, `detection`, `correction` with full mocking of external services.
- **Integration tests**: Exercise retrieval→detection→correction flow with controlled mocks.
- **E2E tests**: Simulate a user scenario end-to-end with hermetic mocks and produce an HTML report.

### Tools
- **Pytest** for testing
- **pytest-mock** for mocking
- **pytest-html** for HTML report
- **pytest-cov** for coverage
- **flake8** for linting

### Project layout for tests
```
tests/
  unit/
  integration/
  e2e/
  conftest.py
  requirements.txt
pytest.ini
.flake8
```

### Install
```
pip install -r requirements.txt
pip install -r tests/requirements.txt
```

### Run locally
- All tests with coverage and HTML report:
```
pytest -v --cov=src --html=report.html --self-contained-html
```
- Only unit tests:
```
pytest -v -m unit
```
- Only integration tests:
```
pytest -v -m integration
```
- Only E2E tests:
```
pytest -v -m e2e --html=e2e_report.html --self-contained-html
```

### Linting
```
flake8
```

### CI/CD
GitHub Actions workflow at `.github/workflows/tests.yml`:
- Checks out repo
- Installs dependencies
- Runs pytest with coverage
- Uploads coverage to Codecov (optional)
- Publishes HTML report as an artifact

Add badges to `README.md`:
```
![Build](https://github.com/Abhijeet-4402/AI_Hallucination_Detection_and_Correction/actions/workflows/tests.yml/badge.svg)
```
Optionally add Codecov badge once set up.

### Notes
- External services (Wikipedia, ChromaDB, LLMs, NLTK models) are mocked in tests for speed and determinism.
- `conftest.py` ensures `src` is on `sys.path` and provides reusable fixtures.


