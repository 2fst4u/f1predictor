# Contributing to F1 Predictor

Thank you for your interest in contributing to F1 Predictor!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/f1predictor.git`
3. Create a virtual environment: `python -m venv .venv`
4. Activate it:
   - Windows: `.venv\Scripts\activate`
   - Linux/macOS: `source .venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt pytest pytest-cov httpx`

## Making Changes

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Test that the tool still works: `python main.py --round next`
4. Run the test suite: `make test`
5. Commit your changes with a clear message
6. Push to your fork
7. Open a Pull Request

## Code Standards

- Follow PEP 8 style guidelines where possible
- Use type hints for function signatures
- Keep functions focused and readable
- Add docstrings for complex functions
- Maximum line length: 120 characters

## Testing Changes

Before submitting, ensure:
- The tool runs without errors (`python main.py --round next`)
- All tests pass locally (`make test`)
- Code coverage does not drop below the required threshold

## Releasing

Versioning is automatic — every merge to `main` creates a patch version bump.

For minor or major releases, use the **Release** workflow in GitHub Actions with the `workflow_dispatch` trigger. See [AGENTS.md](AGENTS.md) for full details.

To skip a release, include `[skip release]` in the merge commit message.

## Questions?

Open an issue for discussion before starting major changes.

## Code of Conduct

Be respectful, constructive, and collaborative.
