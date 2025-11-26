# Contributing to F1 Predictor

Thank you for your interest in contributing to F1 Predictor!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/f1predictor.git`
3. Create a virtual environment: `python -m venv .venv`
4. Activate it:
   - Windows: `.venv\Scripts\activate`
   - Linux/macOS: `source .venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt -r requirements-dev.txt`
6. Install pre-commit hooks: `pre-commit install`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest tests/`
4. Run linters: `make lint`
5. Format code: `make format`
6. Commit your changes with a clear message
7. Push to your fork
8. Open a Pull Request

## Code Standards

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write docstrings for all public functions/classes
- Add tests for new functionality
- Keep functions focused and small
- Maximum line length: 120 characters

## Testing

- Write unit tests for all new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage
- Test edge cases and error conditions

## Commit Messages

Use clear, descriptive commit messages:

- `feat: add weather sensitivity feature`
- `fix: correct DNF probability calculation`
- `docs: update README with new examples`
- `test: add tests for roster derivation`
- `refactor: simplify feature engineering logic`

## Pull Request Process

1. Update README.md if needed
2. Add entry to CHANGELOG.md (if exists)
3. Ensure CI passes
4. Request review from maintainers
5. Address review comments
6. Squash commits if requested

## Questions?

Open an issue for discussion before starting major changes.

## Code of Conduct

Be respectful, constructive, and collaborative.
