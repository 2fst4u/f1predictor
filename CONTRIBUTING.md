# Contributing to F1 Predictor

Thank you for your interest in contributing to F1 Predictor!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/f1predictor.git`
3. Create a virtual environment: `python -m venv .venv`
4. Activate it:
   - Windows: `.venv\Scripts\activate`
   - Linux/macOS: `source .venv/bin/activate`
5. Install dependencies: `pip install -r requirements.txt`

## Making Changes

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Test that the tool still works: `python main.py --round next`
4. Commit your changes with a clear message
5. Push to your fork
6. Open a Pull Request

## Code Standards

- Follow PEP 8 style guidelines where possible
- Use type hints for function signatures
- Keep functions focused and readable
- Add docstrings for complex functions
- Maximum line length: 120 characters

## Testing Changes

Before submitting, ensure:
- The tool runs without errors
- Predictions are generated successfully
- No existing functionality is broken

## Questions?

Open an issue for discussion before starting major changes.

## Code of Conduct

Be respectful, constructive, and collaborative.
