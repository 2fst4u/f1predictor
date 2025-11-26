.PHONY: install install-dev test lint format clean help

help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make test         - Run tests with coverage"
	@echo "  make lint         - Run linters (ruff, mypy)"
	@echo "  make format       - Format code with ruff"
	@echo "  make clean        - Remove cache and build artifacts"
	@echo "  make predict      - Run prediction for next race"
	@echo "  make backtest     - Run backtesting"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v

lint:
	ruff check f1pred/ tests/
	mypy f1pred/

format:
	ruff format f1pred/ tests/
	ruff check --fix f1pred/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '.coverage' -delete

predict:
	python main.py --round next --html

backtest:
	python main.py --backtest
