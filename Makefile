.PHONY: install predict backtest clean help

help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies"
	@echo "  make predict   - Run prediction for next race"
	@echo "  make backtest  - Run backtesting"
	@echo "  make clean     - Remove cache files"

install:
	pip install -r requirements.txt

predict:
	python main.py --round next --html

backtest:
	python main.py --backtest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf .cache/
