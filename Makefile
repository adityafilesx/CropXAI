.PHONY: help setup install install-dev lint format typecheck test test-unit \
        test-integration test-regression test-perf test-dataset \
        validate explore report preprocess clean docs

PYTHON  ?= python
UV      ?= uv
SRC     := src
TESTS   := tests

# ─── Colors ───────────────────────────────────────────────────────────────────
BOLD   := \033[1m
RESET  := \033[0m
GREEN  := \033[32m
YELLOW := \033[33m
BLUE   := \033[34m
CYAN   := \033[36m

help: ## Show this help message
	@echo ""
	@echo "$(BOLD)CropXAI — Research Framework$(RESET)"
	@echo ""
	@echo "$(BOLD)Usage:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ─── Setup ────────────────────────────────────────────────────────────────────
setup: ## Full environment setup (install deps + pre-commit hooks)
	@echo "$(BOLD)$(GREEN)Setting up CropXAI environment...$(RESET)"
	$(UV) pip install -e ".[dev]"
	pre-commit install
	cp -n .env.example .env || true
	@echo "$(BOLD)$(GREEN)✓ Setup complete. Edit .env with your configuration.$(RESET)"

install: ## Install package in editable mode
	$(UV) pip install -e .

install-dev: ## Install package + dev dependencies
	$(UV) pip install -e ".[dev]"

# ─── Code Quality ─────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	@echo "$(BOLD)Running ruff...$(RESET)"
	ruff check $(SRC)/ $(TESTS)/ profiling/

format: ## Format code with black + ruff
	@echo "$(BOLD)Formatting code...$(RESET)"
	black $(SRC)/ $(TESTS)/ profiling/
	ruff check --fix $(SRC)/ $(TESTS)/ profiling/

typecheck: ## Run mypy type checker
	@echo "$(BOLD)Type checking...$(RESET)"
	mypy $(SRC)/ --ignore-missing-imports

qa: lint typecheck ## Run all quality checks (lint + typecheck)

# ─── Testing ──────────────────────────────────────────────────────────────────
test: ## Run all tests with coverage
	@echo "$(BOLD)Running all tests...$(RESET)"
	pytest $(TESTS)/ -v --cov=$(SRC) --cov-report=term-missing --cov-report=html

test-unit: ## Run unit tests only (fast)
	pytest $(TESTS)/unit/ -v -m "not slow"

test-integration: ## Run integration tests
	pytest $(TESTS)/integration/ -v

test-regression: ## Run regression tests
	pytest $(TESTS)/regression/ -v

test-perf: ## Run performance benchmarks
	pytest $(TESTS)/performance/ -v --benchmark-autosave

test-dataset: ## Run dataset validation tests (requires data)
	pytest $(TESTS)/dataset/ -v -m "dataset"

# ─── CropXAI Commands ─────────────────────────────────────────────────────────
download: ## Download the IBM-NASA dataset
	@echo "$(BOLD)Downloading IBM-NASA dataset...$(RESET)"
	cropxai download

validate: ## Validate the dataset integrity
	@echo "$(BOLD)Validating dataset...$(RESET)"
	cropxai validate

preprocess: ## Run the preprocessing pipeline
	@echo "$(BOLD)Running preprocessing pipeline...$(RESET)"
	cropxai preprocess

split: ## Generate train/val/test splits
	@echo "$(BOLD)Generating dataset splits...$(RESET)"
	cropxai split

explore: ## Launch the interactive dataset dashboard
	@echo "$(BOLD)Launching dataset explorer...$(RESET)"
	cropxai explore

report: ## Generate the dataset summary report
	@echo "$(BOLD)Generating dataset report...$(RESET)"
	cropxai report

visualize: ## Generate visualization figures
	@echo "$(BOLD)Generating visualizations...$(RESET)"
	cropxai visualize

# ─── Hydra ────────────────────────────────────────────────────────────────────
cfg: ## Show the composed Hydra configuration
	$(PYTHON) main.py --cfg job --resolve

# ─── Documentation ────────────────────────────────────────────────────────────
docs: ## Build documentation (placeholder for sphinx/mkdocs)
	@echo "$(BOLD)$(YELLOW)Documentation build not yet configured. See docs/ directory.$(RESET)"

# ─── Maintenance ──────────────────────────────────────────────────────────────
clean: ## Remove build artifacts, caches, and outputs
	@echo "$(BOLD)Cleaning project...$(RESET)"
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ coverage.xml .coverage
	@echo "$(BOLD)$(GREEN)✓ Clean complete.$(RESET)"

freeze: ## Export frozen requirements
	$(UV) pip freeze > requirements_frozen.txt
	@echo "$(BOLD)$(GREEN)✓ Frozen requirements written to requirements_frozen.txt$(RESET)"

env-info: ## Display environment information
	$(PYTHON) -c "from src.utils.environment.capture import capture_environment; import json; print(json.dumps(capture_environment(), indent=2))"
