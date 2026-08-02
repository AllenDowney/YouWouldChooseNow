.PHONY: help env delete_env update_env rebuild_env clean lint format tests ipynb md

PROJECT_NAME = YouWouldChooseNow
CONDA = conda run --no-capture-output -n $(PROJECT_NAME)

help:
	@echo "Available targets:"
	@echo "  env           - Create conda environment from environment.yml"
	@echo "  delete_env    - Remove conda environment"
	@echo "  update_env    - Update environment (with --prune)"
	@echo "  rebuild_env   - Delete and recreate environment"
	@echo "  clean         - Remove Python caches"
	@echo "  lint          - flake8 + black --check on notebooks/*.py"
	@echo "  format        - black notebooks/*.py"
	@echo "  tests         - pytest --nbmake notebooks/*.ipynb"
	@echo "  ipynb         - Convert notebooks/*.md → .ipynb (jupytext)"
	@echo "  md            - Convert notebooks/*.ipynb → .md (jupytext)"

# Environment management (using mamba for speed)
env:
	mamba env create -f environment.yml -y
	@echo ""
	@echo ">>> Environment ready. Activate with:"
	@echo ">>> conda activate $(PROJECT_NAME)"

delete_env:
	mamba env remove --name $(PROJECT_NAME) -y

update_env:
	mamba env update -f environment.yml --prune -y

# Full rebuild (PyMC major upgrades: prefer this over update_env)
rebuild_env: delete_env env

# Code quality
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

lint:
	flake8 notebooks/*.py
	black --check notebooks/*.py

format:
	black notebooks/*.py

# Testing
tests:
	pytest --nbmake notebooks/*.ipynb

# Jupytext: Markdown is canonical; .ipynb is generated
ipynb:
	cd notebooks && jupytext --to ipynb *.md

md:
	cd notebooks && jupytext --to md *.ipynb
