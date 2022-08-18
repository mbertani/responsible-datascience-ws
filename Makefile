# Makefile
.PHONY: help
help:
	@echo "Commands:"
	@echo "venv               : create a virtual environment."
	@echo "activate           : activates the environment."
	@echo "jupyter            : Runs jupyter lab"
	@echo "clean              : cleans all unecessary files."

# Installation
.PHONY: venv
venv:
	conda update -n base -c defaults conda
	conda env update -f ml_workshop.yaml

.PHONY: activate
activate:
	conda activate ml_workshop

.PHONY: jupyter
jupyter:
	jupyter lab

# Cleaning
.PHONY: clean
clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf

