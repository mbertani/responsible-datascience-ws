# Makefile
.PHONY: help
help:
	@echo "Commands:"
	@echo "venv               : create the virtual environments for this workshop."
	@echo "remove-venv        : remove the virtual environments for this workshop."
	@echo "jupyter            : Runs jupyter lab"
	@echo "clean              : cleans all unecessary files."

# Installation
.PHONY: venv
venv:
	conda update -n base -c defaults conda
	conda env update -f ml_workshop.yaml
	conda env update -f ml_automl.yaml
	conda env update -f ml_data_validation.yaml
	conda env update -f ml_eda.yaml

.PHONY: remove-venv
remove-venv:
	conda env remove -n ml_workshop
	conda env remove -n ml_automl
	conda env remove -n ml_data_validation
	conda env remove -n ml_eda

.PHONY: jupyter
jupyter:
	jupyter lab

# Cleaning
.PHONY: clean
clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf

