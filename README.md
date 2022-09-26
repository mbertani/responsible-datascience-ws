# responsible-datascience-ws
A workshop to learn how to create reproducible, clean and testable pipelines.

## Steps to run for the Setup

Preferably, you have run these steps before our workshop. This will save us some time.

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) in your operative system. Of course, if you have conda from before, that is fine. Miniconda is a lighter distribution with less baggage. Make sure you initialize conda for the shell you typically use by running `conda init` in it. 
2. To create the environment run: `make venv`. Beware: this wil also run `conda update -n base -c defaults conda` to update your environment to the latest version (currently `4.14.0`)
3. Then activate the environment `conda activate ml_workshop`
4. Run `make jupyter` to start jupyter lab and open the [ML_intro.ipynb]() notebook. If you came this far, your environment is ready.


## Environments to use during this workshop

We will use several conda environments for different notebooks:

- [ml_workshop](ml_wokshop.yaml): This is the main environmnent for the workshop. Used to run notebooks within [reproducible-ml](), [misc]().
- [ml_eda](ml_eda.yaml): Used to run notebooks under [eda]().
- [ml_data_validation](ml_data_validation.yaml): Used to run notebooks under [data_validation]().
- [ml_automl](ml_automl.yaml): Used to run notebooks under [auto-ml]().

Please make sure you activate the correct environment before runing the notebooks. This is done from the command line by runing `conda activate <environment>`.