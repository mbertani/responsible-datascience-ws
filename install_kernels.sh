conda init

for env_name in ml_workshop ml_data_validation ml_eda ml_automl; do
    echo "Installing kernel for environment: $env_name"
    conda activate $env_name
    python -m ipykernel install --user --name $env_name
    conda deactivate
done
