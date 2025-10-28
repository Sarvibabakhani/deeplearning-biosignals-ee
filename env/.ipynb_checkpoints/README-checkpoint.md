## Reproducing the environment

### Option A — Conda (portable)
mamba env create -f env/environment.yml -n myproject-env
mamba activate myproject-env

# (Optional) exact reproduction on Linux:
# mamba create -n myproject-env --file env/conda-linux-64.lock

# Register Jupyter kernel (optional)
python -m ipykernel install --user --name myproject-env --display-name "Python (myproject-env)"
