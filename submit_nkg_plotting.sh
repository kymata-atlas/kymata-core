#!/bin/bash

###
# To run get_feats on the queue at the CBU, run the following command in command line:
#   sbatch submit_nkg_plotting.sh
###


#SBATCH --job-name=nkg_plotting
#SBATCH --output=slurm_log_plot.txt
#SBATCH --error=slurm_log_plot.txt
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --array=1-1
#SBATCH --exclusive

export PATH="$HOME/.local/bin:$PATH"
source $(poetry env info --path)/bin/activate
xvfb-run -a poetry run python -m kymata.invokers.invoker_run_scatter_plotting_new