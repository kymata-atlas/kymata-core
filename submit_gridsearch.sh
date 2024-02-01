#!/bin/bash

###
# To run gridsearch on the queue at the CBU, run the following command in command line:
#   sbatch submit_gridsearch.sh
###


#SBATCH --job-name=gridsearch
#SBATCH --output=slurm_log5.txt
#SBATCH --error=slurm_log5.txt
# SBATCH --ntasks=1
#SBATCH --time=05:00:00
# SBATCH --mem=240G
#SBATCH --array=0-0
#SBATCH --exclusive

conda activate mne_venv

func_names=("d_IL2" "d_IL3")

part_names=("participant_01"\
            "participant_01b"\
            "participant_02"\
            "participant_03"\
            "participant_04"\
            "participant_05"\
            "participant_06"\
            "participant_08"\
            "participant_09"\
            "participant_10"\
            "participant_11"\
            "participant_12"\
            "participant_13"\
            "participant_14"\
            "participant_15"\
            "participant_16"\
            "participant_17"\
            "pilot_01"\
            "pilot_02"\
            )

rep_names=("-ave"
           "_rep0"
           "_rep1"
           "_rep2"
           "_rep3"
           "_rep4"
           "_rep5"
           "_rep6"
           "_rep7")

ARG="${part_names[$(($SLURM_ARRAY_TASK_ID % 19))]}${rep_names[$(($SLURM_ARRAY_TASK_ID / 19))]}"

# inv_op_name="participant_01_ico5-3L-loose02-cps-nodepth-inv.fif"
inv_op_name="participant_01_ico5-3L-loose02-cps-nodepth-fusion-inv.fif"
# inv_op_name="participant_01_ico5-3L-loose02-cps-nodepth-test.fif"
# inv_op_name="meg15_0051_ico-5-3L-loose02-diagnoise-nodepth-reg-inv-csd.fif"
inv_op_name="none"


python invokers/run_gridsearch.py \
    --downsample_rate 5 \
    --inverse_operator_name $inv_op_name

conda deactivate
