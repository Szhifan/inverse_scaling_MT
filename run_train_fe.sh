#!/bin/bash
# Author(s): Siqi Sun (siqi.sun@ed.ac.uk)
# How to use: 
# 1. cd the repo
# 2. $sbatch run_train_fe.sh config/fe/edi/fe_edi_r1_h384_lr5e-5.json /work/tc046/tc046/siqisun/exp/
#


# ====================
# Options for sbatch
# ====================

# The partition specifies the set of nodes you want to run on.
# See https://cirrus.readthedocs.io/en/main/user-guide/batch.html#specifying-resources-in-job-scripts
#SBATCH --partition=gpu

# The QoS specifies the limits to apply to your job.
# See https://cirrus.readthedocs.io/en/main/user-guide/batch.html#specifying-resources-in-job-scripts
#SBATCH --qos=gpu

# Generic resources to use - typically you'll want gpu:n to get n gpus
#SBATCH --gres=gpu:1

# Maximum time for the job to run, format: days-hours:minutes:seconds
#SBATCH --time=12:00:00

# Location for stdout log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --output=/work/tc046/tc046/%u/slurm_logs/slurm-%A_%a.out

# Location for stderr log - see https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --error=/work/tc046/tc046/%u/slurm_logs/slurm-%A_%a.out

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=tc046-pool3

# =====================
# Logging information
# =====================

# slurm info - more at https://slurm.schedmd.com/sbatch.html#lbAJ
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

# Load the required modules
module load pytorch/1.12.1-gpu
echo "modules are loaded"

echo "config_path: $1"
echo "output_path: $2"

COMMAND="python train_fe.py --config_path $1 --output_path $2"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"

# =========================
# Post experiment logging
# =========================
echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"