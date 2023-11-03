#!/bin/sh
#SBATCH --job-name=FaceDetection
#SBATCH --time=72:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/cap6411.student17/
#SBATCH --output=/home/cap6411.student17/%x-%j.out

module purge
module load anaconda
module load cuda

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export MASTER_PORT=12668
echo "MASTER_PORT=12668"

### init conda environment if needed
source activate FaceDetection

### the command to run
srun python main.py