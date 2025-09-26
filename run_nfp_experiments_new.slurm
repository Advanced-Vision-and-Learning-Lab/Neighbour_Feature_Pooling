#!/bin/bash
#SBATCH --job-name=gap_only-resnet18-eurosat
#SBATCH --output=/scratch/user/fahimehorvatinia/Neighbour_Feature_Pooling/logs_table2_eurosat/gap_only-resnet18-eurosat/slurm_%j.out
#SBATCH --error=/scratch/user/fahimehorvatinia/Neighbour_Feature_Pooling/logs_table2_eurosat/gap_only-resnet18-eurosat/slurm_%j.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

EXPERIMENT_NAME="gap_only-resnet18-eurosat"
DATASET="EuroSAT"
DATA_DIR="data/EuroSAT"

module purge
module load GCC/13.3.0
module load Python/3.12.3
source /scratch/user/fahimehorvatinia/Neighbour_Feature_Pooling/Extra_files/myenv/bin/activate
cd /scratch/user/fahimehorvatinia/Neighbour_Feature_Pooling

mkdir -p logs_table2_eurosat/${EXPERIMENT_NAME}
python demo.py \
    --name ${EXPERIMENT_NAME} \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --model_type resnet18\
    --model_variant gap_only
