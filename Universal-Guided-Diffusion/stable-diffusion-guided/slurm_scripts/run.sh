#!/bin/bash
#SBATCH --job-name=ugd-watermark # Specify a name for your job
#SBATCH --output=logs/ugd-watermark.log       # Specify the output log file
#SBATCH --error=logs/ugd-watermark.log         # Specify the error log file  -=

# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00           # Maximum execution time (HH:MM:SS)
# Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa4000:1     # Number of GPUs to request and specify the GPU type
#SBATCH --mem=32G                 # Memory per node

# Load any required modules or activate your base environment here if necessary
# Example: module load anaconda/3.8.3

forward_guidance_wts=(5 10 20 30 40 50 100 200)

cd /fs/nexus-scratch/rezashkv/research/projects/Universal-Guided-Diffusion/stable-diffusion-guided || exit

for forward_guidance_wt in "${forward_guidance_wts[@]}"; do
    python watermark.py --indexes 1 --text "Walker hound, Walker foxhound on snow" --scale 1.5 --optim_forward_guidance \
    --optim_num_steps 1 --optim_forward_guidance_wt "$forward_guidance_wt" --optim_original_conditioning --ddim_steps 500 \
    --optim_folder ./test_segmentation/text_type_4/ --ckpt "./ckpts/sd-v1-4.ckpt" --trials 5
done