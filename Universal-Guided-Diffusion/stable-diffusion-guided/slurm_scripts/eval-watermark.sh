#!/bin/bash
#SBATCH --job-name=sd-watermark # Specify a name for your job
#SBATCH --output=output-watermark.log       # Specify the output log file
#SBATCH --error=error-watermark.log         # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00           # Maximum execution time (HH:MM:SS)
# Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --qos=default             # Quality of service
#SBATCH --gres=gpu:rtxa5000:1     # Number of GPUs to request and specify the GPU type
#SBATCH --mem=32G                 # Memory per node

# Load any required modules or activate your base environment here if necessary
# Example: module load anaconda/3.8.3


model_config_name_or_path="stabilityai/stable-diffusion-2-1"
resolution=512

cd /fs/nexus-scratch/rezashkv/research/projects/Universal-Guided-Diffusion/stable-diffusion-guided/ || exit

python3 eval_watermark.py --dataset_name "laion/dalle-3-dataset" --num_training_samples 10 \
--enable_xformers_memory_efficient_attention \
--train_data_files "data/train-00000-of-00041-88a791d8d175f015.parquet" \
--model_config_name_or_path $model_config_name_or_path \
--cache_dir "/fs/nexus-scratch/rezashkv/.cache/huggingface/" \
--output_dir "/fs/nexus-scratch/rezashkv/research/projects/Universal-Guided-Diffusion/stable-diffusion-guided/test_segmentation/outputs" \
--resolution $resolution \
--train_batch_size=10 \
--num_epochs=1 \
--gradient_accumulation_steps=1 \
--mixed_precision=no \
--logger="wandb" \
--ddim_num_inference_steps 100

