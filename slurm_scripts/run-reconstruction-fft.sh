#!/bin/bash
#SBATCH --job-name=sd-demo # Specify a name for your job
#SBATCH --output=output-demo.log       # Specify the output log file
#SBATCH --error=error-demo.log         # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00           # Maximum execution time (HH:MM:SS)
# Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --qos=default             # Quality of service
#SBATCH --gres=gpu:rtxa4000:1     # Number of GPUs to request and specify the GPU type
#SBATCH --mem=32G                 # Memory per node

# Load any required modules or activate your base environment here if necessary
# Example: module load anaconda/3.8.3

datasets=("tglcourse/lsun_church_train" "laion/dalle-3-dataset")
dataset_types=("real" "generated")
model_config_name_or_path="stabilityai/stable-diffusion-2-base"
resolution=512
dft=1

cd cd /path/to/project || exit

for i in "${!datasets[@]}"; do
  dataset="${datasets[$i]}"
  dataset_type="${dataset_types[$i]}"
  python3 demo.py --dataset_name "$dataset" --dataset_type "$dataset_type" --num-training_samples 500 \
  --enable_xformers_memory_efficient_attention \
  --model_config_name_or_path $model_config_name_or_path\
  --resolution $resolution \
  --output_dir "${model_config_name_or_path}-${resolution}-${dataset}-${dataset_type}-fft-${dft}" \
  --train_batch_size=1 \
  --num_epochs=1 \
  --gradient_accumulation_steps=1 \
  --mixed_precision=no \
  --logger="wandb" \
  --ddim_num_inference_steps 50 \
  --dft \
  --error_type "reconstruction"
done
