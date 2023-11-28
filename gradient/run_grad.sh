#!/bin/bash
#SBATCH --job-name=sd-demo-noise # Specify a name for your job
#SBATCH --output=output-demo-noise.log       # Specify the output log file
#SBATCH --error=error-demo-noise.log         # Specify the error log file
# Specify the partition (queue) you want to use
#SBATCH --nodes=1                 # Number of nodes to request
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
# Number of tasks (CPUs) to request
#SBATCH --cpus-per-task=1         # Number of CPU cores per task
#SBATCH --qos=default             # Quality of service
#SBATCH --gres=gpu:rtxa4000:1     # Number of GPUs to request and specify the GPU type
#SBATCH --mem=32G                 # Memory per node

# Load any required modules or activate your base environment here if necessary
# Example: module load anaconda/3.8.3

# datasets=("tglcourse/lsun_church_train")
# datasets=("laion/dalle-3-dataset" "tglcourse/lsun_church_train")
# datasets=("fake_768" "real_768")
# datasets=("tglcourse/lsun_church_train" "laion/dalle-3-dataset")
datasets=("real_768" "fake_768")
dataset_types=("real" "generated")
#dataset_types=("generated" "real")
model_config_name_or_path="stabilityai/stable-diffusion-2-base"
resolution=768
time_steps=(5 10 20 25)
# time_steps=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950)
dft=0


cd /fs/nexus-scratch/olkowski/gen-image-detector/gradient || exit

# iterate over datasets with index
for i in "${!datasets[@]}"; do
  dataset="${datasets[$i]}"
  dataset_type="${dataset_types[$i]}"
  python3 detect_local.py --dataset_name "$dataset" --dataset_type "$dataset_type" --num-training_samples 100 \
  --enable_xformers_memory_efficient_attention \
  --model_config_name_or_path $model_config_name_or_path\
  --resolution $resolution \
  --output_dir "${model_config_name_or_path}-${resolution}-${dataset}-${dataset_type}-fft-${dft}" \
  --train_batch_size 1 \
  --num_epochs 1 \
  --gradient_accumulation_steps=1 \
  --mixed_precision=no \
  --logger="wandb" \
  --ddim_num_inference_steps 1000 \
  --error_type "noise_scale" \
  --timesteps "${time_steps[@]}"
done

