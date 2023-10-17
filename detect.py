import argparse
import logging
import math
import os

import requests
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader

from inverse_ddim import InvertibleStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from torchvision import transforms
from diffusers.utils import is_tensorboard_available, is_wandb_available


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the real Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )

    parser.add_argument(
        "--generated_dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the generated Dataset (from the HuggingFace hub) to train on (could be your own,"
            "possibly private, dataset). It can also be a path pointing to a local copy of a dataset in your filesystem"
            "or to a folder containing files that HF Datasets can understand."
        ),
    )

    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )

    parser.add_argument(
        "--num-training_samples",
        type=int,
        default=1000,
        help="The number of training samples to use from the dataset.",
    )

    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDIM configuration.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--generated_train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the generated training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ddim-model-64",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--save_images_epochs", type=int, default=10, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=False,
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument("--ddim_num_steps", type=int, default=1000)
    parser.add_argument("--ddim_num_inference_steps", type=int, default=50)
    parser.add_argument("--ddim_beta_schedule", type=str, default="linear")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument(
        "--delta_t",
        type=int,
        default=10,
        help="the step for reconstruction prediction",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args


def main(args):
    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
        # login using WANDB_LOGIN env variable
        wandb.login(key=os.environ.get("WANDB_LOGIN"))
        wandb.init(project="stablediffusion-detection", config=args)
    else:
        raise ValueError(f"Unknown logger: {args.logger}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    pipe = InvertibleStableDiffusionPipeline.from_pretrained(args.model_config_name_or_path)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_config_name_or_path, subfolder='scheduler')
    pipe.scheduler = scheduler
    pipe = pipe.to("cuda")

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")

    if args.generated_dataset_name is not None:
        generated_dataset = load_dataset(
            args.generated_dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    else:
        generated_dataset = load_dataset(path=args.generated_train_data_dir,
                                         cache_dir=args.cache_dir, split="train")
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    dataset = dataset.select(range(args.num_training_samples))
    generated_dataset = generated_dataset.select(range(args.num_training_samples))

    # Preprocessing the datasets and DataLoaders creation.
    transformations = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        # if "image" not in examples:
        #     images = []
        #     for image in examples["URL"]:
        #         try:
        #             im = transformations(Image.open(requests.get(image, stream=True).raw).convert("RGB"))
        #             images.append(im)
        #         except:
        #             logging.info(f"Image {image} could snot be downloaded.")
        #             continue
        #
        # else:
        images = [transformations(image.convert("RGB")) for image in examples["image"]]
        # if len of images is less than len of examples, clone the last image to fill the gap
        # if len(images) < len(examples):
        #     images += [images[-1]] * (len(examples) - len(images))
        # images = torch.stack(images)
        return {"input": images}

    logging.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(transform_images)
    generated_dataset.set_transform(transform_images)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    generated_train_dataloader = torch.utils.data.DataLoader(
        generated_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(dataset)}")
    logging.info(f"  Num Epochs = {args.num_epochs}")
    logging.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logging.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logging.info(f"  Total optimization steps = {max_train_steps}")

    prompt = ""
    text_embeddings = pipe.get_text_embedding(prompt)

    real_errors = pipe.reconstruction_error(args, train_dataloader, text_embeddings)

    generated_errors = pipe.reconstruction_error(args, generated_train_dataloader, text_embeddings)

    # logging.info("real_errors: ", real_errors)
    real_errors = [[err] for err in real_errors]

    # logging.info("generated_errors: ", generated_errors)
    generated_errors = [[err] for err in generated_errors]

    # create a histogram of the errors in wandb
    if args.logger == "wandb":
        real_table = wandb.Table(data=real_errors, columns=["errors"])
        wandb.log({args.output_dir: wandb.plot.histogram(real_table, "errors",
                                                         title="Real")})
        generated_table = wandb.Table(data=generated_errors, columns=["errors"])
        wandb.log({args.output_dir: wandb.plot.histogram(generated_table, "errors",
                                                         title="Generated")})

    elif args.logger == "tensorboard":
        raise NotImplementedError("Tensorboard logging is not implemented yet.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
