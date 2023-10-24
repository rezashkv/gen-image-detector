from functools import partial
from typing import Callable, Optional

import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler


# Source: https://github.com/cccntu/efficient-prompt-to-prompt

def backward_ddim(x_t, alpha_t, alpha_tm1, eps_xt):
    """ from noise to image"""
    return (
            alpha_tm1 ** 0.5
            * (
                    (alpha_t ** -0.5 - alpha_tm1 ** -0.5) * x_t
                    + ((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt
            )
            + x_t
    )


def forward_ddim(x_t, alpha_t, alpha_tp1, eps_xt):
    """ from image to noise, it's the same as backward_ddim"""
    return backward_ddim(x_t, alpha_t, alpha_tp1, eps_xt)


class InvertibleStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(self,
                 vae,
                 text_encoder,
                 tokenizer,
                 unet,
                 scheduler,
                 safety_checker,
                 feature_extractor,
                 requires_safety_checker: bool = True,
                 ):
        super(InvertibleStableDiffusionPipeline, self).__init__(vae,
                                                                text_encoder,
                                                                tokenizer,
                                                                unet,
                                                                scheduler,
                                                                safety_checker,
                                                                feature_extractor,
                                                                requires_safety_checker)

        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)

    def get_random_latents(self, latents=None, height=512, width=512, generator=None):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        batch_size = 1
        device = self._execution_device

        num_channels_latents = self.unet.in_channels

        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            self.text_encoder.dtype,
            device,
            generator,
            latents,
        )

        return latents

    @torch.inference_mode()
    def get_text_embedding(self, prompt):
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        return text_embeddings

    @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents

    @torch.inference_mode()
    def latents_to_images(self, latents):
        x = self.decode_image(latents)
        x = self.torch_to_numpy(x)
        x = self.numpy_to_pil(x)
        return x

    @torch.inference_mode()
    def backward_diffusion(
            self,
            use_old_emb_i=25,
            text_embeddings=None,
            old_text_embeddings=None,
            new_text_embeddings=None,
            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            reverse_process: True = False,
            **kwargs,
    ):
        """ Generate image from text prompt and latents
        """
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.

        do_classifier_free_guidance = guidance_scale > 1.0
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        # Some schedulers like PNDM have timesteps as arrays.
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        if old_text_embeddings is not None and new_text_embeddings is not None:
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False

        for i, t in enumerate(
                self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))):
            latents, _, _, _ = self.diffusion_step(i, t, use_old_emb_i, text_embeddings,
                                                   old_text_embeddings, new_text_embeddings, latents, guidance_scale,
                                                   callback, callback_steps, reverse_process=reverse_process,
                                                   do_classifier_free_guidance=do_classifier_free_guidance,
                                                   prompt_to_prompt=prompt_to_prompt)

        return latents

    @torch.inference_mode()
    def diffusion_step(self, i, t, use_old_emb_i, text_embeddings, old_text_embeddings,
                       new_text_embeddings, latents, guidance_scale, callback,
                       callback_steps, reverse_process=False, do_classifier_free_guidance=False,
                       prompt_to_prompt=False, noise_pred=None):
        if prompt_to_prompt:
            if i < use_old_emb_i:
                text_embeddings = old_text_embeddings
            else:
                text_embeddings = new_text_embeddings

        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        if noise_pred is None:
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )

        prev_timestep = (
                t
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps
        )
        # call the callback, if provided
        if callback is not None and i % callback_steps == 0:
            callback(i, t, latents)

        # ddim
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        if reverse_process:
            alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t

        latents = backward_ddim(
            x_t=latents,
            alpha_t=alpha_prod_t,
            alpha_tm1=alpha_prod_t_prev,
            eps_xt=noise_pred,
        )

        return latents, noise_pred, alpha_prod_t, alpha_prod_t_prev

    @torch.inference_mode()
    def decode_image(self, latents: torch.FloatTensor, **kwargs):
        scaled_latents = 1 / 0.18215 * latents
        image = [
            self.vae.decode(scaled_latents[i: i + 1]).sample for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    @torch.inference_mode()
    def torch_to_numpy(self, image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image

    @torch.inference_mode()
    def reconstruction_error(self, args, dataloader, text_embeddings, dft=False):
        errors = []
        for step, batch in enumerate(dataloader):
            img = batch["input"]
            img = img.to("cuda")

            image_latents = self.get_image_latents(img,
                                                   rng_generator=torch.Generator(device=self.device).manual_seed(0))

            reversed_latents = self.forward_diffusion(
                latents=image_latents,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=args.ddim_num_inference_steps,
            )

            reconstructed_latents = self.backward_diffusion(
                latents=reversed_latents,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=args.ddim_num_inference_steps
                ,
            )

            reconstructed_img = self.decode_image(reconstructed_latents)

            # if dft compute the error in the frequency domain
            if dft:
                img = torch.fft.fft(img)
                reconstructed_img = torch.fft.fft(reconstructed_img)

                # compute the error only for high frequencies
                img = img[:, :, img.shape[2] // 2:, :]
                reconstructed_img = reconstructed_img[:, :, reconstructed_img.shape[2] // 2:, :]

            error = torch.abs(img - reconstructed_img).mean(dim=(1, 2, 3))
            errors.append(error.item())
        return errors

    @torch.inference_mode()
    def stepwise_error(self,
                       use_old_emb_i=25,
                       text_embeddings=None,
                       old_text_embeddings=None,
                       new_text_embeddings=None,
                       latents: Optional[torch.FloatTensor] = None,
                       num_inference_steps: int = 50,
                       guidance_scale: float = 7.5,
                       callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                       callback_steps: Optional[int] = 1,
                       reverse_process: True = False,
                       **kwargs):

        self.scheduler.set_timesteps(num_inference_steps)

        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        latents_t = latents * self.scheduler.init_noise_sigma

        noise_pred_t = None
        err = torch.zeros(latents_t.shape[0], device=self.device)

        for i in range(0, len(timesteps_tensor)):
            t = self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))[i]

            latents_t_p_1, noise_pred_t, alpha_prod_t, alpha_prod_t_p_1 = self.diffusion_step(i, t, use_old_emb_i,
                                                                                              text_embeddings,
                                                                                              old_text_embeddings,
                                                                                              new_text_embeddings,
                                                                                              latents_t,
                                                                                              guidance_scale,
                                                                                              callback, callback_steps,
                                                                                              reverse_process=reverse_process,
                                                                                              noise_pred=noise_pred_t)

            t_p_1 = self.progress_bar(timesteps_tensor if not reverse_process else reversed(timesteps_tensor))[i + 1]
            _, noise_pred_t_p_1, _, _ = self.diffusion_step(i + 1, t_p_1, use_old_emb_i,
                                                            text_embeddings,
                                                            old_text_embeddings,
                                                            new_text_embeddings,
                                                            latents_t_p_1,
                                                            guidance_scale,
                                                            callback,
                                                            callback_steps,
                                                            reverse_process=reverse_process)
            reconstructed_latents = backward_ddim(latents_t_p_1, alpha_prod_t_p_1, alpha_prod_t, noise_pred_t_p_1)

            err += torch.norm(latents_t - reconstructed_latents, dim=(1, 2, 3))

            noise_pred_t = noise_pred_t_p_1
            latents_t = latents_t_p_1

        return err

    @torch.inference_mode()
    def reconstruct_image_step_t(self, image_latents, timestep, text_embeddings, noise):
        timestep_t = torch.LongTensor([timestep]).to(image_latents.device)
        noisy_latents = self.scheduler.add_noise(image_latents, noise, timesteps=timestep_t)
        noise_pred = self.unet(noisy_latents, timestep_t, encoder_hidden_states=text_embeddings).sample
        alpha_t = self.scheduler.alphas_cumprod[timestep]
        reconstructed_latents = (noisy_latents - (1 - alpha_t) ** 0.5 * noise_pred) / (alpha_t ** 0.5)
        reconstructed_image = self.decode_image(reconstructed_latents)
        return reconstructed_image

    @torch.inference_mode()
    def noise_scale_error(self, args, dataloader, text_embeddings, dft=False):
        errors = []
        for step, batch in enumerate(dataloader):
            img = batch["input"]
            img = img.to("cuda")

            image_latents = self.get_image_latents(img,
                                                   rng_generator=torch.Generator(device=self.device).manual_seed(0))

            noise = torch.randn(image_latents.shape).to(img.device)

            reconstructed_image1 = self.reconstruct_image_step_t(image_latents, args.timestep1, text_embeddings, noise)
            reconstructed_image2 = self.reconstruct_image_step_t(image_latents, args.timestep2, text_embeddings, noise)

            # if dft compute the error in the frequency domain
            if dft:
                img = torch.fft.fft(img)
                reconstructed_image1 = torch.fft.fft(reconstructed_image1)
                reconstructed_image2 = torch.fft.fft(reconstructed_image2)

                # compute the error only for high frequencies
                img = img[:, :, img.shape[2] // 2:, :]
                reconstructed_image1 = reconstructed_image1[:, :, reconstructed_image1.shape[2] // 2:, :]
                reconstructed_image2 = reconstructed_image2[:, :, reconstructed_image2.shape[2] // 2:, :]

            # Technically error could be computed as the norm of the difference between the predicted noise
            # and the actual noise, but for code readability we compute the error as the norm of the difference between
            # the reconstructed images
            error1 = torch.abs(img - reconstructed_image1).mean(dim=(1, 2, 3))
            error2 = torch.abs(img - reconstructed_image2).mean(dim=(1, 2, 3))
            error = error1 / error2
            errors.append(error.item())

        return errors
