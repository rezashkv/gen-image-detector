from functools import partial
from typing import Callable, Optional

import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler

import pandas as pd
import numpy as np
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
        with torch.no_grad():
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
        with torch.no_grad():
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
    def reconstruct_image_step_t(self, image_latents, j, text_embeddings, noise):
        with torch.no_grad():
            recon_latents = self.scheduler.add_noise(image_latents, noise, timesteps=self.scheduler.timesteps[-j])

        for i, t in enumerate(self.scheduler.timesteps):
            if j < len(self.scheduler.timesteps) - i:
                continue

            with torch.no_grad():
                residual = self.unet(recon_latents, t, encoder_hidden_states=text_embeddings).sample
                recon_latents = self.scheduler.step(residual, t, recon_latents).prev_sample
                
        recon_image = self.decode_image(recon_latents)
        return recon_image

    @torch.inference_mode()
    def noise_scale_error_batches(self, args, dataloader, text_embeddings, dft=False):
        self.scheduler.set_timesteps(args.ddim_num_inference_steps)

        errors = {}
        for i in range(len(args.timesteps)):
            errors[args.timesteps[i]] = []

        for step, batch in enumerate(dataloader):
            print(step)
            imgs = batch["input"]
            imgs = imgs.to("cuda")

            image_latents = self.get_image_latents(imgs,
                                                   rng_generator=torch.Generator(device=self.device).manual_seed(
                                                       0))

            err = []
            for timestep in args.timesteps:
                # compute the noise scale
                noise = torch.randn(image_latents.shape).to(image_latents.device)

                reconstructed_imgs = self.reconstruct_image_step_t(image_latents, timestep, text_embeddings, noise)

                errors[timestep].append(torch.sum((imgs - reconstructed_imgs) ** 2, dim=(1, 2, 3)).item())

            
        return errors



    @torch.inference_mode()
    def noise_scale_error(self, args, dataloader, text_embeddings, dft=False):
        self.scheduler.set_timesteps(args.ddim_num_inference_steps)

        errors = {}
        for i in range(len(args.timesteps)):
            for j in range(i + 1, len(args.timesteps)):
                errors[(args.timesteps[i], args.timesteps[j])] = []
        for step, batch in enumerate(dataloader):
            print(step)
            img = batch["input"]
            img = img.to("cuda")

            image_latents = self.get_image_latents(img,
                                                   rng_generator=torch.Generator(device=self.device).manual_seed(
                                                       0))

            err = {}
            for timestep in args.timesteps:
                # compute the noise scale
                noise = torch.randn(image_latents.shape).to(image_latents.device)

                reconstructed_img = self.reconstruct_image_step_t(image_latents, timestep, text_embeddings, noise)

                # if dft compute the error in the frequency domain
                if dft:
                    img_dft = torch.fft.rfft(img)
                    reconstructed_img_dft = torch.fft.rfft(reconstructed_img)

                    # compute the error only for high frequencies
                    img_dft = img_dft[:, :, img_dft.shape[2] // 2:, :]
                    reconstructed_img_dft = reconstructed_img_dft[:, :, reconstructed_img_dft.shape[2] // 2:, :]

                    err[timestep] = torch.sum((img_dft - reconstructed_img_dft) ** 2, dim=(1, 2, 3)).item()
                else:
                    err[timestep] = torch.sum((img - reconstructed_img) ** 2, dim=(1, 2, 3)).item()

            for i in range(len(args.timesteps)):
                for j in range(i + 1, len(args.timesteps)):
                    errors[(args.timesteps[i], args.timesteps[j])].append(
                        err[args.timesteps[i]] / err[args.timesteps[j]])
        return errors
    
    def grad_to_dist(self, grad, bins):
     # takes a 2D image gradient (Tensor) and approximates it by a distribution over reals (Tensor)
        grad_flat = torch.flatten(grad)
        grad_flat = grad_flat.tolist()

        df = pd.DataFrame(grad_flat, columns=['original'])
        df['bin'] = pd.cut(df['original'], bins=bins)
        df = df.groupby('bin', observed=False).count()    
    
        #adding eps to avoid empty support
        eps = 0.0000000000001
        df['original'] = df['original'] + eps
        df['original'] = df['original'] / df['original'].sum()
        return torch.FloatTensor(df['original'])
    
    def gradient_kl_div(self, batch_im1, batch_im2):
    # takes im1, im2 gradients in format (3xHxW) and returns coordinate-wise average of KL divergence of the gradients
        total = 0
        for i in range(len(batch_im1)):
            # discretizing the distribution with 0.01 precision on the interval [-1, 1] 
            # bins = [float(i) / 100 for i in range(-100, 101)]
    
            P = batch_im1[i] #self.grad_to_dist(batch_im1[0][i].cpu(), bins)
            Q = batch_im2[i] #self.grad_to_dist(batch_im2[0][i].cpu(), bins)
            
            # calculating the similarity, smaller means Q is closer to P
            kl_metric = (P * (P / Q).log()).sum()
            total += kl_metric
    
        return total.item() / len(batch_im1)
    
    
    def _compute_image_gradients(self, img):
        # """Compute image gradients (dy/dx) for a given image."""
        batch_size, channels, height, width = img.shape

        dy = img[..., 1:, :] - img[..., :-1, :]
        dx = img[..., :, 1:] - img[..., :, :-1]

        shapey = [batch_size, channels, 1, width]
        dy = torch.cat([dy, torch.zeros(shapey, device=img.device, dtype=img.dtype)], dim=2)
        dy = dy.view(img.shape)

        shapex = [batch_size, channels, height, 1]
        dx = torch.cat([dx, torch.zeros(shapex, device=img.device, dtype=img.dtype)], dim=3)
        dx = dx.view(img.shape)

        return dy, dx

    @torch.inference_mode()
    def gradient_kl_error(self, args, dataloader, text_embeddings):
        errors = {}
        for i in range(len(args.timesteps)):
            for j in range(i + 1, len(args.timesteps)):
                errors[(args.timesteps[i], args.timesteps[j])] = []
        
        for step, batch in enumerate(dataloader):
            #if step > 10: 
            #    break
            img = batch["input"]
            img = img.to("cuda")

            image_latents = self.get_image_latents(img,
                                                   rng_generator=torch.Generator(device=self.device).manual_seed(
                                                       0))
            for k in range(1):
                gradients_x = {}
                gradients_y = {}
                errors_local = {}
                for i in range(len(args.timesteps)):
                    for j in range(i + 1, len(args.timesteps)):
                        errors_local[(args.timesteps[i], args.timesteps[j])] = []

                for timestep in args.timesteps:
                    # compute the noise scale
                    noise = torch.randn(image_latents.shape).to(image_latents.device)

                    reconstructed_img = self.reconstruct_image_step_t(image_latents, timestep, text_embeddings, noise)
                 
                    # dx, dy are 1x3xHxW tensors
                    bins = [float(i) / 100 for i in range(-100, 101)]
                    dx, dy = self._compute_image_gradients(reconstructed_img)
                    gradients_x[timestep] = [self.grad_to_dist(grad.cpu(), bins) for grad in dx[0]]
                    gradients_y[timestep] = [self.grad_to_dist(grad.cpu(), bins) for grad in dy[0]]

                for i in range(len(args.timesteps)):
                    for j in range(i + 1, len(args.timesteps)):
                        errors_local[(args.timesteps[i], args.timesteps[j])].append(
                            0.5 * (self.gradient_kl_div(gradients_x[args.timesteps[i]], gradients_x[args.timesteps[j]])
                            + self.gradient_kl_div(gradients_y[args.timesteps[i]], gradients_y[args.timesteps[j]]))
                    )


            for i in range(len(args.timesteps)):
                for j in range(i + 1, len(args.timesteps)):
                    errors[(args.timesteps[i], args.timesteps[j])].append(np.mean(errors_local[(args.timesteps[i], args.timesteps[j])]))
            
            print("Step: ", step)
        print(errors)
        return errors

    @torch.inference_mode()
    def gradient_l2_error(self, args, dataloader, text_embeddings):
        errors = {}
        for i in range(len(args.timesteps)):
            for j in range(i + 1, len(args.timesteps)):
                errors[(args.timesteps[i], args.timesteps[j])] = []

        for step, batch in enumerate(dataloader):
            #if step > 10: 
            #    break
            img = batch["input"]
            img = img.to("cuda")

            image_latents = self.get_image_latents(img,
                                                   rng_generator=torch.Generator(device=self.device).manual_seed(
                                                       0))
            
            errors_local = {}
            for i in range(len(args.timesteps)):
                for j in range(i + 1, len(args.timesteps)):
                    errors_local[(args.timesteps[i], args.timesteps[j])] = []

            for timestep in args.timesteps:
                # compute the noise scale
                noise = torch.randn(image_latents.shape).to(image_latents.device)

                reconstructed_img = self.reconstruct_image_step_t(image_latents, timestep, text_embeddings, noise)

                # dx, dy are 1x3xHxW tensors
                dx, dy = self._compute_image_gradients(reconstructed_img)
                grad_norm = 0.5 * (torch.norm(dx) + torch.norm(dy))
                errors_local[timestep] = grad_norm


            for i in range(len(args.timesteps)):
                for j in range(i + 1, len(args.timesteps)):
                    errors[(args.timesteps[i], args.timesteps[j])].append(errors_local[args.timesteps[i]].item() / errors_local[args.timesteps[j]].item())

            print("Step: ", step)
        print(errors)
        return errors



    @torch.inference_mode()
    def noise_scale_error_dft_binned(self, args, dataloader, text_embeddings, n_bins=12):
        errors = {}
        for i in range(len(args.timesteps)):
            for j in range(i + 1, len(args.timesteps)):
                errors[(args.timesteps[i], args.timesteps[j])] = []
        for step, batch in enumerate(dataloader):
            img = batch["input"]
            img = img.to("cuda")

            image_latents = self.get_image_latents(img,
                                                   rng_generator=torch.Generator(device=self.device).manual_seed(
                                                       0))

            err = {}
            for timestep in args.timesteps:
                # compute the noise scale
                noise = torch.randn(image_latents.shape).to(image_latents.device)

                reconstructed_img = self.reconstruct_image_step_t(image_latents, timestep, text_embeddings, noise)

                img_dft = torch.fft.rfft(img)
                reconstructed_img_dft = torch.fft.rfft(reconstructed_img)

                # divide the img and reconstructed_img to 12 bins
                img_dft = torch.chunk(img_dft, n_bins, dim=2)
                reconstructed_img_dft = torch.chunk(reconstructed_img_dft, n_bins, dim=2)

                err[timestep] = [torch.abs(img_dft[i] - reconstructed_img_dft[i]).mean(dim=(1, 2, 3)).item() for i in
                                 range(n_bins)]

            for i in range(len(args.timesteps)):
                for j in range(i + 1, len(args.timesteps)):
                    errors[(args.timesteps[i], args.timesteps[j])].append(
                        [err[args.timesteps[i]][k] / err[args.timesteps[j]][k] for k in range(n_bins)])

        return errors
