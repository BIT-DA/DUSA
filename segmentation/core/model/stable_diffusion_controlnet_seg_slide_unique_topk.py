import random

import torchvision.transforms.v2
from math import ceil
from typing import Union, List

import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.schedulers import PNDMScheduler
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel

from .wrapped_models import WrappedEmbeddings
from ..utils.imagenet_templates import IMAGENET_TEMPLATES
from ..utils import categories
import torch.nn.functional as F
from ..utils.super_indexing import index_select_plus
from .auxiliary import BaseAuxiliary, Preprocessor
from mmengine.registry import MODELS
from einops import rearrange, repeat
from .device_model import WrappedDeviceModel
from tqdm import tqdm


@MODELS.register_module()
class StableDiffusionControlnetSegSlideUniqueTopK(BaseAuxiliary):
    vae_model: Union[AutoencoderKL, WrappedDeviceModel]
    text_encoder: Union[CLIPTextModel, WrappedDeviceModel]
    tokenizer: CLIPTokenizer
    unet: Union[UNet2DConditionModel, WrappedDeviceModel]
    scheduler: PNDMScheduler
    vae_scalar: float
    preprocessor: Union[Preprocessor, WrappedDeviceModel]
    class_embeddings: Union[None, torch.Tensor]

    def __init__(self, cfg):
        super(StableDiffusionControlnetSegSlideUniqueTopK, self).__init__(cfg)
        self.class_names = categories.__dict__[cfg.get("class_names")]

        self.prompt = self.cfg.get("prompt", None)  # please make sure: self.prompt is given like "xxx xxx {} xxx"
        # using --cfg-options model.auxiliary_model.prompt="a bad photo of {}." always get "abadphotoof{}."
        # please use  --cfg-options model.auxiliary_model.prompt="a_bad_photo_of_a_{}."
        if self.prompt is not None:
            self.prompt = self.prompt.replace("_", " ")
            print(f"using prompt: {self.prompt}")

        self.register_buffer("class_embeddings", self.prepare_class_embeddings())  # the same device with text_encoder
        self.timestep_range = cfg.get('timestep_range', (0, self.scheduler.num_train_timesteps))
        self.topk = cfg.get("topk", 1)
        self.temperature = cfg.get("temperature", 1.0)
        self.rand_budget = cfg.get("rand_budget", 0)
        self.classes_threshold = cfg.get("classes_threshold", 20)
    @torch.no_grad()
    def prepare_class_embeddings(self):
        fixed_prompt = "a photo of a {}" if self.prompt is None else self.prompt
        text_features = []
        for class_name in self.class_names:
            text_inputs = self.tokenizer(
                    fixed_prompt.format(class_name),
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
            text_input_ids = text_inputs.input_ids
            text_input_ids = self.cast_data(text_input_ids)
            prompt_embeds = self.text_encoder(text_input_ids)[0]  # [B, L, D], B = 1, L = 77, D = 1024
            text_features.append(prompt_embeds)
        class_embeddings = torch.cat(text_features, dim=0)  # [C, L, D]
        return class_embeddings

    @staticmethod
    def get_component_device(cfg):
        if cfg is not None:
            return cfg.get("device", None)
        else:
            return None

    @staticmethod
    def build_components(components_cfg):
        name2components = dict()

        controlnet = ControlNetModel.from_pretrained(components_cfg.get("controlnet_ckpt"), torch_dtype=torch.float32)

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            components_cfg.get("model_path"), controlnet=controlnet, torch_dtype=torch.float32
        )
        # pipe.enable_xformers_memory_efficient_attention()
        vae_model = pipe.vae
        vae_scalar = pipe.vae.config.scaling_factor
        text_encoder = pipe.text_encoder
        tokenizer = pipe.tokenizer
        unet = pipe.unet
        scheduler = pipe.scheduler
        controlnet = pipe.controlnet

        preprocessor = StableDiffusionControlnetSegSlideUniqueTopK.build_preprocessor(components_cfg.get("preprocessor"))

        vae_device = StableDiffusionControlnetSegSlideUniqueTopK.get_component_device(components_cfg.get("vae", None))
        if vae_device is not None:
            vae_model = WrappedDeviceModel(device=vae_device, model=vae_model)

        text_encoder_device = StableDiffusionControlnetSegSlideUniqueTopK.get_component_device(components_cfg.get("text_encoder", None))
        if text_encoder_device is not None:
            text_encoder = WrappedDeviceModel(device=text_encoder_device, model=text_encoder)

        unet_device = StableDiffusionControlnetSegSlideUniqueTopK.get_component_device(components_cfg.get("unet", None))

        unet_config = components_cfg.get("unet", None)
        if unet_config is not None:
            unet_checkpointing = unet_config.get("checkpointing", False)
            if unet_checkpointing:
                print("check pointing is using on unet")
                unet.enable_gradient_checkpointing()

        if unet_device is not None:
            unet = WrappedDeviceModel(device=unet_device, model=unet)

        controlnet_device = StableDiffusionControlnetSegSlideUniqueTopK.get_component_device(components_cfg.get("controlnet", None))

        if controlnet_device is not None:
            controlnet = WrappedDeviceModel(device=controlnet_device, model=controlnet)

        name2components.setdefault("vae_model", vae_model)
        name2components.setdefault("vae_scalar", vae_scalar)
        name2components.setdefault("text_encoder", text_encoder)
        name2components.setdefault("tokenizer", tokenizer)
        name2components.setdefault("unet", unet)
        name2components.setdefault("scheduler", scheduler)
        name2components.setdefault("preprocessor", preprocessor)
        name2components.setdefault("controlnet", controlnet)

        return name2components

    def config_train_grad(self):
        # default unet update
        if not self.cfg.get("unet") or self.cfg.unet.get("update", True):
            self.unet.requires_grad_(True)
        else:
            self.unet.requires_grad_(False)
        # default controlnet update
        if not self.cfg.get("controlnet") or self.cfg.controlnet.get("update", True):
            self.controlnet.requires_grad_(True)
        else:
            self.controlnet.requires_grad_(False)
        self.controlnet.requires_grad_(True)
        self.vae_model.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def sample_time_step(self, size):
        left, right = self.timestep_range
        return torch.randint(left, right, (size,)).long()

    def slide_image(self, inputs: Union[List[torch.Tensor], torch.Tensor], stride_size=(360, 640), crop_size=(360, 640)):
        if isinstance(inputs, List):
            inputs = torch.stack(inputs)
        h_stride, w_stride = stride_size
        h_crop, w_crop = crop_size
        batch_size, channels ,h_img, w_img = inputs.size()
        h_grids = 1 if h_stride==0 else max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = 1 if w_stride==0 else max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        slide_inputs = []

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_input = inputs[:, :, y1:y2, x1:x2]
                slide_inputs.append(crop_input)
        return slide_inputs

    def forward(self, inputs, logits):
        # operation on the self.device
        # prepare data
        inputs = self.cast_data(inputs)
        logits = self.cast_data(logits)  # (bs, n, h, w)
        aux_slide_cfg = self.cfg.get("auxiliary_slide")
        aux_stride_size = aux_slide_cfg.get("stride")
        aux_crop_size = aux_slide_cfg.get("crop_size")
        slide_inputs = self.slide_image(inputs, aux_stride_size, aux_crop_size)

        task_stride_size = (aux_stride_size[0] // 8, aux_stride_size[1] // 8)
        task_crop_size = (aux_crop_size[0] // 8, aux_crop_size[1] // 8)

        slide_logits = self.slide_image(logits, task_stride_size, task_crop_size)

        # idx = random.randint(0, len(slide_logits) - 1)
        #
        # slide_input, slide_logits = slide_inputs[idx], slide_logits[idx]
        slide_window_num = len(slide_inputs)
        for i, (slide_input, slide_logits) in enumerate(zip(slide_inputs, slide_logits)):
            torch.cuda.empty_cache()
            if i < slide_window_num - 1:
                loss_task_grad, loss_aux_grad = self.single_forward(slide_input, slide_logits, islast=False)
                loss_task_grad = loss_task_grad / slide_window_num
                loss_aux_grad = loss_aux_grad / slide_window_num
                loss_task_grad.backward(retain_graph=True)
                loss_aux_grad.backward()
            else:
                return self.single_forward(slide_input, slide_logits) / slide_window_num

    def single_forward(self, inputs: List[torch.Tensor], logits: torch.Tensor, islast=True):
        # random sample t, and noise for the input samples, and return the diffusion loss
        new_inputs = self.preprocessor(inputs)  # always not wrapped

        bsz = new_inputs.shape[0]

        zero_controlnet_cond = torch.zeros(bsz, 3, inputs.shape[-2], inputs.shape[-1], dtype=torch.float32)
        zero_controlnet_cond = self.cast_data(zero_controlnet_cond)

        # get the latent variable
        tmp = self.vae_model.encode(new_inputs).latent_dist.mean  # always not wrapped
        latent = tmp * self.vae_scalar

        # sample time steps
        time_steps = self.sample_time_step(len(latent))
        time_steps = self.cast_data(time_steps)

        # sample the noise
        noise = torch.randn_like(latent, device=latent.device)

        # get resized logits
        resized_logits = F.interpolate(logits, size=noise.shape[2:], mode='bilinear', align_corners=True)

        # get topk of logits
        topk_logits, topk_idx = torch.topk(resized_logits, self.topk, dim=1)  # (bs, topk, h, w)


        # Note that the implementation of multinomial selection here diverges from that in classification,
        # this is due to the use of old code, and as we decide to keep rand_budget=0, there should be no issue.
        # We reserve the code here just for better extensiveness
        if self.rand_budget > 0:
            # choose random budget number of index, but exclude those in topk_idx
            non_topk_logits, non_topk_idx = torch.topk(resized_logits, resized_logits.shape[1] - self.topk, dim=1,
                                                       largest=False)  # (bs, n-topk, h, w)
            b, k, h, w = non_topk_logits.shape
            tmp_twodim_logits = rearrange(torch.div(non_topk_logits, self.temperature).softmax(1), "b k h w -> (b h w) k")
            tmp_twodim_idx = torch.multinomial(tmp_twodim_logits,
                                         self.rand_budget, replacement=False).cuda()  # (bs*h*w, rand_budget)
            rand_idx = rearrange(tmp_twodim_idx, "(b h w) k -> b k h w", b=b, h=h, w=w)
            rand_idx = torch.gather(non_topk_idx, 1, rand_idx)  # (bs, rand_budget, h, w)
            # combine topk_idx and rand_idx
            topk_idx = torch.cat([topk_idx, rand_idx], dim=1)  # (bs, topk + rand_budget)


        b, k, h, w = resized_logits.shape
        mask_metric = torch.ones(bsz, 1, h, w)
        mask_metric = self.cast_data(mask_metric)
        topk_idx_unique = topk_idx.unique()
        if topk_idx_unique.shape[0] > self.classes_threshold:
            # TODO: make sure all top1 is selected and randomly select other class
            _, top1_idx = torch.topk(resized_logits, 1, dim=1) # (bs, 1, h, w)
            top1_idx_unique = top1_idx.unique()
            if top1_idx_unique.shape[0] > self.classes_threshold:
                topk_idx_unique = top1_idx_unique[torch.randperm(top1_idx_unique.shape[0])[:self.classes_threshold]]
                # set mask_metric to 0 in the position where top1_idx isn't in topk_idx_unique
                mask_metric = mask_metric * torch.where(torch.isin(top1_idx,topk_idx_unique), 1, 0)
            else:
                topk_idx_unique = topk_idx_unique[~torch.isin(topk_idx_unique, top1_idx_unique)]
                # randomly select other class index
                topk_idx_unique = topk_idx_unique[torch.randperm(topk_idx_unique.shape[0])[:self.classes_threshold - top1_idx_unique.shape[0]]]
                topk_idx_unique = torch.cat([top1_idx_unique, topk_idx_unique], dim=0)

        topk_idx = topk_idx_unique.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(bsz, 1, h, w)  # (bs, topk, h, w)
        topk_logits = torch.gather(resized_logits, 1, topk_idx)

        # topk probs get a sum of 1
        topk_probs = F.softmax(topk_logits, dim=1)  # (bs, topk, h, w)

        # get the noised latent
        noised_latent = self.scheduler.add_noise(latent, noise, time_steps)

        cond = self.class_embeddings  # (N, embedding_dim1, embedding_dim2)
        # select cond in topk_idx_unique
        cond = torch.index_select(cond, 0, topk_idx_unique)  # (N, embedding_dim1, embedding_dim2)

        classes = cond.shape[0]

        cond = cond.repeat(bsz, 1, 1, 1)  # (bs, N, embedding_dim1, embedding_dim2)

        cond = rearrange(cond, "b k h w -> (b k) h w")  # (bs * N, embedding_dim1, embedding_dim2)

        # repeat for pairing rep and cond
        time_steps = time_steps.repeat_interleave(classes, dim=0)  # (bs * N,)
        noised_latent = noised_latent.repeat_interleave(classes, dim=0)  # (bs * N, 4, 32, 32)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noised_latent,
            time_steps,
            encoder_hidden_states=cond,
            controlnet_cond=zero_controlnet_cond,
            conditioning_scale=1.0,
            guess_mode=False,
            return_dict=False,
        )

        down_block_res_samples = self.cast_data(down_block_res_samples)
        mid_block_res_sample = self.cast_data(mid_block_res_sample)
        pred_noise = self.unet(
            noised_latent,
            time_steps,
            encoder_hidden_states=cond,
            timestep_cond=None,
            cross_attention_kwargs=None,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]

        pred_noise = self.cast_data(pred_noise)

        if pred_noise.dtype != topk_probs.dtype:
            pred_noise = pred_noise.to(topk_probs.dtype)

        pred_noise = torch.masked_fill(pred_noise, mask_metric == 0, 0.0)
        noise = torch.masked_fill(noise, mask_metric == 0, 0.0)

        if islast:
            weighted_noise_all_grad = self.get_prob_score(topk_probs, pred_noise) # noqa
            loss_all_grad = F.mse_loss(weighted_noise_all_grad, noise)
            return loss_all_grad
        else:
            weighted_noise_task_grad = self.get_prob_score(topk_probs, pred_noise.clone().detach()) # noqa
            weighted_noise_aux_grad = self.get_prob_score(topk_probs.clone().detach(), pred_noise)

            loss_task_grad = F.mse_loss(weighted_noise_task_grad, noise.clone().detach())
            loss_aux_grad = F.mse_loss(weighted_noise_aux_grad, noise)
            return loss_task_grad, loss_aux_grad

    def get_prob_score(self, probs, model_output):
        # probs: (bs, n)
        # model_output: (bs * N, 4, 32, 32)
        model_output = rearrange(model_output, "(b k) c h w -> b k c h w", b=probs.shape[0])
        # weighted_output: (bs, N) * (bs, N, 4, 32, 32) -> (bs, 4, 32, 32)
        weighted_output = torch.einsum("b k h w, b k c h w -> b c h w", probs, model_output)
        return weighted_output

    def get_topK_prob_score(self, topk_probs, topk_idx, model_output):
        # topk_probs: (bs, topk, h, w)
        # topk_idx: (bs, topk, h, w)
        # model_output: (bs * N, 4, 32, 32)
        model_output = rearrange(model_output, "(b n) c h w -> b n c h w", b=topk_probs.shape[0])

        tmp_output = rearrange(model_output, "b n c h w -> b h w n c")

        tmp_idx = rearrange(topk_idx, "b k h w -> b h w k")

        topk_output = index_select_plus(tmp_output, tmp_idx)  # b h w k c

        weighted_output = torch.einsum("b k h w, b h w k c -> b c h w", topk_probs, topk_output)

        return weighted_output
