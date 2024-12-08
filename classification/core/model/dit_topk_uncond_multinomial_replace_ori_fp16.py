from typing import List, Union

from diffusers.models import AutoencoderKL

from .DiT import models
from .DiT.download import find_model
import torch
from .auxiliary import BaseAuxiliary, Preprocessor
from .DiT.diffusion import create_diffusion
import torch.nn.functional as F
from mmengine.registry import MODELS
from .device_model import WrappedDeviceModel
from ..utils.super_indexing import index_select_plus
from einops import rearrange


@MODELS.register_module()
class DiTTopKUncondMultinomialReplaceOriFP16(BaseAuxiliary):
    diffusion_model: Union[models.DiT, WrappedDeviceModel]
    vae_model: Union[AutoencoderKL, WrappedDeviceModel]
    vae_scalar: float
    preprocessor: Union[Preprocessor, WrappedDeviceModel]
    class_embeddings: Union[torch.Tensor, None]

    def __init__(self, cfg):
        super().__init__(cfg)
        self.scheduler = create_diffusion(timestep_respacing="")
        # self.class_embeddings = None
        self.register_buffer("class_embeddings", self.prepare_class_embeddings())  # the same device with text_encoder
        self.register_buffer("uncond_class_embedding", self.prepare_uncond_class_embedding())  # the same device with text_encoder
        # get training timestep range, in [left, right)
        self.timestep_range = cfg.get('timestep_range', (0, self.scheduler.original_num_steps))
        self.topk = cfg.get("topk", 1)
        self.rand_budget = cfg.get("rand_budget", 0)
        self.temperature = cfg.get("temperature", 1.0)
        assert self.topk + self.rand_budget <= self.class_embeddings.shape[0], "topk + rand_budget should be less than the number of classes"

    def prepare_class_embeddings(self):
        with torch.no_grad():
            if isinstance(self.diffusion_model, models.DiT):
                all_class_idx = torch.arange(self.diffusion_model.y_embedder.num_classes, device=self.device)
                class_embeddings = self.diffusion_model.y_embedder(all_class_idx, False)
            elif isinstance(self.diffusion_model, WrappedDeviceModel):
                all_class_idx = torch.arange(self.diffusion_model.model.y_embedder.num_classes,
                                             device=self.diffusion_model.device)
                class_embeddings = self.diffusion_model.model.y_embedder(all_class_idx, False)
        return self.cast_data(class_embeddings)

    def prepare_uncond_class_embedding(self):
        with torch.no_grad():
            if isinstance(self.diffusion_model, models.DiT):
                uncond_class_idx = torch.tensor([self.diffusion_model.y_embedder.num_classes], device=self.device).long()
                uncond_class_embeddings = self.diffusion_model.y_embedder(uncond_class_idx, False)
            elif isinstance(self.diffusion_model, WrappedDeviceModel):
                uncond_class_idx = torch.tensor([self.diffusion_model.model.y_embedder.num_classes],
                                                device=self.diffusion_model.device).long()
                uncond_class_embeddings = self.diffusion_model.model.y_embedder(uncond_class_idx, False)
        return self.cast_data(uncond_class_embeddings)

    @staticmethod
    def build_diffusion(diffusion_cfg):
        dit_type: str = diffusion_cfg.get("type", "DiT_XL_2")
        image_size = diffusion_cfg.get("image_size", 256)

        assert image_size in [256, 512], "DiT only supports the 256x256 and 512x512 resolutions."
        latent_size = image_size // 8
        dit_model: Union[models.DiT, WrappedDeviceModel] = models.__dict__[dit_type](input_size=latent_size)
        path = dit_type.replace("_", "-")
        state_dict = find_model(f"{path}-{image_size}x{image_size}.pt")
        dit_model.load_state_dict(state_dict)

        dit_device = diffusion_cfg.get("device", None)
        if dit_device is not None:
            dit_model = WrappedDeviceModel(device=dit_device, model=dit_model)

        return dit_model

    @staticmethod
    def build_vae(vae_cfg):
        # load vae model
        vae_model = vae_cfg.get("pretrain")
        vae = AutoencoderKL.from_pretrained(vae_model)
        vae_device = vae_cfg.get("device", None)
        if vae_device is not None:
            vae = WrappedDeviceModel(device=vae_device, model=vae)
        vae_scalar = 0.18215

        return vae, vae_scalar

    @classmethod
    def build_components(cls, component_cfg):
        name2components = dict()

        diffusion_model = cls.build_diffusion(component_cfg.get("diffusion"))
        vae_model, vae_scalar = cls.build_vae(component_cfg.get("vae"))
        preprocessor = cls.build_preprocessor(component_cfg.get("preprocessor"))

        name2components.setdefault("diffusion_model", diffusion_model)
        name2components.setdefault("vae_model", vae_model)
        name2components.setdefault("vae_scalar", vae_scalar)
        name2components.setdefault("preprocessor", preprocessor)

        return name2components

    def config_train_grad(self):
        self.diffusion_model.requires_grad_(True)

        self.vae_model.requires_grad_(False)
        if isinstance(self.diffusion_model, models.DiT):
            self.diffusion_model.y_embedder.requires_grad_(False)
        elif isinstance(self.diffusion_model, WrappedDeviceModel):
            self.diffusion_model.model.y_embedder.requires_grad_(False)

        self.print_trainable_parameters(self.diffusion_model, 'diffusion_model')
        self.print_trainable_parameters(self.vae_model, 'vae_model')

    def get_conditions(self, probs: torch.Tensor):
        # probs: (bs, c)
        bs, num_c = probs.shape
        if self.class_embeddings.device != probs.device:
            self.class_embeddings = self.class_embeddings.to(probs.device)
        return probs @ self.class_embeddings

    def sample_time_step(self, size):
        left, right = self.timestep_range
        return torch.randint(left, right, (size,)).long()

    def forward(self, inputs: List[torch.Tensor], logits: torch.Tensor, ori_logits: torch.Tensor = None):
        assert ori_logits is not None, "ori_logits should not be None"
        # prepare data
        inputs = self.cast_data(inputs)
        logits = self.cast_data(logits)  # (bs, N)
        ori_logits = self.cast_data(ori_logits)  # (bs, N)

        # do color channel convert, resize, and value range scaling for diffusion inputs
        inputs = self.preprocessor(inputs)  # (bs, 3, 256, 256)
        bsz = inputs.shape[0]

        # get topk of logits
        topk_logits, topk_idx = torch.topk(logits, self.topk, dim=-1)  # (bs, topk)
        if self.rand_budget > 0:
            # choose random budget number of index, but exclude those in topk_idx
            non_topk_logits, non_topk_idx = torch.topk(logits, logits.shape[1] - self.topk, dim=-1, largest=False)  # (bs, n-topk)
            # NOTE that this impl makes it different from paper but equivalent, we want to reduce numerical instability
            rand_idx = torch.multinomial(torch.div(torch.gather(ori_logits, 1, non_topk_idx), self.temperature).softmax(1), self.rand_budget, replacement=False).cuda()  # (bs, rand_budget)
            # remap sampled index to class index
            rand_idx = torch.gather(non_topk_idx, 1, rand_idx)  # (bs, rand_budget)
            # combine topk_idx and rand_idx
            topk_idx = torch.cat([topk_idx, rand_idx], dim=-1)  # (bs, topk + rand_budget)
            topk_logits = torch.cat([topk_logits, torch.gather(logits, 1, rand_idx)], dim=-1)  # (bs, topk + rand_budget)
        # topk probs get a sum of 1
        topk_probs = F.softmax(topk_logits, dim=-1)  # (bs, topk)
        # number of selected classes to adapt
        topk = self.topk + self.rand_budget

        # get the latent variable
        latent = self.vae_model.encode(inputs).latent_dist.mean * self.vae_scalar  # (bs, 4, 32, 32)

        # sample time steps
        time_steps = self.sample_time_step(len(latent))  # (bs,)
        time_steps = self.cast_data(time_steps)

        # sample the noise
        noise = torch.randn_like(latent, device=latent.device)  # (bs, 4, 32, 32)

        # get the noised latent
        noised_latent = self.scheduler.q_sample(latent, time_steps, noise)  # (bs, 4, 32, 32)

        # uncond noise requires grad
        uncond = self.uncond_class_embedding  # (1, embedding_dim)
        uncond = uncond.repeat(bsz, 1)  # (bs, embedding_dim)
        uncond_model_output = self.diffusion_model(noised_latent, time_steps, None, uncond)  # (bs, 8, 32, 32)

        bs, c, h, w = noised_latent.shape
        assert uncond_model_output.shape == (bs, c * 2, h, w), "error output of the model"
        pred_noise, _ = torch.split(uncond_model_output, c, dim=1)  # (bs, 4, 32, 32)
        pred_noise = self.cast_data(pred_noise).to(dtype=torch.float32)
        loss_aux = F.mse_loss(pred_noise, noise)

        # conditional noise requires no grad
        with torch.no_grad():
            cond = self.class_embeddings  # (N, embedding_dim)
            # filter cond with topk_idx
            cond = cond.repeat(bsz, 1, 1)  # (bs, N, embedding_dim)
            cond = index_select_plus(cond, topk_idx)  # (bs, topk, embedding_dim)
            cond = rearrange(cond, "b k d -> (b k) d")  # (topk * bs, embedding_dim)

            # repeat for pairing rep and cond
            time_steps = time_steps.repeat_interleave(topk, dim=0)  # (bs * topk,)
            noised_latent = noised_latent.repeat_interleave(topk, dim=0)  # (bs * topk, 4, 32, 32)

            bs, c, h, w = noised_latent.shape
            model_output = self.diffusion_model(noised_latent, time_steps, None, cond)  # (bs * topk, 8, 32, 32)
            model_output = model_output.to(dtype=torch.float32)
            assert model_output.shape == (bs, c * 2, h, w), "error output of the model"
            pred_noise, _ = torch.split(model_output, c, dim=1)  # (bs * topk, 4, 32, 32)
            pred_noise = self.cast_data(pred_noise)

        weighted_noise = self.get_prob_score(topk_probs, pred_noise)  # (bs, 4, 32, 32)
        loss_task = F.mse_loss(weighted_noise, noise)
        loss = loss_aux + loss_task
        return loss

    @staticmethod
    def get_prob_score(probs, model_output):
        # probs: (bs, topk)
        # model_output: (bs * topk, 4, 32, 32)
        model_output = rearrange(model_output, "(b k) c h w -> b k c h w", b=probs.shape[0])
        # weighted_output: (bs, topk) * (bs, topk, 4, 32, 32) -> (bs, 4, 32, 32)
        weighted_output = torch.einsum("bk,bkchw->bchw", probs, model_output)
        return weighted_output

    # borrowed from HF
    @staticmethod
    def get_nb_trainable_parameters(model) -> tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_bytes = param.quant_storage.itemsize if hasattr(param, "quant_storage") else 1
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    # borrowed from HF
    def print_trainable_parameters(self, model, name='default') -> None:
        """
        Prints the number of trainable parameters in the model.

        Note: print_trainable_parameters() uses get_nb_trainable_parameters() which is different from
        num_parameters(only_trainable=True) from huggingface/transformers. get_nb_trainable_parameters() returns
        (trainable parameters, all parameters) of the Peft Model which includes modified backbone transformer model.
        For techniques like LoRA, the backbone transformer model is modified in place with LoRA modules. However, for
        prompt tuning, the backbone transformer model is unmodified. num_parameters(only_trainable=True) returns number
        of trainable parameters of the backbone transformer model which can be different.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters(model)

        print(
            f"{name} trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )