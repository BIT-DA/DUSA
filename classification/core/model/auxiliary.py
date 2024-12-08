import logging
import warnings

import torch
import torch.nn as nn
from einops import rearrange

from typing import List, Dict
import torch.nn.functional as F
from typing import List, Union, Sequence
from .device_model import WrappedDeviceModel


class Preprocessor(nn.Module):
    """
    This preprocessor is for Diffusion Models only
    """
    def __init__(self, input_size=256, map2negone=True, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.map2negone = map2negone
        self.bgr2rgb = kwargs.get("bgr2rgb", True)

    def forward(self, inputs: List[torch.Tensor]):
        bgr_to_rgb = self.bgr2rgb
        if bgr_to_rgb:
            inputs = self.to_rgb(inputs)
        else:
            warnings.warn("Please make sure the input is in RGB format!", UserWarning)
        inputs = [tmp.float() for tmp in inputs]
        try:
            new_inputs = torch.stack(inputs, dim=0)
            b, c, h, w = new_inputs.shape
            if h != self.input_size or w != self.input_size:
                new_inputs = F.interpolate(input=new_inputs, size=(self.input_size, self.input_size), mode="bilinear",
                                           align_corners=True)
        except:
            new_inputs = []
            for tmp in inputs:
                assert isinstance(tmp, torch.Tensor), "error input type"
                c, h, w = tmp.shape
                tmp = rearrange(tmp, "c h w -> () c h w")
                if h != self.input_size or w != self.input_size:
                    tmp = F.interpolate(input=tmp, size=(self.input_size, self.input_size), mode="bilinear",
                                        align_corners=True)
                new_inputs.append(tmp)
            new_inputs = torch.cat(new_inputs, dim=0)

        # new_inputs: [b, c, h, w], value: [0, 255]
        new_inputs = new_inputs / 255
        if self.map2negone:
            new_inputs = new_inputs * 2 - 1

        return new_inputs

    # adapted from mmpretrain.models.utils.data_preprocessor.ClsDataPreprocessor.forward
    @staticmethod
    def to_rgb(inputs: List[torch.Tensor]):
        if isinstance(inputs, torch.Tensor):
            # If `default_collate` is used as the collate_fn in the
            # dataloader.

            # ------ To RGB ------
            if inputs.size(1) == 3:
                inputs = inputs.flip(1)
            return inputs
        else:
            # If `pseudo_collate` is used as the collate_fn in the
            # dataloader.

            processed_inputs = []
            for input_ in inputs:
                # ------ To RGB ------
                if input_.size(0) == 3:
                    input_ = input_.flip(0)

                processed_inputs.append(input_)
            return processed_inputs


class BaseAuxiliary(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert torch.cuda.is_available(), "cuda is not available!"
        self.cfg = cfg

        self.device = cfg.get("device", None)
        if self.device is not None:
            # prevent unnecessary data transfer
            if self.device == 'cuda:0':
                self.device = None
            # str to torch.device
            if isinstance(self.device, str):
                self.device = torch.device(self.device)

        name2components: Dict = self.build_components(self.cfg)
        for key, value in name2components.items():
            setattr(self, key, value)

        if self.device is not None:
            # if the model is not the WrappedDeviceModel, it will move to self.device
            # else it will move to its # always not wrapped
            self.to(self.device)

    def config_train_grad(self):
        raise NotImplementedError

    @staticmethod
    def build_preprocessor(preprocessor_cfg: Dict):
        preprocess = Preprocessor(**preprocessor_cfg)
        preprocess_device = preprocessor_cfg.get("device", None)
        if preprocess_device is not None:
            preprocess = WrappedDeviceModel(device=preprocess_device, model=preprocess)
        return preprocess

    @classmethod
    def build_components(cls, component_cfg):
        raise NotImplementedError

    def get_conditions(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, inputs: List[torch.Tensor], probs: torch.Tensor):
        raise NotImplementedError

    def freeze_vae_text_encoder_tokenizer(self):
        raise NotImplementedError

    def cast_data(self, data: Union[Sequence[torch.Tensor], torch.Tensor]):
        if self.device is None:
            # do not set the device, cuda:0 is the default device
            if isinstance(data, Sequence):
                if all(d.device == torch.device("cuda:0") for d in data):
                    new_data = data
                else:
                    new_data = [d.cuda() for d in data]
            elif isinstance(data, torch.Tensor):
                if data.device != torch.device("cuda:0"):
                    new_data = data.cuda()
                else:
                    new_data = data
            else:
                raise RuntimeError
            return new_data

        # we may use another card for the diffusion model
        if isinstance(data, Sequence):
            assert isinstance(data[0], torch.Tensor), "error type of data"
            if all(d.device == self.device for d in data):
                new_data = data
            else:
                new_data = [d.to(self.device) for d in data]
        elif isinstance(data, torch.Tensor):
            if data.device != self.device:
                new_data = data.to(self.device)
            else:
                new_data = data
        else:
            raise RuntimeError

        return new_data

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if dtype is not None:
            if not (dtype.is_floating_point or dtype.is_complex):
                raise TypeError('nn.Module.to only accepts floating point or complex '
                                f'dtypes, but got desired dtype={dtype}')
            if dtype.is_complex:
                warnings.warn(
                    "Complex modules are a new feature under active development whose design may change, "
                    "and some modules might not work as expected when using complex tensors as parameters or buffers. "
                    "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
                    "if a complex module does not work as expected.")

        def convert(t):
            if convert_to_format is not None and t.dim() in (4, 5):
                return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None,
                            non_blocking, memory_format=convert_to_format)
            return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)

        for child in self.children():
            child.to(*args, **kwargs)

        # remove recurse=False for version problem
        # return self._apply(convert)
        return self._apply(convert, recurse=False)
