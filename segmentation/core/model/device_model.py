import warnings

from mmengine.registry import MODELS
import torch
import torch.nn as nn
from typing import Union, Dict, Sequence, Mapping


class WrappedDeviceModel(nn.Module):
    def __init__(self, device: Union[str, torch.device], model: Union[Dict, nn.Module, None]):
        super().__init__()
        if isinstance(device, str):
            device = torch.device(device)

        assert isinstance(device, torch.device), "error type of device!"

        self.device = device

        if isinstance(model, dict):
            model = MODELS.build(cfg=model)

        if model is not None:
            assert isinstance(model, nn.Module), "error type of model!"

        self.model = model

    def cast_data(self, *args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, Sequence):
                new_arg, _ = self.cast_data(*arg)
            elif isinstance(arg, torch.Tensor):
                new_arg = self.cast_tensor(arg)
            elif isinstance(arg, Mapping):
                _, new_arg = self.cast_data(**arg)
            elif hasattr(arg, "to"):
                new_arg = arg.to(self.device)
            else:
                new_arg = arg
            new_args.append(new_arg)

        new_kwargs = dict()
        for key, arg in kwargs.items():
            if isinstance(arg, Sequence):
                new_arg, _ = self.cast_data(*arg)
            elif isinstance(arg, torch.Tensor):
                new_arg = self.cast_tensor(arg)
            elif isinstance(arg, Mapping):
                _, new_arg = self.cast_data(**arg)
            elif hasattr(arg, "to"):
                new_arg = arg.to(self.device)
            else:
                new_arg = arg
            new_kwargs.setdefault(key, new_arg)

        return new_args, new_kwargs

    def cast_tensor(self, tensor: torch.Tensor):
        if self.device != tensor.device:
            tensor = tensor.to(self.device)
        return tensor

    def forward(self, *args, **kwargs):
        args, kwargs = self.cast_data(*args, **kwargs)
        return self.model(*args, **kwargs)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        # if the target device is not the defined device, it will not change the device
        # to avoid changing device by their parents
        # only cpu or self.device is allowed
        if device is not None and device != self.device and device != torch.device("cpu"):
            warnings.warn("the target device is not the defined device or cpu, reset the device to defined device!")

        device = self.device

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

        return self._apply(convert)




