import warnings

from mmengine.registry import MODELS
from mmengine.model import BaseModule
import torch
import torch.nn as nn
from copy import deepcopy

from torch import Tensor
from torchvision import transforms

from ..model.auxiliary import BaseAuxiliary
from typing import Union
from .wrapped_networks import WrappedEncoderDecoder
import torch.nn.functional as F


@MODELS.register_module()
class WrappedModels(BaseModule):
    def __init__(self, task_model=None, auxiliary_model=None, **kwargs):
        super().__init__()
        self.task_cfg: dict = deepcopy(task_model)
        self.auxiliary_cfg: dict = deepcopy(auxiliary_model)
        self.kwargs: dict = deepcopy(kwargs)

        assert torch.cuda.is_available(), "cuda is not available"
        # TODO: change here to adapt for DDP, not sure if this will damage model parallel
        self.default_device = torch.device("cuda")

        self.task_model: Union[WrappedEncoderDecoder, None] = MODELS.build(self.task_cfg)  # model from mmseg
        # move the task model to device task
        device_task = self.task_cfg.get("device", None)
        self.device_task = torch.device(device_task) if device_task is not None else self.default_device
        self.task_model.to(self.device_task)

        if self.auxiliary_cfg is not None:
            auxiliary_cls = MODELS.get(self.auxiliary_cfg.get("type"))
            self.auxiliary_cfg.setdefault("device", self.default_device)
            self.auxiliary_model: BaseAuxiliary = auxiliary_cls(self.auxiliary_cfg)
            # get the auxiliary model on device diff
        else:
            # For Source or other tta methods like tent, cotta
            self.auxiliary_model = None

    def forward(self, batch_data, mode='logits'):
        assert mode in ('logits', 'probs', 'normed_logits')
        origin_inputs = batch_data["inputs"]

        task_batch_data = batch_data

        # There are 'training' option in data_preprocessor for whether to enable training time augmentation.
        task_batch_data = self.task_model.data_preprocessor(task_batch_data)
        inputs, data_samples = task_batch_data["inputs"], task_batch_data["data_samples"]

        with torch.cuda.amp.autocast(enabled=False):
            task_output = self.task_model(inputs, data_samples, mode="tensor")

        aux_inputs = origin_inputs
        logits = task_output
        # softmax, normalize can autocast to float32
        if mode == 'logits':
            condition_loss = self.auxiliary_model(aux_inputs, logits=logits)
        elif mode == 'probs':
            probs = torch.softmax(logits, dim=1)
            condition_loss = self.auxiliary_model(aux_inputs, probs=probs)
        elif mode == 'normed_logits':
            normed_logits = F.normalize(logits, p=2, dim=1) #(1, n, h, w)
            condition_loss = self.auxiliary_model(aux_inputs, logits=normed_logits)
        else:
            raise NotImplementedError
        return data_samples, task_output, condition_loss

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
        return self._apply(convert, recurse=False)


class WrappedEmbeddings(nn.Module):
    def __init__(self, original_embedding: nn.Embedding, num_prompt, initial_ids=None):
        super(WrappedEmbeddings, self).__init__()
        self.original_embedding = original_embedding
        self.new_embedding = nn.Embedding(num_embeddings=num_prompt, embedding_dim=original_embedding.embedding_dim)
        self.new_embedding.to(self.original_embedding.weight.device)

        if initial_ids is not None:
            assert num_prompt == len(initial_ids), "error initial ids"
            with torch.no_grad():
                self.new_embedding.weight.data = self.original_embedding.weight.data[initial_ids]

    def forward(self, input: Tensor) -> Tensor:
        new_weight = torch.cat([self.original_embedding.weight, self.new_embedding.weight])
        return F.embedding(
            input, new_weight, self.original_embedding.padding_idx, self.original_embedding.max_norm,
            self.original_embedding.norm_type, self.original_embedding.scale_grad_by_freq, self.original_embedding.sparse)
