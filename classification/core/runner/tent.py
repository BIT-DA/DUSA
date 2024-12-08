from .ttarunner import BaseTTARunner, logits_entropy
from mmengine.registry import RUNNERS
import torch
from ..model.wrapped_models import WrappedModels
from torch.nn.modules.batchnorm import _BatchNorm, _NormBase
from torch.nn.modules import LayerNorm, GroupNorm
from torch.nn.modules.instancenorm import _InstanceNorm


@RUNNERS.register_module()
class TentCls(BaseTTARunner):
    model: WrappedModels

    def tta_one_batch(self, batch_data):
        self.model.eval()
        all_loss = dict()
        with self.optim_wrapper.optim_context(self.model):
            task_batch_data = self.model.task_model.data_preprocessor(batch_data)
            inputs, data_samples = task_batch_data["inputs"], task_batch_data["data_samples"]
            task_output = self.model.task_model(inputs, data_samples, mode="tensor")
            loss = torch.mean(logits_entropy(task_output, task_output))
        self.optim_wrapper.update_params(loss)
        self.set_cls_predictions(task_output, data_samples)
        all_loss["entropy"] = loss.item()
        return data_samples, all_loss

    def config_tta_model(self):
        # find norm layers: norm in name (layer norm in ViTs), instance of subclass of _NormBase (batch norm)
        self.model.requires_grad_(False)
        all_norm_layers = []
        for name, sub_module in self.model.named_modules():
            if "norm" in name.lower() or isinstance(sub_module, (_NormBase, _InstanceNorm, LayerNorm, GroupNorm)):
                all_norm_layers.append(name)

        for name in all_norm_layers:
            sub_module = self.model.get_submodule(name)
            # fine tune the affine parameters in norm layers
            sub_module.requires_grad_(True)
            # if the sub_module is BN, then only current statistic is used for normalization
            if isinstance(sub_module, _BatchNorm) \
                    and hasattr(sub_module, "track_running_stats") \
                    and hasattr(sub_module, "running_mean") \
                    and hasattr(sub_module, "running_var"):
                sub_module.track_running_stats = False
                sub_module.running_mean = None
                sub_module.running_var = None

