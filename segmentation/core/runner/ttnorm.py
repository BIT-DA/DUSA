from mmengine.registry import RUNNERS
from torch.nn.modules.batchnorm import _NormBase

from .source import Source


@RUNNERS.register_module()
class TTNorm(Source):
    def config_tta_model(self):
        # find norm layers: norm in name (layer norm in ViTs), instance of subclass of _NormBase (batch norm)
        self.model.requires_grad_(False)
        all_norm_layers = []
        for name, sub_module in self.model.named_modules():
            if "norm" in name or isinstance(sub_module, _NormBase):
                all_norm_layers.append(name)

        for name in all_norm_layers:
            sub_module = self.model.get_submodule(name)
            # for other methods based on test time normalization,
            # which will fine tune the affine parameters in norm layers
            sub_module.requires_grad_(True)
            # if the sub_module is BN, then only current statistic is used for normalization
            if hasattr(sub_module, "track_running_stats") \
                    and hasattr(sub_module, "running_mean") \
                    and hasattr(sub_module, "running_var"):
                sub_module.track_running_stats = False
                sub_module.running_mean = None
                sub_module.running_var = None
