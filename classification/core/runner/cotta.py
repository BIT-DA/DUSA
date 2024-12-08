import torch
from torch.nn.modules.batchnorm import _BatchNorm, _NormBase
from torch.nn.modules import LayerNorm, GroupNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from mmengine.registry import RUNNERS
from ..model.wrapped_models import WrappedModels
from ..utils import cotta_transforms
from .ttarunner import BaseTTARunner
from mmpretrain.models.classifiers import TimmClassifier


def softmax_entropy(x, x_ema): # -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


@RUNNERS.register_module()
class CottaCls(BaseTTARunner):
    model: WrappedModels

    def __init__(self, cfg):
        super(CottaCls, self).__init__(cfg)
        self.anchor = self.build_ema_model(self.model)
        self.ema_model = self.build_ema_model(self.model)
        self.transform = cotta_transforms.get_tta_transforms()
        self.threshold = 0.1
        self.alpha_teacher = 0.999
        self.restoration = 0.001

    def config_tta_model(self):
        # find norm layers: norm in name (layer norm in ViTs), instance of subclass of _NormBase (batch norm)
        self.model.train()
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)

        # enable all trainable
        for name, sub_module in self.model.named_modules():
            if "norm" in name.lower() or isinstance(sub_module, (_NormBase, _BatchNorm, _InstanceNorm, LayerNorm, GroupNorm)):
                sub_module.requires_grad_(True)
                if hasattr(sub_module, "track_running_stats") \
                        and hasattr(sub_module, "running_mean") \
                        and hasattr(sub_module, "running_var"):   # force use of batch stats in train and eval modes
                    sub_module.track_running_stats = False
                    sub_module.running_mean = None
                    sub_module.running_var = None
            else:
                sub_module.requires_grad_(True)

    def reset_model(self):
        if self.model_state_dict is None or self.optim_state_dict is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.model.load_state_dict(self.model_state_dict, strict=True)
        self.optim_wrapper.load_state_dict(self.optim_state_dict)

        self.ema_model = self.build_ema_model(self.model)
        self.anchor = self.build_ema_model(self.model)

    def tta_one_batch(self, batch_data):
        self.model.eval()
        self.anchor.eval()
        self.ema_model.eval()
        all_loss = dict()
        with self.optim_wrapper.optim_context(self.model):
            task_batch_data = self.model.task_model.data_preprocessor(batch_data)
            inputs, data_samples = task_batch_data["inputs"], task_batch_data["data_samples"]
            outputs = self.model.task_model(inputs, data_samples, mode="tensor")

            # Teacher Prediction
            anchor_prob = torch.nn.functional.softmax(self.anchor.task_model(inputs, data_samples, mode="tensor"), dim=1).max(1)[0]
            standard_ema = self.ema_model.task_model(inputs).detach()

            # Augmentation-averaged Prediction
            if anchor_prob.mean(0) < self.threshold:
                N = 32
                outputs_emas = []
                for i in range(N):
                    outputs_ = self.ema_model.task_model(self.transform(inputs)).detach()
                    outputs_emas.append(outputs_)
                outputs_ema = torch.stack(outputs_emas).mean(0)
            else:
                outputs_ema = standard_ema

            # Student update
            loss = (softmax_entropy(outputs, outputs_ema)).mean(0)
        self.optim_wrapper.update_params(loss)
        self.set_cls_predictions(outputs_ema, data_samples)

        # Teacher update
        self.ema_model = self.update_ema_variables(ema_model=self.ema_model, model=self.model, alpha_teacher=self.alpha_teacher)

        # Stochastic restore
        for nm, m in self.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape) < self.restoration).float().cuda()
                    with torch.no_grad():
                        state_key = f"{nm}.{npp}"
                        if 'task_model' in nm and isinstance(self.model.task_model, TimmClassifier):
                            state_key = state_key.replace(".model.", ".")
                        p.data = self.model_state_dict[state_key] * mask + p.data * (1. - mask)
        all_loss["entropy"] = loss.item()
        return data_samples, all_loss



