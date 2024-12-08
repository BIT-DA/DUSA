import torch
import PIL
from torch.nn.modules.batchnorm import _BatchNorm, _NormBase
from torch.nn.modules import LayerNorm, GroupNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from mmengine.registry import RUNNERS
from ..model.wrapped_models import WrappedModels
from .ttarunner import BaseTTARunner, logits_entropy
from ..utils import rotta_memory, cotta_transforms
from ..utils.rotta_bn_layers import RobustBN1d, RobustBN2d



def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))


def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


@RUNNERS.register_module()
class RottaCls(BaseTTARunner):
    model: WrappedModels

    def __init__(self, cfg):
        self.alpha = 0.05  # will be used in config_tta_model
        super(RottaCls, self).__init__(cfg)
        self.mem = rotta_memory.CSTU(capacity=cfg.tta_data_loader.batch_size, num_class=cfg.data_preprocessor.num_classes,
                                     lambda_t=1.0, lambda_u=1.0)
        self.model_ema = self.build_ema_model(self.model)
        self.transform = cotta_transforms.get_tta_transforms()
        self.nu = 0.001
        self.update_frequency = cfg.tta_data_loader.batch_size  # actually the same as the size of memory bank
        self.current_instance = 0

    def config_tta_model(self):
        # find norm layers: norm in name (layer norm in ViTs), instance of subclass of _NormBase (batch norm)
        self.model.requires_grad_(False)
        normlayer_names = []
        for name, sub_module in self.model.named_modules():
            if "norm" in name.lower() or isinstance(sub_module, (_NormBase, _BatchNorm, _InstanceNorm, LayerNorm, GroupNorm)):
                normlayer_names.append(name)

        for name in normlayer_names:
            sub_module = self.model.get_submodule(name)
            if isinstance(sub_module, torch.nn.BatchNorm1d):
                momentum_bn = RobustBN1d(sub_module, self.alpha)
                momentum_bn.requires_grad_(True)
                self.set_named_submodule(name, momentum_bn)
            elif isinstance(sub_module, torch.nn.BatchNorm2d):
                momentum_bn = RobustBN2d(sub_module, self.alpha)
                momentum_bn.requires_grad_(True)
                self.set_named_submodule(name, momentum_bn)
            else:
                sub_module.requires_grad_(True)

    def set_named_submodule(self, sub_name, value):
        names = sub_name.split(".")
        module = self.model
        for i in range(len(names)):
            if i != len(names) - 1:
                module = getattr(module, names[i])
            else:
                setattr(module, names[i], value)

    def reset_model(self):
        if self.model_state_dict is None or self.optim_state_dict is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.model.load_state_dict(self.model_state_dict, strict=True)
        self.optim_wrapper.load_state_dict(self.optim_state_dict)
        self.model_ema = self.build_ema_model(self.model)
        self.mem.reset()

    def update_model(self):
        self.model.train()
        self.model_ema.train()
        # get memory data
        sup_data, ages = self.mem.get_memory()
        l_sup = None
        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data)
            strong_sup_aug = self.transform(sup_data)
            ema_sup_out = self.model_ema.task_model(sup_data)
            stu_sup_out = self.model.task_model(strong_sup_aug)
            instance_weight = timeliness_reweighting(ages)
            l_sup = (softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

        if l_sup is not None:
            self.optim_wrapper.update_params(l_sup)
        self.ema_model = self.update_ema_variables(ema_model=self.model_ema, model=self.model,
                                                   alpha_teacher=1-self.nu)
        return l_sup

    def tta_one_batch(self, batch_data):
        self.model.eval()
        self.model_ema.eval()
        all_loss = dict()

        with self.optim_wrapper.optim_context(self.model):
            task_batch_data = self.model.task_model.data_preprocessor(batch_data)
            inputs, data_samples = task_batch_data["inputs"], task_batch_data["data_samples"]

            with torch.no_grad():
                ema_out = self.model_ema.task_model(inputs)
                predict = torch.softmax(ema_out, dim=1)
                pseudo_label = torch.argmax(predict, dim=1)
                entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)

            # add into memory
            for i, data in enumerate(inputs):
                p_l = pseudo_label[i].item()
                uncertainty = entropy[i].item()
                current_instance = (data, p_l, uncertainty)
                self.mem.add_instance(current_instance)
                self.current_instance += 1
                if self.current_instance % self.update_frequency == 0:
                    l_sup = self.update_model()
                    all_loss["l_sup"] = l_sup.item()

        self.set_cls_predictions(ema_out, data_samples)
        return data_samples, all_loss



