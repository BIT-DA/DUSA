import torch
import math
import numpy as np
from copy import deepcopy
from torch.nn.modules.batchnorm import _BatchNorm, _NormBase
from torch.nn.modules import LayerNorm, GroupNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from mmengine.registry import RUNNERS
from ..model.wrapped_models import WrappedModels
from .ttarunner import BaseTTARunner
from ..utils.sar_sam import SAM


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@RUNNERS.register_module()
class SarCls(BaseTTARunner):
    model: WrappedModels

    def __init__(self, cfg):
        super(SarCls, self).__init__(cfg)
        self.episodic = False
        self.margin_e0 = math.log(1000)*0.40  # margin E_0 for reliable entropy minimization, Eqn. (2)
        self.reset_constant_em = 0.2  # threshold e_m for model recovery scheme
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria
        self.imbalance_ratio = 500000.  # imbalance ratio for label shift exps, selected from [1, 1000, 2000, 3000, 4000, 5000, 500000], 1  denotes totally uniform and 500000 denotes (almost the same to Pure Class Order).
        params, _ = self.collect_params()
        if self.cfg.get("tta_optimizer").get("type") == "SGD":
            # lr = (0.001 / 64) * cfg.tta_data_loader.batch_size
            lr = 0.001
            kwargs = dict(base_optimizer=torch.optim.SGD, lr=lr, momentum=0.9)
        elif self.cfg.get("tta_optimizer").get("type") == "Adam":
            lr = 0.00001
            kwargs = dict(base_optimizer=torch.optim.Adam, lr=lr, betas=(0.9, 0.999))
        else:
            raise RuntimeError
        self.optimizer = SAM(params, **kwargs)
        self.optimizer_state_dict = deepcopy(self.optimizer.state_dict())

    def config_tta_model(self):
        """Configure model for use with SAR."""
        # train mode, because SAR optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what SAR updates
        self.model.requires_grad_(False)
        # configure norm for SAR updates: enable grad + force batch statisics (this only for BN models)
        for name, m in self.model.named_modules():
            if "norm" in name.lower() or isinstance(m, (_NormBase, _BatchNorm, _InstanceNorm, LayerNorm, GroupNorm)):
                m.requires_grad_(True)
                if hasattr(m, "track_running_stats") \
                        and hasattr(m, "running_mean") \
                        and hasattr(m, "running_var"):   # force use of batch stats in train and eval modes
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None

    def collect_params(self):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue
            # skip the top stage of ConvNext
            if "stage.3" in nm or "norm3" in nm:
                continue
            if isinstance(m, (_NormBase, _BatchNorm, _InstanceNorm, LayerNorm, GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def reset_model(self):
        if self.model_state_dict is None or self.optimizer_state_dict is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.model.load_state_dict(self.model_state_dict, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state_dict)
        self.ema = None

    def tta_one_batch(self, batch_data):
        self.model.eval()
        all_loss = dict()
        with self.optim_wrapper.optim_context(self.model):
            if self.episodic:
                self.reset_model()
            task_batch_data = self.model.task_model.data_preprocessor(batch_data)
            inputs, data_samples = task_batch_data["inputs"], task_batch_data["data_samples"]
            self.optimizer.zero_grad()
            # forward
            outputs = self.model.task_model(inputs)
            # adapt
            # filtering reliable samples/gradients for further adaptation; first time forward
            entropys = softmax_entropy(outputs)
            filter_ids_1 = torch.where(entropys < self.margin_e0)
            entropys = entropys[filter_ids_1]
            loss = entropys.mean(0)
            loss.backward()
            all_loss["loss"] = loss.item()
            self.optimizer.first_step(zero_grad=True)  # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)

            entropys2 = softmax_entropy(self.model.task_model(inputs))
            entropys2 = entropys2[filter_ids_1]  # second time forward
            filter_ids_2 = torch.where(entropys2 < self.margin_e0)  # here filtering reliable samples again, since model weights have been changed to \Theta+\hat{\epsilon(\Theta)}
            loss_second = entropys2[filter_ids_2].mean(0)
            if not np.isnan(loss_second.item()):
                self.ema = update_ema(self.ema, loss_second.item())  # record moving average loss values for model recovery

            # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
            loss_second.backward()
            all_loss["loss_second"] = loss_second.item()
            self.optimizer.second_step(zero_grad=True)

            # perform model recovery
            reset_flag = False
            if self.ema is not None:
                if self.ema < 0.2:
                    print("ema < 0.2, now reset the model")
                    reset_flag = True

            if reset_flag:
                self.reset_model()
            self.set_cls_predictions(outputs, data_samples)
        return data_samples, all_loss


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data

