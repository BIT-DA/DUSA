from mmengine.model import detect_anomalous_params, MMDistributedDataParallel, is_model_wrapper
from torch.nn.modules.batchnorm import _BatchNorm, _NormBase
from torch.nn.modules import LayerNorm, GroupNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from .ttarunner import BaseTTARunner
from mmengine.registry import RUNNERS
import torch
from ..model.wrapped_models import WrappedModels


@RUNNERS.register_module()
class TextImageAuxiliaryTTAClsLogitsFP16(BaseTTARunner):
    model: MMDistributedDataParallel

    def tta_one_batch(self, batch_data, evaluator):
        # what's different between two frame is that tta_one_batch didn't return and info is passed back by evaluator
        self.model.eval()

        with self.optim_wrapper.optim_context(self.model):
            # self.model is WrappedModels further wrapped in MMDistributedDataParallel
            logits_mode = self.cfg.get("logits_mode", "logits")
            data_samples, task_output, condition_loss = self.model(batch_data, mode=logits_mode)
            loss = condition_loss

        if self.distributed and self.model.detect_anomalous_params:
            detect_anomalous_params(loss, model=self.model)

        self.optim_wrapper.update_params(loss)

        self.model.task_model.postprocess_result(task_output, data_samples)
        evaluator.process(data_samples)

    def config_tta_model(self):
        if is_model_wrapper(self.model):
            ori_model = self.model.module  # WrappedModels
        else:
            ori_model = self.model
        # close all grads
        ori_model.requires_grad_(False)
        # we only focus on batch-agnostic models
        # free the vae, text encoder, tokenizer
        if self.cfg.get("update_auxiliary", False):
            ori_model.auxiliary_model.config_train_grad()

        if self.cfg.get("update_norm_only", False):
            ori_model.task_model.requires_grad_(False)
            all_norm_layers = []
            for name, sub_module in ori_model.task_model.named_modules():
                if "norm" in name.lower() or isinstance(sub_module, (_NormBase, _InstanceNorm, LayerNorm, GroupNorm)):
                    all_norm_layers.append(name)

            for name in all_norm_layers:
                sub_module = ori_model.task_model.get_submodule(name)
                # fine tune the affine parameters in norm layers
                sub_module.requires_grad_(True)
                # if the sub_module is BN, then only current statistic is used for normalization
                # actually, we only perform TTA for models without BN
                if isinstance(sub_module, _BatchNorm) \
                        and hasattr(sub_module, "track_running_stats") \
                        and hasattr(sub_module, "running_mean") \
                        and hasattr(sub_module, "running_var"):
                    sub_module.track_running_stats = False
                    sub_module.running_mean = None
                    sub_module.running_var = None
        else:
            ori_model.task_model.requires_grad_(True)
