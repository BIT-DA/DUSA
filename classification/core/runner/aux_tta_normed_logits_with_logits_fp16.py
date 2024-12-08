from mmengine.model import detect_anomalous_params, MMDistributedDataParallel, is_model_wrapper
from torch.nn.modules.batchnorm import _BatchNorm, _NormBase
from torch.nn.modules import LayerNorm, GroupNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from .ttarunner import BaseTTARunner, logits_entropy
from mmengine.registry import RUNNERS
from mmpretrain.utils import get_ori_model
import torch
from mmengine.optim.optimizer import OptimWrapper, OptimWrapperDict, AmpOptimWrapper
from contextlib import ExitStack


@RUNNERS.register_module()
class TextImageAuxiliaryTTAClsNormedLogitsWithLogitsFP16(BaseTTARunner):
    model: MMDistributedDataParallel

    def tta_one_batch(self, batch_data):
        self.model.eval()
        all_loss = dict()

        aux_info = dict()
        # in case of multiple optim wrappers, we have to enter the context manually
        with ExitStack() as stack:
            if isinstance(self.optim_wrapper, OptimWrapperDict):
                for name, optim_wrapper in self.optim_wrapper.optim_wrappers.items():
                    stack.enter_context(optim_wrapper.optim_context(self.model.get_submodule(name)))
            else:
                stack.enter_context(self.optim_wrapper.optim_context(self.model))
            # self.model is WrappedModels further wrapped in MMDistributedDataParallel
            data_samples, task_output, condition_loss = self.model(batch_data, mode="normed_logits_with_logits")
            if isinstance(condition_loss, tuple):
                condition_loss, aux_info = condition_loss
            loss = condition_loss
        if self.distributed and self.model.detect_anomalous_params:
            detect_anomalous_params(loss, model=self.model)

        # in case of multiple optim wrappers, we scale loss with the first optim wrapper
        if isinstance(self.optim_wrapper, OptimWrapperDict):
            # implement missing update_params for multiple optimizers
            # update inner_count of all optimizers, while backward only once
            first_amp_optim_wrapper = None
            first_optim_wrapper = None
            for name, optim_wrapper in self.optim_wrapper.optim_wrappers.items():
                # we only scale loss for the first optim wrapper
                if first_optim_wrapper is None:
                    first_optim_wrapper = optim_wrapper
                    loss = first_optim_wrapper.scale_loss(loss)
                # we only use scaler from the first amp optim wrapper
                if isinstance(optim_wrapper, AmpOptimWrapper):
                    if first_amp_optim_wrapper is None:
                        first_amp_optim_wrapper = optim_wrapper
                        loss = first_amp_optim_wrapper.loss_scaler.scale(loss)
                optim_wrapper._inner_count += 1  # noqa
            loss.backward()

            # Update parameters only if `self._inner_count` is divisible by
            # `self._accumulative_counts` or `self._inner_count` equals to
            # `self._max_counts`
            # NOTE that for simplicity, we trickily use the last optim_wrapper for condition
            for name, optim_wrapper in self.optim_wrapper.optim_wrappers.items():
                if optim_wrapper.should_update():
                    if isinstance(optim_wrapper, AmpOptimWrapper):
                        if optim_wrapper.clip_grad_kwargs:
                            optim_wrapper.loss_scaler.unscale_(optim_wrapper.optimizer)
                            optim_wrapper._clip_grad()
                        optim_wrapper.loss_scaler.step(optim_wrapper.optimizer)
                        optim_wrapper.loss_scaler.update(optim_wrapper._scale_update_param)
                    else:
                        optim_wrapper.step()
                    optim_wrapper.zero_grad()
        else:
            self.optim_wrapper.update_params(loss)
        all_loss["loss_auxiliary"] = condition_loss.item()
        all_loss.update(aux_info)

        ori_model = get_ori_model(self.model)
        if self.cfg.get("record_timestep", False):
            all_loss["trainable_timestep"] = ori_model.auxiliary_model.scheduler.original_num_steps * torch.sigmoid(ori_model.auxiliary_model.trainable_timestep).item()

        self.set_cls_predictions(task_output, data_samples)
        return data_samples, all_loss

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
        else:
            ori_model.task_model.requires_grad_(True)
        # process the normalization layers
        all_norm_layers = []
        for name, sub_module in ori_model.task_model.named_modules():
            if isinstance(sub_module, (_NormBase, _BatchNorm, _InstanceNorm, LayerNorm, GroupNorm)):
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
