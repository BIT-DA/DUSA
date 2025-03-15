import os
import os.path as osp

import numpy as np
import torch
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.runner.checkpoint import save_checkpoint
from mmengine.evaluator import Evaluator

from ..model.wrapped_models import WrappedModels
from ..model.wrapped_networks import WrappedEncoderDecoder
from ..utils.local_activation_checkpointing import turn_on_activation_checkpointing

from typing import List, Dict
from copy import deepcopy

from mmseg.datasets import BaseSegDataset
from tqdm import tqdm

from mmseg.structures import SegDataSample
from mmseg.models.utils.wrappers import resize

from einops import rearrange

from mmseg.visualization import SegLocalVisualizer
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mmengine.logging import MMLogger, print_log
from PIL import Image
from itertools import chain

class BaseTTARunner(Runner):
    def __init__(self, cfg):
        # initialize model, logger, hook and so on
        super().__init__(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=None,
            val_dataloader=None,
            test_dataloader=None,
            train_cfg=None,
            val_cfg=None,
            test_cfg=None,
            auto_scale_lr=None,
            optim_wrapper=None,
            param_scheduler=None,
            val_evaluator=None,
            test_evaluator=None,
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl'))),
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
        )

        self.optim_wrapper = cfg.get("tta_optim_wrapper")
        self.tasks = cfg.get("tasks")
        self.data_loader = cfg.get("tta_data_loader")
        self.continual = cfg.get("continual")
        self.evaluator = cfg.get("tta_evaluator")

        # init the model's weight
        self._init_model_weights()
        # configure the model
        # set parameters needs update with requires_grad=True, vice versa
        # modify BN and so on
        self.config_tta_model()

        self.state_dict = deepcopy(self.model.state_dict())

        if is_model_wrapper(self.model):
            ori_model = self.model.module
        else:
            ori_model = self.model
        # try to enable activation_checkpointing feature
        modules = cfg.get('activation_checkpointing', None)
        if modules is not None:
            self.logger.info(f'Enabling the "activation_checkpointing" feature'
                             f' for sub-modules: {modules}')
            turn_on_activation_checkpointing(ori_model, modules)

        # build optimizer
        self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)

        self.tasks = self.build_tta_tasks(self.tasks)

        self.debug = cfg.get("debug", False)

    def config_tta_model(self):
        pass

    @staticmethod
    def build_tta_tasks(tasks):
        """
        format the tasks, it should be a list of dict, each elements represents a dataset to perform test-time adaptation
        :param tasks: Dict or List[Dict], or List[dataset]
        :return: List[Dict] or List[dataset]
        """
        if isinstance(tasks, dict) or isinstance(tasks, BaseSegDataset):
            tasks = [tasks]  # single task

        if isinstance(tasks, list):
            return tasks
        else:
            raise TypeError

    def reset_model(self):
        self.logger.info("Fully Test-time Adaptation: Resetting the model!")
        self.model.load_state_dict(self.state_dict)

    def tta(self):
        all_metric = []
        for i, task in enumerate(self.tasks):
            self.set_randomness(**self._randomness_cfg)
            if not self.continual:
                self.reset_model()
            metric = self.perform_one_task(task, f"[{i}][{len(self.tasks)}]")
            self.logger.info(f"Task {i}: mIoU: {metric['mIoU']}")
            all_metric.append(metric['mIoU'])
            if self.cfg.get("save_checkpoint", False):
                task_name = task.data_prefix.img_path.split('/')[1]+'.pth'
                ckpt_path = os.path.join(self.cfg.work_dir, task_name)
                if isinstance(self.model, WrappedEncoderDecoder):
                    save_checkpoint(self.model.state_dict(), ckpt_path)
                elif isinstance(self.model, WrappedModels):
                    save_checkpoint(self.model.task_model.state_dict(), ckpt_path)
                self.logger.info(f"{task_name} is saved in {ckpt_path}")

        self.logger.info("mIoU summary: " + "\t".join([f"{mIoU:.2f}" for mIoU in all_metric]))
        self.logger.info(f"Average: {sum(all_metric)/len(all_metric)}")

    def perform_one_task(self, task, task_name=""):
        evaluator: Evaluator = self.build_evaluator(self.evaluator)
        # without data is also ok
        data_loader = deepcopy(self.data_loader)
        data_loader['dataset'] = task
        data_loader = self.build_dataloader(dataloader=data_loader)
        if hasattr(data_loader.dataset, 'metainfo'):
            evaluator.dataset_meta = data_loader.dataset.metainfo

        tbar = tqdm(data_loader)

        # for online metric close logger info
        logger: MMLogger = MMLogger.get_current_instance()
        logger.setLevel('ERROR')
        # 500 for each task, consistent with cotta
        for i, batch_data in enumerate(tbar):
            self.tta_one_batch(batch_data, evaluator)
            online_metrics = evaluator.metrics[0].compute_metrics(evaluator.metrics[0].results)

            tbar.set_postfix(online_metrics)

            all_scalars = dict()
            for k, v in chain(online_metrics.items(), online_metrics.items()):
                new_k = f"{task_name}:{k}"
                all_scalars[new_k] = v
            self.visualizer.add_scalars(all_scalars, step=i)
        logger.setLevel('INFO')

        task_matrics = evaluator.evaluate(len(data_loader.dataset))
        return task_matrics

    def tta_one_batch(self, batch_data, evaluator: Evaluator):
        raise NotImplementedError

    def test_one_task(self, task, task_name=""):
        evaluator: Evaluator = self.build_evaluator(self.evaluator)
        # without data is also ok
        data_loader = deepcopy(self.data_loader)
        data_loader['dataset'] = task
        data_loader = self.build_dataloader(dataloader=data_loader)
        if hasattr(data_loader.dataset, 'metainfo'):
            evaluator.dataset_meta = data_loader.dataset.metainfo

        tbar = tqdm(data_loader)

        # 500 for each task, consistent with cotta
        for i, batch_data in enumerate(tbar):
            self.test_one_batch(batch_data, evaluator)

        task_matrics = evaluator.evaluate(len(data_loader.dataset))
        return task_matrics

    @torch.no_grad()
    def test_one_batch(self, batch_data, evaluator):
        raise NotImplementedError

    @classmethod
    def from_cfg(cls, cfg) -> 'Runner':
        return cls(cfg)

    @staticmethod
    def build_ema_model(model: torch.nn.Module):
        ema = deepcopy(model)
        ema.requires_grad_(False)
        return ema

    @staticmethod
    def update_ema_variables(ema_model, model, alpha_teacher):
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.mul_(alpha_teacher).add_(param, alpha=1 - alpha_teacher)
        return ema_model


def softmax_cross_entropy(logits: torch.Tensor, target_logits: torch.Tensor, weights=None) -> torch.Tensor:
    # logits: (B, C, H, W)
    logits = logits.contiguous()
    target_logits = target_logits.contiguous()
    flattened_logits = rearrange(logits, "b c h w -> (b h w) c")
    flattened_target = rearrange(target_logits, "b c h w -> (b h w) c")
    entropy_map = torch.sum(-flattened_target.softmax(1) * flattened_logits.log_softmax(1), dim=1)
    if weights is None:
        entropy_map = entropy_map
    else:
        weights = rearrange(weights, "b h w -> (b h w)")
        entropy_map = entropy_map * weights
    return torch.mean(entropy_map)


def cross_entropy(logits: torch.Tensor, target_label: torch.Tensor, weights=None) -> torch.Tensor:
    # logits: (B, C, H, W)
    logits = logits.contiguous()
    target_label = target_label.contiguous()
    flattened_logits = rearrange(logits, "b c h w -> (b h w) c")
    flattened_target = rearrange(target_label, "b c h w -> (b h w) c")
    entropy_map = torch.sum(-flattened_target * flattened_logits.log_softmax(1), dim=1)
    if weights is None:
        entropy_map = entropy_map
    else:
        weights = rearrange(weights, "b h w -> (b h w)")
        entropy_map = entropy_map * weights
    return torch.mean(entropy_map)


def heatmap_to_rgb(heatmap):
    colored_heatmap = np.stack([np.zeros_like(heatmap), np.zeros_like(heatmap), heatmap], axis=-1)
    return colored_heatmap

