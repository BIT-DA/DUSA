import os

import mmengine
import numpy as np
import torch
from mmengine import MODEL_WRAPPERS
from mmengine.model import revert_sync_batchnorm, is_model_wrapper, convert_sync_batchnorm, MMDistributedDataParallel
from mmengine.runner import Runner
from mmengine.evaluator import Evaluator
from mmengine.dataset import BaseDataset
from mmpretrain.utils import get_ori_model
from typing import List, Dict, Optional, Union, Tuple, Sequence
from copy import deepcopy
import csv

from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from mmengine.registry import Registry
from mmengine.optim.optimizer import OptimWrapper, OptimWrapperDict, AmpOptimWrapper

from mmpretrain.structures import DataSample
from ..utils.functional import ban_build_function_of_registry
from ..utils.mmseg_utils import add_prefix
from ..utils.local_activation_checkpointing import turn_on_activation_checkpointing
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

from core.model.wrapped_models import WrappedModels


MyFUNC = Registry("Functions to process output message!", build_func=ban_build_function_of_registry)


class BaseTTARunner(Runner):
    """
    Base class for all TTARunners
    """
    def __init__(self, cfg):
        # remove previously defined train_val_test configurations
        super().__init__(
            model=nn.Linear(1, 1),  # useless, all the model and optimizers will be re-initialized in later
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
            default_hooks=None,
            custom_hooks=None,
            data_preprocessor=None,
            load_from=None,
            resume=False,
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
        self.tasks = cfg.get("tasks")
        self.data_loader = cfg.get("tta_data_loader")
        self.continual = cfg.get("continual")
        self.evaluator = cfg.get("tta_evaluator")
        # logging for accuracy and ece analysis
        self.analysis = cfg.get("analysis", False)
        self.analysis_path = None
        # save checkpoint path
        self.save_ckpt_path = cfg.get("save_ckpt_path", None)
        self.debug = cfg.get("debug", False)

        # build a model
        model = cfg['model']
        data_preprocessor = cfg.get('data_preprocessor')
        if isinstance(model, dict) and data_preprocessor is not None:
            # Merge the data_preprocessor to model config.
            task_model = model.get("task_model", None)
            if task_model is not None:
                task_model.setdefault('data_preprocessor', data_preprocessor)
            else:
                model.setdefault('data_preprocessor', data_preprocessor)
        self.model = self.build_model(model)
        # wrap model
        if self.distributed:
            # for dist train, must change the devices of models
            wrap_fn = self.wrap_model
        else:
            # for single GPU train, the model will be moved to the target cuda device in their init function
            wrap_fn = self.wrap_model_but_not_change_device
        self.model = wrap_fn(
            self.cfg.get('model_wrapper_cfg'), self.model)

        # check and warn about device difference in distributed training
        if self.distributed:
            ori_model = get_ori_model(self.model)
            if isinstance(ori_model, WrappedModels) and ori_model.auxiliary_cfg is not None:
                assert ori_model.device_task == ori_model.auxiliary_model.device, \
                    'Distributed training is used, the device of the model will ' \
                    'be changed to the target device by DDP. If you want to use ' \
                    'the original device, please set the device of the model to ' \
                    'None in its init function.'

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        # init the model's weight
        self._init_model_weights()
        # configure the model
        # set parameters needs update with requires_grad=True, vice versa
        # modify BN and so on
        self.config_tta_model()

        # try to enable activation_checkpointing feature
        modules = cfg.get('activation_checkpointing', None)
        if modules is not None:
            ori_model = get_ori_model(self.model)
            self.logger.info(f'Enabling the "activation_checkpointing" feature'
                             f' for sub-modules: {modules}')
            turn_on_activation_checkpointing(ori_model, modules)

        # build optimizer
        self.optim_wrapper = cfg.get("tta_optim_wrapper")
        self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
        # we opt not to perform auto_scale_lr as linear law needs rectification for Adam
        self.scale_lr(self.optim_wrapper, None)

        # we have to only use one loss_scaler for all the amp optimizers
        if isinstance(self.optim_wrapper, OptimWrapperDict):
            first_amp_optimizer = None
            for name, optim_wrapper in self.optim_wrapper.items():
                if isinstance(optim_wrapper, AmpOptimWrapper):
                    if first_amp_optimizer is None:
                        first_amp_optimizer = optim_wrapper
                    else:
                        optim_wrapper.loss_scaler = first_amp_optimizer.loss_scaler

        self.model_state_dict = deepcopy(self.model.state_dict())
        self.optim_state_dict = deepcopy(self.optim_wrapper.state_dict())

        self.tasks = self.build_tta_tasks(self.tasks)

        self.debug = cfg.get("debug", False)

        self.info_functions: BaseInfoFunctions = MyFUNC.get(cfg.get("info_functions", None))()

        self.tta_step = cfg.get("tta_step", 1)

        self.resume_from_task = cfg.get("resume_from_task", None)  # NOTE that this is task indexed from 0

    def config_tta_model(self):
        pass

    def wrap_model_but_not_change_device(
            self, model_wrapper_cfg: Optional[Dict],
            model: nn.Module) -> Union[DistributedDataParallel, nn.Module]:
        """
        the same as Runner.wrap_model(), but this function don't change the device
        the model will be move to the target cuda device in their init function
        """
        if is_model_wrapper(model):
            if model_wrapper_cfg is not None:
                raise TypeError(
                    'model has been wrapped and "model_wrapper_cfg" should be '
                    f'None, but got {model_wrapper_cfg}')

            return model

        if not self.distributed:
            self.logger.info(
                'Distributed training is not used, all SyncBatchNorm (SyncBN) '
                'layers in the model will be automatically reverted to '
                'BatchNormXd layers if they are used.')
            model = revert_sync_batchnorm(model)
            return model  # type: ignore
        else:
            sync_bn = self.cfg.get('sync_bn', None)
            if sync_bn is not None:
                try:
                    model = convert_sync_batchnorm(model, sync_bn)
                except ValueError as e:
                    self.logger.error('cfg.sync_bn should be "torch" or '
                                      f'"mmcv", but got {sync_bn}')
                    raise e
        if model_wrapper_cfg is None:
            find_unused_parameters = self.cfg.get('find_unused_parameters',
                                                  False)
            # Sets the `find_unused_parameters` parameter in
            # torch.nn.parallel.DistributedDataParallel
            # TODO: may use a more elegant way to get local device ID.
            model = MMDistributedDataParallel(
                module=model,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model_wrapper_cfg.setdefault('type', 'MMDistributedDataParallel')
            model_wrapper_type = MODEL_WRAPPERS.get(
                model_wrapper_cfg.get('type'))  # type: ignore
            default_args: dict = dict()
            if issubclass(
                    model_wrapper_type,  # type: ignore
                    DistributedDataParallel):
                default_args['device_ids'] = [int(os.environ['LOCAL_RANK'])]
            default_args['module'] = model
            model = MODEL_WRAPPERS.build(
                model_wrapper_cfg, default_args=default_args)
        return model

    @staticmethod
    def build_tta_tasks(tasks):
        """
        format the tasks, it should be a list of dict, each elements represents a dataset to perform test-time adaptation
        :param tasks: Dict or List[Dict], or List[dataset]
        :return: List[Dict] or List[dataset]
        """
        if isinstance(tasks, dict) or isinstance(tasks, BaseDataset):
            tasks = [tasks]  # single task

        if isinstance(tasks, list):
            # check the type of each element
            assert all(isinstance(one_task, (dict, BaseDataset)) for one_task in tasks)
            return tasks
        else:
            raise TypeError

    def reset_model(self, verbose=True):
        if verbose:
            self.logger.info("Fully Test-time Adaptation: Resetting the model!")
        self.model.load_state_dict(self.model_state_dict)
        self.optim_wrapper.load_state_dict(self.optim_state_dict)

    def tta(self):
        all_metric = []
        for i, task in enumerate(self.tasks):
            if not self.continual and self.resume_from_task is not None and i < self.resume_from_task:
                self.logger.info(f"Skip Task [{i}][{len(self.tasks)}]: `{task['data_prefix']['img_path']}`!")
                continue
            if not self.continual and i > 0:
                self.reset_model()
                self.set_randomness(**self._randomness_cfg)
            if self.analysis:
                self.analysis_path = os.path.join(self.work_dir, f"analysis_{task['data_prefix']['img_path'].replace('/', '_')}.csv")
                header = ['pred_conf', 'pred_label', 'gt_label', 'img_path']
                csv_handle = open(self.analysis_path, 'w')
                csvwriter = csv.writer(csv_handle)
                csvwriter.writerow(header)
                csv_handle.close()
            self.logger.info(f"Begin Task [{i}][{len(self.tasks)}]: `{task['data_prefix']['img_path']}`!")
            if not self.debug:
                metric = self.perform_one_task(task, f"Tasks: [{i}][{len(self.tasks)}]")
                self.info_functions.finish_one_task(self.logger, i, metric)
                all_metric.append(self.info_functions.get_metric(metric))
            if self.save_ckpt_path is not None:
                from mmengine.runner.checkpoint import save_checkpoint, weights_to_cpu
                ori_model = get_ori_model(self.model)
                save_checkpoint(weights_to_cpu(ori_model.task_state_dict()), os.path.join(self.work_dir, self.save_ckpt_path))
                save_checkpoint(weights_to_cpu(ori_model.state_dict()), os.path.join(self.work_dir, 'whole_' + self.save_ckpt_path))

        self.info_functions.finish_all_tasks(self.logger, all_metric)

    def perform_one_task(self, task, task_name=""):
        # evaluate the metrics for the current task
        evaluator: Evaluator = self.build_evaluator(self.evaluator)
        post_evaluator: Evaluator = self.build_evaluator(self.evaluator)
        data_loader = deepcopy(self.data_loader)
        data_loader['dataset'] = task
        data_loader = self.build_dataloader(dataloader=data_loader)
        if hasattr(data_loader.dataset, 'metainfo'):
            evaluator.dataset_meta = data_loader.dataset.metainfo

        online_metric, post_online_metric = [], []
        online_metrics_topk, post_online_metrics_topk = {}, {}
        tbar = tqdm(data_loader, desc=task_name)

        if self.cfg.get("single_sample", False):
            assert data_loader.batch_size == 1, "Single sample mode, batch size should be 1!"
            self.logger.info("Single sample mode, model will be reset after each sample!")

        for i, batch_data in enumerate(tbar):
            if self.cfg.get("single_sample", False):
                self.reset_model(verbose=False)
            data_samples, loss_info = self.tta_one_batch(batch_data)

            info: dict = self.info_functions.process_online_metric(online_metric, data_samples)
            if self.evaluator.topk is not None and self.evaluator.topk != (1,):
                info: dict = self.info_functions.process_online_topk_metrics(online_metrics_topk, data_samples, self.evaluator.topk)

            if self.cfg.get("post_update_acc", False):
                ori_model = get_ori_model(self.model)
                with torch.no_grad():
                    post_data_samples, post_task_output = ori_model.task_forward(batch_data)
                self.set_cls_predictions(post_task_output, post_data_samples)
                post_info: dict = self.info_functions.process_online_metric(post_online_metric, post_data_samples)
                if self.evaluator.topk is not None and self.evaluator.topk != (1,):
                    post_info: dict = self.info_functions.process_online_topk_metrics(post_online_metrics_topk, post_data_samples,
                                                                                 self.evaluator.topk)
                post_info = add_prefix(post_info, "post")
                info.update(post_info)

            all_scalars = dict()
            for k, v in chain(info.items(), loss_info.items()):
                new_k = f"{task_name}:{k}"
                all_scalars[new_k] = v
            self.visualizer.add_scalars(all_scalars, step=i)
            # save for analysis
            if self.analysis:
                csv_handle = open(self.analysis_path, 'a')
                csvwriter = csv.writer(csv_handle)
                for data_sample in data_samples:
                    pred_score = data_sample.get("pred_score")
                    pred_label = data_sample.get("pred_label").item()
                    gt_label = data_sample.get("gt_label").item()
                    img_path = data_sample.get("img_path")
                    pred_conf = pred_score[pred_label].item()
                    csvwriter.writerow([pred_conf, pred_label, gt_label, img_path])
                csv_handle.close()
            # if i % 10 == 0:
            tbar.set_postfix(**info)
            evaluator.process(data_samples)
            if self.cfg.get("post_update_acc", False):
                post_evaluator.process(post_data_samples)
        task_metrics = evaluator.evaluate(len(data_loader.dataset))
        if self.cfg.get("post_update_acc", False):
            post_task_metrics = post_evaluator.evaluate(len(data_loader.dataset))
            post_task_metrics = add_prefix(post_task_metrics, "post")
            task_metrics.update(post_task_metrics)
        return task_metrics

    def tta_one_batch(self, batch_data):
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
                # ema_param.data = alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
                # ema_param: torch.nn.Parameter
                # param: torch.nn.Parameter
                ema_param.mul_(alpha_teacher).add_(param, alpha=1 - alpha_teacher)
        return ema_model

    @torch.no_grad()
    def set_cls_predictions(self, cls_score, data_samples):
        pred_scores = F.softmax(cls_score, dim=1)
        pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, label in zip(data_samples, pred_scores,
                                             pred_labels):
            if data_sample is None:
                data_sample = DataSample()

            data_sample.set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)
        return out_data_samples


def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value


class BaseInfoFunctions:
    def finish_one_task(self, *args):
        # maybe different for different type of task
        raise NotImplementedError

    @staticmethod
    def get_metric(*args):
        # maybe different for different type of task
        raise NotImplementedError

    @staticmethod
    def finish_all_tasks(*args):
        # maybe different for different type of task
        raise NotImplementedError

    @staticmethod
    def process_online_metric(*args):
        raise NotImplementedError

    @staticmethod
    def process_online_topk_metrics(*args):
        raise NotImplementedError


@MyFUNC.register_module()
class ClsInfoFunctions(BaseInfoFunctions):
    def finish_one_task(self, logger, task_id, metric):
        logger.info(f"Finished Task {task_id}: Accuracy {self.get_metric(metric):.2f}")

    @staticmethod
    def get_metric(metrics):
        return metrics["accuracy/top1"]

    @staticmethod
    def finish_all_tasks(logger, all_metrics):
        logger.info(f"All tasks are finished\n")
        logger.info("Acc summary: " + "\t".join([f"{acc:.2f}" for acc in all_metrics]))
        logger.info(f"Average: {sum(all_metrics) / len(all_metrics)}")

    @staticmethod
    def process_online_metric(online_metric: List, data_samples: List[DataSample]):
        for data_sample in data_samples:
            online_metric.append((data_sample.get("gt_label") == data_sample.get("pred_label")).item())

        return dict(acc=sum(online_metric)/len(online_metric))


@MyFUNC.register_module()
class TopKClsInfoFunctions(ClsInfoFunctions):
    def finish_one_task(self, logger, task_id, metrics):
        # metrics : dict:{top1: %, top2: % .....}
        logger.info(f"Finished Task {task_id}")
        for k, v in metrics.items():
            logger.info(f"{k} : {v:.2f}")

    @staticmethod
    def process_online_metric(online_metric: List, data_samples: List[DataSample]):
        for data_sample in data_samples:
            online_metric.append((data_sample.get("gt_label") == data_sample.get("pred_label")).item())

        return dict(acc=sum(online_metric) / len(online_metric))

    @staticmethod
    def process_online_topk_metrics(online_metrics_topk: Dict[int, List], data_samples: List[Dict], topks: Tuple[int]):

        for topk in topks:
            if topk not in online_metrics_topk:
                online_metrics_topk[topk] = []

        for data_sample in data_samples:
            maxk = max(topks)
            target = data_sample.get("gt_label")
            pred_scores = data_sample.get("pred_score")
            pred = to_tensor(pred_scores)
            target = to_tensor(target).to(torch.int64)
            pred = pred.float()
            if maxk > pred.size(0):
                raise ValueError(
                    f'Top-{maxk} accuracy is unavailable since the number of '
                    f'categories is {pred.size(0)}.')

            pred_score, pred_label = pred.topk(maxk, dim=0)
            pred_label = pred_label.t()
            correct = pred_label.eq(target)
            for k in topks:
                correct_k = correct[:k].reshape(-1).float().sum(
                    0, keepdim=True)
                online_metrics_topk[k].append(correct_k.item())
        topk_acc_results = {f"top{topk}acc": sum(online_metrics_topk[topk]) / len(online_metrics_topk[topk]) for topk in topks}
        return topk_acc_results


def logits_entropy(in_logits, target):
    """
    sum(- p_t log p_in)
    return: (B, )
    """
    return - torch.sum(torch.softmax(target, dim=-1) * torch.log_softmax(in_logits, dim=-1), dim=-1)



