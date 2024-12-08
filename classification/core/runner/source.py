from .ttarunner import BaseTTARunner
from mmengine.registry import RUNNERS
import torch
from ..model.wrapped_models import WrappedModels


@RUNNERS.register_module()
class SourceCls(BaseTTARunner):
    model: WrappedModels

    @torch.no_grad()
    def tta_one_batch(self, batch_data):
        self.model.eval()

        task_batch_data = self.model.task_model.data_preprocessor(batch_data)
        inputs, data_samples = task_batch_data["inputs"], task_batch_data["data_samples"]
        task_output = self.model.task_model(inputs, data_samples, mode="tensor")
        self.set_cls_predictions(task_output, data_samples)
        return data_samples, dict()

