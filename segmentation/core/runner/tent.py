from mmengine.registry import RUNNERS
import torch
from torch.nn.modules.batchnorm import _NormBase

from mmseg.models.utils import resize
from .ttarunner import softmax_cross_entropy
from .ttnorm import TTNorm


@RUNNERS.register_module()
class Tent(TTNorm):
    def tta_one_batch(self, batch_data, evaluator):
        self.model.eval()
        with self.optim_wrapper.optim_context(self.model):
            batch_data = self.model.data_preprocessor(batch_data, True)
            inputs, data_samples = batch_data['inputs'], batch_data['data_samples']
            seg_logits = self.model(inputs, data_samples, mode='tensor')
            if hasattr(data_samples[0], "img_shape"):
                seg_logits = resize(
                        seg_logits,
                        size=data_samples[0].img_shape,
                        mode='bilinear',
                        align_corners=self.model.align_corners,
                        warning=False)
            loss = softmax_cross_entropy(seg_logits, seg_logits)

        self.optim_wrapper.update_params(loss)

        # what should be careful is that the LoadAnnotation should be directly followed by the PackSegInputs
        # so that the gt_seg_map will not be process by any pipeline,
        # and we can directly use self.model.postprocess_result(seg_logits, data_samples) to up sample the logits
        # and only Resize, Padding and RandomFlip are used for augmentation,
        # the self.model.postprocess_result will process the logits autonomously for these augmentation
        self.model.postprocess_result(seg_logits.detach(), data_samples)

        evaluator.process(data_samples)

