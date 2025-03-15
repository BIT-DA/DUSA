import mmcv
import numpy
import torch
from mmengine.registry import RUNNERS

from mmseg.models.utils import resize
from .ttarunner import BaseTTARunner
from einops import rearrange


@RUNNERS.register_module()
class Source(BaseTTARunner):
    def tta_one_batch(self, batch_data, evaluator):
        with torch.no_grad():
            self.model.eval()
            pre_batch_data = self.model.data_preprocessor(batch_data, True)
            inputs, data_samples = pre_batch_data['inputs'], pre_batch_data['data_samples']
            seg_logits = self.model(inputs, data_samples, mode='tensor')
            # what should be careful is that the LoadAnnotation should be directly followed by the PackSegInputs
            # so that the gt_seg_map will not be process by any pipeline,
            # then we can directly use self.model.postprocess_result(seg_logits, data_samples) to up sample the logits
            # only Resize and RandomFlip are used for augmentation,
            # the self.model.postprocess_result will process the logits autonomously for these augmentation
            # we will process the ground truth according to the metainfor (size and flip) to compute loss by ourself
            # of cause more augmentation is ok, but modifications on self.get_label are needed.
            self.model.postprocess_result(seg_logits, data_samples)

        evaluator.process(data_samples)


@RUNNERS.register_module()
class MaskedSource(BaseTTARunner):
    def tta_one_batch(self, batch_data, evaluator):
        with torch.no_grad():
            self.model.eval()
            batch_data = self.model.data_preprocessor(batch_data, True)
            inputs, data_samples = batch_data['inputs'], batch_data['data_samples']
            seg_logits = self.model(inputs, data_samples, mode='tensor')
            seg_logits = resize(
                seg_logits,
                size=data_samples[0].img_shape,
                mode='bilinear',
                align_corners=self.model.align_corners,
                warning=False)

            probs = torch.softmax(seg_logits, dim=1)
            top_2 = torch.topk(probs, k=2, dim=1, largest=True, sorted=True).values
            confidence = top_2[:, 0, ...] - top_2[:, 1, ...]  # [b, h, w]

            flattened_confidence = rearrange(confidence, "b h w -> b (h w)")
            sorted_confidence, _ = torch.sort(flattened_confidence, dim=1, descending=False)
            index = int(0.0001 * len(sorted_confidence[0]))
            threshold = sorted_confidence[:, index, None]

            mask = confidence < threshold
            mask = mask[:, None, ...]
            mask = torch.broadcast_to(mask, inputs.shape)
            inputs[mask] = 0.0

            seg_logits = self.model(inputs, data_samples, mode='tensor')

            self.model.postprocess_result(seg_logits, data_samples)
        evaluator.process(data_samples)

