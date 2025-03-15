from mmengine.registry import RUNNERS

from torch.nn.modules.batchnorm import _NormBase
import torch
from copy import deepcopy

from mmseg.models.utils import resize
from .ttarunner import BaseTTARunner, softmax_cross_entropy


@RUNNERS.register_module()
class CoTTA(BaseTTARunner):
    def __init__(self, cfg):
        super(CoTTA, self).__init__(cfg)
        self.anchor = self.build_ema_model(self.model)
        self.ema_model = self.build_ema_model(self.model)

    def config_tta_model(self):
        self.model.requires_grad_(True)

    def tta_one_batch(self, batch_data, evaluator):
        self.model.eval()
        self.anchor.eval()
        self.ema_model.eval()
        # actually only all_data_samples[4] is useful
        all_inputs, all_data_samples = batch_data["inputs"], batch_data["data_samples"]

        assert len(all_inputs) == 12, \
            "please use the FixedMultiScaleFlipAug with scales [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] and allow_flip = True"
        # all_inputs[4] will be the default input
        # get anchor output
        default_input, default_data_samples = all_inputs[4], all_data_samples[4]
        assert hasattr(default_data_samples[0], "img_shape"), \
            "some thing error with data_samples, expected img_shape attribute"
        expected_shape = default_data_samples[0].img_shape

        with torch.no_grad():
            tmp_data = dict(inputs=default_input, data_samples=default_data_samples)
            tmp_data = self.anchor.data_preprocessor(tmp_data, True)
            inputs, data_samples = tmp_data['inputs'], tmp_data['data_samples']
            anchor_seg_logits = self.anchor(inputs, data_samples, mode='tensor')
            anchor_seg_logits = resize(
                        anchor_seg_logits,
                        size=expected_shape,
                        mode='bilinear',
                        align_corners=self.anchor.align_corners,
                        warning=False)

            anchor_confidence = torch.max(torch.softmax(anchor_seg_logits, dim=1), dim=1, keepdim=True).values

            all_seg_logits = []
            for inputs, data_samples in zip(all_inputs, all_data_samples):
                tmp_data = dict(inputs=inputs, data_samples=data_samples)

                tmp_data = self.ema_model.data_preprocessor(tmp_data, True)
                inputs, data_samples = tmp_data['inputs'], tmp_data['data_samples']
                tmp_seg_logits = self.ema_model(inputs, data_samples, mode='tensor')
                flip = data_samples[0].get('flip', None)
                if flip:
                    flip_direction = data_samples[0].get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        tmp_seg_logits = tmp_seg_logits.flip(dims=(3,))
                    else:
                        tmp_seg_logits = tmp_seg_logits.flip(dims=(2,))

                tmp_seg_logits = resize(
                            tmp_seg_logits,
                            size=expected_shape,
                            mode='bilinear',
                            align_corners=self.ema_model.align_corners,
                            warning=False)
                all_seg_logits.append(tmp_seg_logits)

            all_seg_logits = torch.stack(all_seg_logits)  # (12, B, C, H, W)
            ema_seg_logits = torch.mean(all_seg_logits, dim=0, keepdim=False)  # (B, C, H, W)

            default_ema_seg_logits = all_seg_logits[4]

            mask = (anchor_confidence > 0.69).float()
            mask = torch.broadcast_to(mask, ema_seg_logits.shape)
            augmented_average_logits = mask * default_ema_seg_logits + (1 - mask) * ema_seg_logits

        tmp_data = dict(inputs=default_input, data_samples=default_data_samples)
        with self.optim_wrapper.optim_context(self.model):
            tmp_data = self.model.data_preprocessor(tmp_data, True)
            inputs, data_samples = tmp_data['inputs'], tmp_data['data_samples']
            model_seg_logits = self.model(inputs, data_samples, mode='tensor')
            model_seg_logits = resize(
                model_seg_logits,
                size=expected_shape,
                mode='bilinear',
                align_corners=self.model.align_corners,
                warning=False)

            loss = softmax_cross_entropy(model_seg_logits, augmented_average_logits)

        self.optim_wrapper.update_params(loss)

        self.model.postprocess_result(augmented_average_logits, data_samples)

        evaluator.process(data_samples)

        # Teacher update
        self.ema_model = self.update_ema_variables(ema_model=self.ema_model, model=self.model, alpha_teacher=0.999)
        # Stochastic restore
        if True:
            for nm, m in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape) < 0.01).float().cuda()
                        with torch.no_grad():
                            p.data = self.state_dict[f"{nm}.{npp}"].cuda() * mask + p * (1. - mask)

    def reset_model(self):
        super(CoTTA, self).reset_model()
        self.ema_model = self.build_ema_model(self.model)