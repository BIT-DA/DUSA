from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.models.decode_heads.segformer_head import SegformerHead
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
import torch
import torch.nn.functional as F


@MODELS.register_module()
class WrappedEncoderDecoder(EncoderDecoder):
    def forward(self,
                inputs,
                data_samples=None,
                mode: str = 'tensor', ret_fea=False, mc_dropout=0):

        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples, ret_fea=ret_fea, mc_dropout=mc_dropout)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def _forward(self,
                 inputs,
                 data_samples=None,
                 ret_fea=False, mc_dropout=0):
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x, ret_fea=ret_fea, mc_dropout=mc_dropout)


@MODELS.register_module()
class WrappedSegformerHead(SegformerHead):
    def forward(self, inputs, ret_fea=False, mc_dropout=0):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        outs = torch.cat(outs, dim=1)

        fea = self.fusion_conv(outs)
        out = self.cls_seg(fea)

        if ret_fea:
            return out, fea
        else:
            return out








