# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS


@MODELS.register_module()
class CAEHead(BaseModule):
    """Head for CAE Pre-training.

    Compute the align loss and the fully loss. In addition, this head also
    generates the prediction target generated by dalle.

    Args:
        loss (dict): The config of loss.
        tokenizer_path (str): The path of the tokenizer.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 loss: dict,
                 init_cfg: Optional[Union[dict, List[dict]]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.loss_module = MODELS.build(loss)

    @torch.no_grad()
    def _generate_target(self, logits_target: torch.Tensor) -> torch.Tensor:
        """Generate the reconstruction target.

        Args:
            logits_target (torch.Tensor): The logits generated by DALL-E.s

        Returns:
            torch.Tensor: The logits target.
        """
        target = torch.argmax(logits_target, dim=1)
        return target.flatten(1)

    def loss(self, logits: torch.Tensor, logits_target: torch.Tensor,
             latent_pred: torch.Tensor, latent_target: torch.Tensor,
             mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate loss.

        Args:
            logits (torch.Tensor): Logits generated by decoder.
            logits_target (img_target): Target generated by dalle for decoder
                prediction.
            latent_pred (torch.Tensor): Latent prediction by regressor.
            latent_target (torch.Tensor): Target for latent prediction,
                generated by teacher.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The tuple of loss.
                - ``loss_main`` (torch.Tensor): Cross entropy loss.
                - ``loss_align`` (torch.Tensor): MSE loss.
        """

        target = self._generate_target(logits_target)  # target features
        target = target[mask].detach()

        # loss fully for decoder, loss align for regressor
        loss_main, loss_align = self.loss_module(logits, target, latent_pred,
                                                 latent_target)

        return (loss_main, loss_align)
