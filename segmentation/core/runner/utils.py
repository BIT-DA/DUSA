import torch
from typing import Sequence
from einops import rearrange


class PatchReplay:
    def __init__(self, patch_size=(256, 256), num_h=4, num_w=8, device=torch.device("cuda"), random_permutation=False):
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        if isinstance(patch_size, Sequence):
            assert isinstance(patch_size[0], int) and len(patch_size) == 2, "error type of patch size"

        self.patch_size = patch_size
        self.num_h = num_h
        self.num_w = num_w
        self.device = device
        self.random_permutation = random_permutation

        self._original_data = torch.ones([num_h * num_w, 3, *patch_size]).to(device) * 125
        self._original_gt = (torch.ones([num_h * num_w, *patch_size]) * 255).long().to(device)

        self._curren_idx = 0

    @property
    def original_data(self):
        return self._original_data

    @property
    def original_gt(self):
        return self._original_gt

    @torch.no_grad()
    def add_patch(self, patch, gt):
        # patch: 3, h, w, gt: h, w
        # check the shape !!!!!
        assert patch.shape[0] == 3, f"expect rgb channel, but got {patch.shape[0]}"
        assert len(self.patch_size) == len(patch.shape) - 1 == len(gt.shape), \
            f"error shape of patch, expect shape of {self.patch_size} \n" \
            f"but got patch shape {patch.shape[1:]} \n" \
            f"and gt shape {gt.shape} \n"
        for s1, s2, s3 in zip(self.patch_size, patch.shape[1:], gt.shape):
            assert s1 == s2 == s3, \
                f"error shape of patch, expect shape of {self.patch_size} \n" \
                f"but got patch shape {patch.shape[1:]} \n" \
                f"and gt shape {gt.shape} \n"
        patch = patch.to(self.device)
        gt = gt.to(self.device)

        self._original_data[self._curren_idx] = patch
        self._original_gt[self._curren_idx] = gt

        self._curren_idx += 1
        self._curren_idx %= self.num_h * self.num_w

    @torch.no_grad()
    def reconstruct(self):
        if self.random_permutation:
            indices = torch.randperm(self.num_h * self.num_w)
        else:
            indices = torch.arange(self.num_h * self.num_w)

        patchs = self.original_data[indices]
        gts = self.original_gt[indices]

        img_inputs = rearrange(patchs, "(nh nw) c h w -> c (nh h) (nw w)", nh=self.num_h)
        ground_truth = rearrange(gts, "(nh nw) h w -> (nh h) (nw w)", nh=self.num_h)

        return img_inputs, ground_truth

