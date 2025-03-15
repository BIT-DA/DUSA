from typing import Dict, Optional, Union, Tuple, List

from mmseg.registry import TRANSFORMS
from copy import deepcopy
from mmcv.transforms.base import BaseTransform
import mmengine
import warnings
from mmcv.transforms.wrappers import Compose


@TRANSFORMS.register_module()
class FixedMultiScaleFlipAug(BaseTransform):

    def __init__(
        self,
        transforms: List[dict],
        scales: Optional[Union[Tuple, List[Tuple]]] = None,
        scale_factor: Optional[Union[float, List[float]]] = None,
        allow_flip: bool = False,
        flip_direction: Union[str, List[str]] = 'horizontal',
        resize_cfg: dict = dict(type='Resize', keep_ratio=True),
        flip_cfg: dict = dict(type='RandomFlip')
    ) -> None:
        super().__init__()
        self.transforms = Compose(transforms)  # type: ignore

        if scales is not None:
            self.scales = scales if isinstance(scales, list) else [scales]
            self.scale_key = 'scale'
            assert mmengine.is_list_of(self.scales, tuple)
        else:
            # if ``scales`` and ``scale_factor`` both be ``None``
            if scale_factor is None:
                self.scales = [1.]  # type: ignore
            elif isinstance(scale_factor, list):
                self.scales = scale_factor  # type: ignore
            else:
                self.scales = [scale_factor]  # type: ignore

            self.scale_key = 'scale_factor'

        self.allow_flip = allow_flip
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmengine.is_list_of(self.flip_direction, str)
        if not self.allow_flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        self.resize_cfg = resize_cfg.copy()
        self.flip_cfg = flip_cfg

    def transform(self, results: dict) -> Dict:
        data_samples = []
        inputs = []
        flip_args = [(False, '')]
        if self.allow_flip:
            flip_args += [(True, direction)
                          for direction in self.flip_direction]
        for scale in self.scales:
            for flip, direction in flip_args:
                _resize_cfg = self.resize_cfg.copy()
                _resize_cfg.update({self.scale_key: scale})
                _resize_flip = [_resize_cfg]

                if flip:
                    _flip_cfg = self.flip_cfg.copy()
                    _flip_cfg.update(prob=1.0, direction=direction)
                    _resize_flip.append(_flip_cfg)
                else:
                    results['flip'] = False
                    results['flip_direction'] = None

                resize_flip = Compose(_resize_flip)
                _results = resize_flip(results.copy())
                packed_results = self.transforms(_results)  # type: ignore

                inputs.append(packed_results['inputs'])  # type: ignore
                data_samples.append(
                    packed_results['data_samples'])  # type: ignore
        return dict(inputs=inputs, data_samples=data_samples)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}'
        repr_str += f', scales={self.scales}'
        repr_str += f', allow_flip={self.allow_flip}'
        repr_str += f', flip_direction={self.flip_direction})'
        return repr_str
