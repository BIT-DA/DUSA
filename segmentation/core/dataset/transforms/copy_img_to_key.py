from typing import Dict, Optional, Union, Tuple, List

from mmseg.registry import TRANSFORMS
from copy import deepcopy
from mmcv.transforms.base import BaseTransform


@TRANSFORMS.register_module()
class CopyImageToKey(BaseTransform):
    def __init__(self, key_name="img_copy"):
        self.key_name = key_name

    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        if "img" in results:
            results.setdefault(self.key_name, deepcopy(results["img"]))

        return results

