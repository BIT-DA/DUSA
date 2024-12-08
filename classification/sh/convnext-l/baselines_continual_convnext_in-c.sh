# in-c-continual
CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/convnext_large_continual/source.py \
--work-dir output/convnext/in-c-continual/source

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/convnext_large_continual/tent.py \
--work-dir output/convnext/in-c-continual/tent

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/convnext_large_continual/sar.py \
--work-dir output/convnext/in-c-continual/sar

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/convnext_large_continual/eata.py \
--work-dir output/convnext/in-c-continual/eata

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/convnext_large_continual/cotta.py \
--work-dir output/convnext/in-c-continual/cotta

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/convnext_large_continual/rotta.py \
--work-dir output/convnext/in-c-continual/rotta