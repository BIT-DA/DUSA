# IN-C
CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/convnext_large/source.py \
--work-dir output/convnext/in-c/source

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/convnext_large/tent.py \
--work-dir output/convnext/in-c/tent

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/convnext_large/sar.py \
--work-dir output/convnext/in-c/sar

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/convnext_large/eata.py \
--work-dir output/convnext/in-c/eata

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/convnext_large/cotta.py \
--work-dir output/convnext/in-c/cotta

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/convnext_large/rotta.py \
--work-dir output/convnext/in-c/rotta