# IN-C
CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/vit-b-16-timm-in1k/source.py \
--work-dir output/vit-timm-in1k/in-c/source

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/vit-b-16-timm-in1k/tent.py \
--work-dir output/vit-timm-in1k/in-c/tent

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/vit-b-16-timm-in1k/sar.py \
--work-dir output/vit-timm-in1k/in-c/sar

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/vit-b-16-timm-in1k/eata.py \
--work-dir output/vit-timm-in1k/in-c/eata

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/vit-b-16-timm-in1k/cotta.py \
--work-dir output/vit-timm-in1k/in-c/cotta

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/vit-b-16-timm-in1k/rotta.py \
--work-dir output/vit-timm-in1k/in-c/rotta