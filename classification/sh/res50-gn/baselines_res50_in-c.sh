# IN-C
CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/resnet50-gn/source.py \
--work-dir output/res50-gn/in-c/source

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/resnet50-gn/tent.py \
--work-dir output/res50-gn/in-c/tent

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/resnet50-gn/sar.py \
--work-dir output/res50-gn/in-c/sar

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/resnet50-gn/eata.py \
--work-dir output/res50-gn/in-c/eata

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/resnet50-gn/cotta.py \
--work-dir output/res50-gn/in-c/cotta

CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/resnet50-gn/rotta.py \
--work-dir output/res50-gn/in-c/rotta