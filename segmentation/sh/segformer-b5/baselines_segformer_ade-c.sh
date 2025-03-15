CUDA_VISIBLE_DEVICES=0 \
python tta_seg.py tta_configs/ade/source.py \
--cfg-options continual=False \
--cfg-options work_dir=output/segformer/ade-c/source

CUDA_VISIBLE_DEVICES=0 \
python tta_seg.py tta_configs/ade/ttnorm.py \
--cfg-options continual=False \
--cfg-options work_dir=output/segformer/ade-c/ttnorm

CUDA_VISIBLE_DEVICES=0 \
python tta_seg.py tta_configs/ade/tent.py \
--cfg-options continual=False \
--cfg-options work_dir=output/segformer/ade-c/tent

CUDA_VISIBLE_DEVICES=0 \
python tta_seg.py tta_configs/ade/cotta.py \
--cfg-options continual=False \
--cfg-options work_dir=output/segformer/ade-c/cotta
