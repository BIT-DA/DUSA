CUDA_VISIBLE_DEVICES=0,1 \
python tta_seg.py tta_configs/ade/sdtta/sd_controlnet_slide_topk.py \
--cfg-options continual=False \
--cfg-options work_dir=output/segformer/ade-c/dusa-seg \
--cfg-options update_auxiliary=True \
--cfg-options update_norm_only=False \
--cfg-options model.auxiliary_model.topk=5 \
--cfg-options model.auxiliary_model.rand_budget=0 \
--cfg-options model.auxiliary_model.classes_threshold=20 \
--cfg-options logits_mode='normed_logits'