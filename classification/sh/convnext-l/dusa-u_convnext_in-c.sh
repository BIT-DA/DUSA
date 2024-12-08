# DUSA-U
# Fully TTA on ImageNet-C with ConvNeXt
CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/convnext_large/bs64_dit_topk_uncond_multinomial_replace_lc-train_k5_bs4x1_fp16_adam_test20_15task.py \
--work-dir output/convnext/in-c/bs64_dit_topk_uncond_multinomial_replace_lc_k5_adam \
--cfg-options update_auxiliary=True \
--cfg-options update_norm_only=False