# IN-C
CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/convnext_large_continual/bs64_dit_topk_multinomial_replace_lc-train_k5_bs4x1_fp16_adam_test20_15task.py \
--work-dir output/convnext/in-c-continual/bs64_dit_topk_multinomial_replace_lc_k5_adam \
--cfg-options update_auxiliary=True \
--cfg-options update_norm_only=False