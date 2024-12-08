# IN-C
CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/vit-b-16-timm-in1k/bs64_dit_topk_multinomial_replace_lc-train_k5_bs4x1_fp16_adam_test20_15task.py \
--work-dir output/vit-timm-in1k/in-c/bs64_dit_topk_multinomial_replace_lc_k5_adam \
--cfg-options update_auxiliary=True \
--cfg-options update_norm_only=False