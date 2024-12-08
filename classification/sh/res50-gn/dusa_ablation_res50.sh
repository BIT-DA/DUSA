# ImageNet-C ResNet50 Ablations

# L1: + score prior loss(4, freezeDiff)
# update_auxiliary=False, no_lc=True, topk=4, rand_budget=0, multinomial=False
CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/ablation_study/bs64_res50_dit_topk_multinomial_replace_lc-train_k5_bs4x1_fp16_adam_test20_noise.py \
--work-dir output/res50-gn/noise/bs64_dit_topk_ablation_k5_adam_l1 \
--cfg-options update_auxiliary=False \
--cfg-options update_norm_only=False \
--cfg-options model.auxiliary_model.no_lc=True \
--cfg-options model.auxiliary_model.topk=4 \
--cfg-options model.auxiliary_model.rand_budget=0 \
--cfg-options model.auxiliary_model.multinomial=False

# L2: + logit norm(4, freezeDiff)
# update_auxiliary=False, no_lc=False, topk=4, rand_budget=0, multinomial=False
CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/ablation_study/bs64_res50_dit_topk_multinomial_replace_lc-train_k5_bs4x1_fp16_adam_test20_noise.py \
--work-dir output/res50-gn/noise/bs64_dit_topk_ablation_k5_adam_l2 \
--cfg-options update_auxiliary=False \
--cfg-options update_norm_only=False \
--cfg-options model.auxiliary_model.no_lc=False \
--cfg-options model.auxiliary_model.topk=4 \
--cfg-options model.auxiliary_model.rand_budget=0 \
--cfg-options model.auxiliary_model.multinomial=False

# L3: + adapt diffusion(4)
# update_auxiliary=True, no_lc=True, topk=4, rand_budget=0, multinomial=False
CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/ablation_study/bs64_res50_dit_topk_multinomial_replace_lc-train_k5_bs4x1_fp16_adam_test20_noise.py \
--work-dir output/res50-gn/noise/bs64_dit_topk_ablation_k5_adam_l3 \
--cfg-options update_auxiliary=True \
--cfg-options update_norm_only=False \
--cfg-options model.auxiliary_model.no_lc=True \
--cfg-options model.auxiliary_model.topk=4 \
--cfg-options model.auxiliary_model.rand_budget=0 \
--cfg-options model.auxiliary_model.multinomial=False

# L4: + logit norm(4)
# update_auxiliary=True, no_lc=False, topk=4, rand_budget=0, multinomial=False
CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/ablation_study/bs64_dit_topk_multinomial_replace_lc-train_k5_bs4x1_fp16_adam_test20_noise.py \
--work-dir output/convnext/noise/bs64_dit_topk_ablation_k5_adam_l4 \
--cfg-options update_auxiliary=True \
--cfg-options update_norm_only=False \
--cfg-options model.auxiliary_model.no_lc=False \
--cfg-options model.auxiliary_model.topk=4 \
--cfg-options model.auxiliary_model.rand_budget=0 \
--cfg-options model.auxiliary_model.multinomial=False

# L5: logit norm(6)
# update_auxiliary=True, no_lc=False, topk=6, rand_budget=0, multinomial=False
CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/ablation_study/bs64_dit_topk_multinomial_replace_lc-train_k5_bs4x1_fp16_adam_test20_noise.py \
--work-dir output/convnext/noise/bs64_dit_topk_ablation_k5_adam_l5 \
--cfg-options update_auxiliary=True \
--cfg-options update_norm_only=False \
--cfg-options model.auxiliary_model.no_lc=False \
--cfg-options model.auxiliary_model.topk=6 \
--cfg-options model.auxiliary_model.rand_budget=0 \
--cfg-options model.auxiliary_model.multinomial=False

# L6: + random budget(4+2)
# update_auxiliary=True, no_lc=False, topk=4, rand_budget=2, multinomial=False
CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/ablation_study/bs64_dit_topk_multinomial_replace_lc-train_k5_bs4x1_fp16_adam_test20_noise.py \
--work-dir output/convnext/noise/bs64_dit_topk_ablation_k5_adam_l6 \
--cfg-options update_auxiliary=True \
--cfg-options update_norm_only=False \
--cfg-options model.auxiliary_model.no_lc=False \
--cfg-options model.auxiliary_model.topk=4 \
--cfg-options model.auxiliary_model.rand_budget=2 \
--cfg-options model.auxiliary_model.multinomial=False

# L7: + multinomial(4+2)(Ours)
# update_auxiliary=True, no_lc=False, topk=4, rand_budget=2, multinomial=True
CUDA_VISIBLE_DEVICES=0 \
python tta.py tta_configs/imagenet-c/ablation_study/bs64_res50_dit_topk_multinomial_replace_lc-train_k5_bs4x1_fp16_adam_test20_noise.py \
--work-dir output/res50-gn/noise/bs64_dit_topk_ablation_k5_adam_l7 \
--cfg-options update_auxiliary=True \
--cfg-options update_norm_only=False \
--cfg-options model.auxiliary_model.no_lc=False \
--cfg-options model.auxiliary_model.topk=4 \
--cfg-options model.auxiliary_model.rand_budget=2 \
--cfg-options model.auxiliary_model.multinomial=True