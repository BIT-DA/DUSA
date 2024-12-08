_base_ = ["../convnext_large.py", "../imagenet-c-bs64.py"]
runner_type = "TextImageAuxiliaryTTAClsNormedLogitsWithLogitsFP16"
update_auxiliary = True
tta_evaluator = dict(type='Accuracy', topk=(1, 5, 20))
info_functions = "TopKClsInfoFunctions"
model = dict(
    auxiliary_model=dict(
        type="DiTTopKMultinomialReplaceOriFP16",
        # training timestep range [left, right)
        timestep_range=(100, 101),
        # only topk+rand logits are used for training
        topk=4,
        rand_budget=2,
        diffusion=dict(type="DiT_XL_2", image_size=256),
        vae=dict(pretrain="stabilityai/sd-vae-ft-ema"),
        preprocessor=dict(input_size=256, map2negone=True)
    )
)
model_wrapper_cfg=dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)
tta_data_loader = dict(
    batch_size=8,
)
# follow diff_tta
tta_optimizer = dict(
    _delete_=True,
    type='Adam',
    lr=1e-5,
    betas=(0.9, 0.999)
)

tta_optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=tta_optimizer,
    accumulative_counts=8,
)