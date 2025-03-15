_base_ = [ "../mit_b5_aux_ade.py","../ade20k_base.py"]
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(_delete_=True, type='SegLocalVisualizer', vis_backends=vis_backends)
runner_type = "TextImageAuxiliaryTTAClsLogitsFP16"
update_auxiliary = True
model = dict(
    type="WrappedModels",
    auxiliary_model=dict(
        type="StableDiffusionControlnetSegSlideUniqueTopK",
        auxiliary_slide=dict(crop_size=(512, 512), stride=(0, 171)),
        # training timestep range [left, right)
        timestep_range=(100, 101),
        class_names="ADE_CATEGORIES",
        controlnet_ckpt="lllyasviel/control_v11p_sd15_seg",
        # model_path="runwayml/stable-diffusion-v1-5",
        model_path="KiwiXR/stable-diffusion-v1-5",
        preprocessor=dict(input_size=(512, 512), map2negone=True),
        unet=dict(device="cuda:1"),
    )
)
tta_data_loader = dict(
    batch_size=1,
)
tta_optimizer = dict(type='AdamW', betas=(0.9, 0.999), weight_decay=0.0)
tta_optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=tta_optimizer
)
activation_checkpointing=['task_model.backbone']