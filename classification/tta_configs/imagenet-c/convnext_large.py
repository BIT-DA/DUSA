_base_ = ["../../configs/convnext/convnext-large_64xb64_in1k.py"]

randomness = dict(seed=2024)

visualizer = dict(_delete_=True, type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

checkpoint = "pretrained_models/convnext-large_3rdparty_64xb64_in1k_20220124-f8a0ded0.pth"

model = dict(
    _delete_=True,
    serial=True,
    type="WrappedModels",
    task_model=dict(
        type='ImageClassifier',
        backbone=dict(type='ConvNeXt', arch='large', drop_path_rate=0.5),
        head=dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=1536,
            loss=dict(
                type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
            init_cfg=None,
        ),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
    )
)

# follow diff_tta
tta_optimizer = dict(
    type='Adam',
    lr=1e-5,
    betas=(0.9, 0.999)
)

tta_optim_wrapper = dict(
    optimizer=tta_optimizer
)
