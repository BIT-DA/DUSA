_base_ = ["../../configs/vision_transformer/vit-base-p16_64xb64_in1k.py"]

randomness = dict(seed=2024)

visualizer = dict(_delete_=True, type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

data_preprocessor = dict(
    num_classes=1000,
    # RGB format normalization parameters
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    # convert image from BGR to RGB
    to_rgb=True,
)

model = dict(
    _delete_=True,
    serial=True,
    type="WrappedModels",
    task_model=dict(
        type='TimmClassifier',
        model_name='vit_base_patch16_224.augreg_in1k',
        # pretrained=True,
        checkpoint_path='pretrained_models/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
    )
)

tta_optimizer = dict(
    type='SGD',
    lr=1e-3,
    momentum=0.9
)

tta_optim_wrapper = dict(
    optimizer=tta_optimizer
)


