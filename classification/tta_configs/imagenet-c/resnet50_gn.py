_base_ = ['../../configs/resnet/resnet50_8xb32_in1k.py']

randomness = dict(seed=2024)

visualizer = dict(_delete_=True, type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])

checkpoint = "pretrained_models/resnet50_gn_a1h2-8fe6c4d0.pth"

model = dict(
    _delete_=True,
    serial=True,
    type="WrappedModels",
    task_model=dict(
        type='TimmClassifier',
        model_name='resnet50_gn.a1h_in1k',
        # pretrained=True,
        checkpoint_path=checkpoint,
    )
)

tta_optimizer = dict(
    type='SGD',
    lr=2.5e-4,
    momentum=0.9
)

tta_optim_wrapper = dict(
    optimizer=tta_optimizer
)
