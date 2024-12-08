continual = False
info_functions = "ClsInfoFunctions"

tta_evaluator = dict(type='Accuracy', topk=(1, ))

tta_data_loader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
)

tta_dataset_type = "ImageNet"
tta_data_root = "data/imagenet-c"

# follow SAR
_tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

tasks = [
    dict(type=tta_dataset_type,
         data_root=tta_data_root,
         data_prefix=dict(
             img_path='gaussian_noise/5'),
         pipeline=_tta_pipeline),
    dict(type=tta_dataset_type,
         data_root=tta_data_root,
         data_prefix=dict(
             img_path='shot_noise/5'),
         pipeline=_tta_pipeline),
    dict(type=tta_dataset_type,
         data_root=tta_data_root,
         data_prefix=dict(
             img_path='impulse_noise/5'),
         pipeline=_tta_pipeline)
]
