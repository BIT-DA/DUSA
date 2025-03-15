tta_optimizer = dict(type='Adam', lr=0.00006/8, betas=(0.9, 0.999))


tta_optim_wrapper = dict(type='OptimWrapper', optimizer=tta_optimizer)

continual = True

tta_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

tta_data_loader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
)

tta_dataset_type = "ADE20KDataset"
tta_data_root = "data"
seg_map_path = "ADEChallengeData2016/annotations/validation"


_tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

tasks = [
    dict(type=tta_dataset_type,
         data_root=tta_data_root,
         data_prefix=dict(
             img_path='ADE20K_val-c/gaussian_noise/5/validation', seg_map_path=seg_map_path),
         pipeline=_tta_pipeline),
    dict(type=tta_dataset_type,
         data_root=tta_data_root,
         data_prefix=dict(
             img_path='ADE20K_val-c/shot_noise/5/validation', seg_map_path=seg_map_path),
         pipeline=_tta_pipeline),
    dict(type=tta_dataset_type,
         data_root=tta_data_root,
         data_prefix=dict(
             img_path='ADE20K_val-c/impulse_noise/5/validation', seg_map_path=seg_map_path),
         pipeline=_tta_pipeline),
    dict(type=tta_dataset_type,
         data_root=tta_data_root,
         data_prefix=dict(
             img_path='ADE20K_val-c/defocus_blur/5/validation', seg_map_path=seg_map_path),
         pipeline=_tta_pipeline),
    dict(type=tta_dataset_type,
         data_root=tta_data_root,
         data_prefix=dict(
             img_path='ADE20K_val-c/glass_blur/5/validation', seg_map_path=seg_map_path),
         pipeline=_tta_pipeline),
    dict(type=tta_dataset_type,
         data_root=tta_data_root,
         data_prefix=dict(
             img_path='ADE20K_val-c/motion_blur/5/validation', seg_map_path=seg_map_path),
         pipeline=_tta_pipeline),
    dict(type=tta_dataset_type,
         data_root=tta_data_root,
         data_prefix=dict(
             img_path='ADE20K_val-c/zoom_blur/5/validation', seg_map_path=seg_map_path),
         pipeline=_tta_pipeline),
    dict(type=tta_dataset_type,
         data_root=tta_data_root,
         data_prefix=dict(
             img_path='ADE20K_val-c/snow/5/validation', seg_map_path=seg_map_path),
         pipeline=_tta_pipeline),
    dict(type=tta_dataset_type,
         data_root=tta_data_root,
         data_prefix=dict(
             img_path='ADE20K_val-c/frost/5/validation', seg_map_path=seg_map_path),
         pipeline=_tta_pipeline),
    dict(type=tta_dataset_type,
             data_root=tta_data_root,
             data_prefix=dict(
                 img_path='ADE20K_val-c/fog/5/validation', seg_map_path=seg_map_path),
             pipeline=_tta_pipeline),
    dict(type=tta_dataset_type,
         data_root=tta_data_root,
         data_prefix=dict(
             img_path='ADE20K_val-c/brightness/5/validation', seg_map_path=seg_map_path),
         pipeline=_tta_pipeline),
    dict(type=tta_dataset_type,
             data_root=tta_data_root,
             data_prefix=dict(
                 img_path='ADE20K_val-c/contrast/5/validation', seg_map_path=seg_map_path),
             pipeline=_tta_pipeline),
    dict(type=tta_dataset_type,
         data_root=tta_data_root,
         data_prefix=dict(
             img_path='ADE20K_val-c/elastic_transform/5/validation', seg_map_path=seg_map_path),
         pipeline=_tta_pipeline),
    dict(type=tta_dataset_type,
         data_root=tta_data_root,
         data_prefix=dict(
             img_path='ADE20K_val-c/pixelate/5/validation', seg_map_path=seg_map_path),
         pipeline=_tta_pipeline),
    dict(type=tta_dataset_type,
         data_root=tta_data_root,
         data_prefix=dict(
             img_path='ADE20K_val-c/jpeg_compression/5/validation', seg_map_path=seg_map_path),
         pipeline=_tta_pipeline),
]
