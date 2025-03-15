_base_ = ['../../configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py']

checkpoint = 'pretrained_models/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth'  # noqa

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=None,
    size_divisor=1.0
)

model = dict(
    _delete_=True,
    serial=True,
    type="WrappedModels",
    task_model=dict(
        type='WrappedEncoderDecoder',
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=data_preprocessor,
        init_cfg=dict(
            checkpoint=checkpoint,
            type='Pretrained'),
        backbone=dict(
            type='MixVisionTransformer',
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=[3, 6, 40, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            init_cfg=None),
        decode_head=dict(
            type='WrappedSegformerHead',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            channels=256,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        pretrained=None,
    )
)
randomness = dict(seed=2024)
