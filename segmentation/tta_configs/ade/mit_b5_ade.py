_base_ = ['../../configs/segformer/segformer_mit-b0_8xb2-160k_ade20k-512x512.py']

checkpoint = 'pretrained_models/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth'  # noqa

data_preprocessor = dict(size=None, size_divisor=1.0)
model = dict(
    type='WrappedEncoderDecoder',
    data_preprocessor=data_preprocessor,
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    backbone=dict(
        init_cfg=None,
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(type="WrappedSegformerHead",
                     in_channels=[64, 128, 320, 512]))

randomness = dict(seed=2024)
