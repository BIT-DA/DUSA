_base_ = ["../vit_b_16_timm_in1k.py", "../imagenet-c-bs64.py"]
runner_type = "EataCls"
update_auxiliary = False

model = dict(
    auxiliary_model=None
)
imagenet_data_root = "data/ImageNet"