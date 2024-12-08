_base_ = ["../resnet50_gn.py", "../imagenet-c-bs64.py"]
runner_type = "TentCls"
update_auxiliary = False

model = dict(
    auxiliary_model=None
)
