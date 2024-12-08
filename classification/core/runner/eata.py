import torch
import torchvision.transforms as transforms
import os
import random
import math
import torchvision.datasets as datasets
from copy import deepcopy
from torch.nn.modules.batchnorm import _BatchNorm, _NormBase
from torch.nn.modules import LayerNorm, GroupNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from mmengine.registry import RUNNERS
from ..model.wrapped_models import WrappedModels
from .ttarunner import BaseTTARunner



# The following ImageFolder supports sample a subset from the entire dataset by index/classes/sample number, at any time after the dataloader created.
class SelectedRotateImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform):
        super(SelectedRotateImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img_input = self.loader(path)
        if self.transform is not None:
            # print("self.transform={}".format(self.transform))
            img = self.transform(img_input)
        else:
            img = img_input
        results = []
        results.append(img)
        results.append(target)
        return results

    def set_dataset_size(self, subset_size):
        num_train = len(self.targets)
        indices = list(range(num_train))
        random.shuffle(indices)
        self.samples = [self.samples[i] for i in indices[:subset_size]]
        self.targets = [self.targets[i] for i in indices[:subset_size]]
        return len(self.targets)



def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x




@RUNNERS.register_module()
class EataCls(BaseTTARunner):
    model: WrappedModels

    def __init__(self, cfg):
        super(EataCls, self).__init__(cfg)
        self.episodic = False
        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples

        self.e_margin = math.log(1000) * 0.40  # hyper-parameter E_0
        self.d_margin = 0.05  # hyper-parameter \epsilon for consine simlarity thresholding
        self.current_model_probs = None  # the moving average of probability vector

        self.fishers = self.compute_fisher_informatrix(fisher_size=2000)  # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        self.fisher_alpha = 2000.  # trade-off \beta for two losses


    def config_tta_model(self):
        print("config_tta_model")
        """Configure model for use with eata."""
        # train mode, because eata optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what eata updates
        self.model.requires_grad_(False)
        # configure norm for eata updates: enable grad + force batch statisics
        for name, sub_module in self.model.named_modules():
            if "norm" in name.lower() or isinstance(sub_module, (_NormBase, _BatchNorm, _InstanceNorm, LayerNorm, GroupNorm)):
                sub_module.requires_grad_(True)
                if hasattr(sub_module, "track_running_stats") \
                        and hasattr(sub_module, "running_mean") \
                        and hasattr(sub_module, "running_var"):   # force use of batch stats in train and eval modes
                    sub_module.track_running_stats = False
                    sub_module.running_mean = None
                    sub_module.running_var = None

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if "norm" in nm.lower() or isinstance(m, (_NormBase, _BatchNorm, _InstanceNorm, LayerNorm, GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def reset_model(self):
        if self.model_state_dict is None or self.optim_state_dict is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.model.load_state_dict(self.model_state_dict, strict=True)
        self.optim_wrapper.load_state_dict(self.optim_state_dict)
        self.current_model_probs = None
        self.num_samples_update_1 = 0
        self.num_samples_update_2 = 0

    def reset_model_probs(self, probs):
        self.current_model_probs = probs

    def compute_fisher_informatrix(self, fisher_size):
        print("compute_fisher_informatrix")
        te_transforms_local = transforms.Compose([
            # transforms.Resize(256),  # follow EATA te_transforms_imageC
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=list(self.model.task_model.data_preprocessor.mean.view(-1).cpu().numpy() / 255.),
                                 std=list(self.model.task_model.data_preprocessor.std.view(-1).cpu().numpy() / 255.)),
        ])
        validdir = os.path.join(self.cfg.imagenet_data_root, 'val')
        fisher_dataset = SelectedRotateImageFolder(validdir, te_transforms_local)
        fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=self.cfg.tta_data_loader.batch_size, shuffle=True,
                                               num_workers=self.cfg.tta_data_loader.num_workers, pin_memory=True)
        fisher_dataset.set_dataset_size(fisher_size)
        params, param_names = self.collect_params()
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = torch.nn.CrossEntropyLoss().cuda()
        for iter_, (images, targets) in enumerate(fisher_loader, start=1):
            images = images.cuda()
            outputs = self.model.task_model(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        self.logger.info("compute fisher matrices finished")
        del ewc_optimizer
        print("copying model_state_dict")
        self.model_state_dict = deepcopy(self.model.state_dict())
        return fishers


    def tta_one_batch(self, batch_data):
        self.model.eval()
        all_loss = dict()
        with self.optim_wrapper.optim_context(self.model):
            if self.episodic:
                self.reset_model()
            task_batch_data = self.model.task_model.data_preprocessor(batch_data)
            inputs, data_samples = task_batch_data["inputs"], task_batch_data["data_samples"]
            # forward
            outputs = self.model.task_model(inputs)
            # adapt
            entropys = softmax_entropy(outputs)
            # filter unreliable samples
            filter_ids_1 = torch.where(entropys < self.e_margin)
            ids1 = filter_ids_1
            ids2 = torch.where(ids1[0] > -0.1)
            entropys = entropys[filter_ids_1]
            # filter redundant samples
            if self.current_model_probs is not None:
                cosine_similarities = torch.nn.functional.cosine_similarity(self.current_model_probs.unsqueeze(dim=0),
                                                              outputs[filter_ids_1].softmax(1), dim=1)
                filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
                entropys = entropys[filter_ids_2]
                ids2 = filter_ids_2
                updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1][filter_ids_2].softmax(1))
            else:
                updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))
            coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))

            # implementation version 1, compute loss, all samples backward (some unselected are masked)
            entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
            loss = entropys.mean(0)
            if self.fishers is not None:
                ewc_loss = 0
                for name, param in self.model.named_parameters():
                    if name in self.fishers:
                        ewc_loss += self.fisher_alpha * (self.fishers[name][0] * (param - self.fishers[name][1]) ** 2).sum()
                loss += ewc_loss
            if inputs[ids1][ids2].size(0) != 0:
                self.optim_wrapper.update_params(loss)
            all_loss["loss"] = loss.item()
            self.set_cls_predictions(outputs, data_samples)

            num_counts_2, num_counts_1, updated_probs = entropys.size(0), filter_ids_1[0].size(0), updated_probs
            self.num_samples_update_2 += num_counts_2
            self.num_samples_update_1 += num_counts_1
            self.reset_model_probs(updated_probs)
        return data_samples, all_loss





def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)

