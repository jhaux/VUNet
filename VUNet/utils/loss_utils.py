import torch
from collections import namedtuple
from torchvision.models import vgg19
from torchvision import transforms
from typing import Tuple


def update_loss_weights_inplace(loss_config, step):
    # update kl weights
    for weight_dict in loss_config.values():
        if "start_ramp_it" in weight_dict:
            if step < weight_dict["start_ramp_it"]:
                weight_dict["weight"] = weight_dict["start_ramp_val"]
            elif step > weight_dict["end_ramp_it"]:
                weight_dict["weight"] = weight_dict["end_ramp_val"]
            else:
                ramp_progress = (step - weight_dict["start_ramp_it"]) / (
                    weight_dict["end_ramp_it"] - weight_dict["start_ramp_it"]
                )
                ramp_diff = weight_dict["end_ramp_val"] - weight_dict["start_ramp_val"]
                weight_dict["weight"] = (
                    ramp_progress * ramp_diff + weight_dict["start_ramp_val"]
                )


class MSELossInstances(torch.nn.MSELoss):
    """MSELoss, which reduces to instances of the batch
    """

    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        super_result = super().forward(image, target)
        reduced = super_result.mean(axis=(1, 2, 3))
        return reduced


class L1LossInstances(torch.nn.L1Loss):
    """L1Loss, which reduces to instances of the batch
    """

    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        super_result = super().forward(image, target)
        reduced = super_result.mean(axis=(1, 2, 3))
        return reduced


def latent_kl(prior_mean, posterior_mean):
    """
    :param prior_mean:
    :param posterior_mean:
    :return:
    """
    kl = 0.5 * torch.pow(prior_mean - posterior_mean, 2)
    kl = torch.sum(kl, dim=[1, 2, 3])
    # kl = torch.mean(kl)

    return kl


def aggregate_kl_loss(prior_means, posterior_means):
    kl_loss = torch.sum(
        torch.cat(
            [
                latent_kl(p, q).unsqueeze(dim=-1)
                for p, q in zip(
                    list(prior_means.values()), list(posterior_means.values())
                )
            ],
            dim=-1,
        ),
        dim=-1,
    )
    return kl_loss


def scale_img(x):
    """
    Scale in between 0 and 1
    :param x:
    :return:
    """
    # ma = torch.max(x)
    # mi = torch.min(x)
    out = (x + 1.0) / 2.0
    out = torch.clamp(out, 0.0, 1.0)
    return out


VGGOutput = namedtuple(
    "VGGOutput", ["input", "relu1_2", "relu2_2", "relu3_2", "relu4_2", "relu5_2"],
)
VGGTargetLayers = {
    "3": "relu1_2",
    "8": "relu2_2",
    "13": "relu3_2",
    "22": "relu4_2",
    "31": "relu5_2",
}


def vgg_loss(custom_vgg, target, pred, weights=None, vgg_feat_weights=None):
    """

    :param custom_vgg:
    :param target:
    :param pred:
    :return:
    """
    target_feats = custom_vgg(target)
    pred_feats = custom_vgg(pred)
    if weights is None:

        loss = torch.cat(
            [
                vgg_feat_weights[i]
                * torch.mean(torch.abs(tf - pf), dim=[1, 2, 3]).unsqueeze(dim=-1)
                for i, (tf, pf) in enumerate(zip(target_feats, pred_feats))
            ],
            dim=-1,
        )
    else:
        pix_loss = [
            vgg_feat_weights[0]
            * torch.mean(weights * torch.abs(target_feats[0] - pred_feats[0]))
            .unsqueeze(dim=-1)
            .to(torch.float)
        ]
        loss = torch.cat(
            pix_loss
            + [
                vgg_feat_weights[i + 1]
                * torch.mean(torch.abs(tf - pf), dim=[1, 2, 3]).unsqueeze(dim=-1)
                for i, (tf, pf) in enumerate(zip(target_feats[1:], pred_feats[1:]))
            ],
            dim=-1,
        )

    loss = torch.sum(loss, dim=1)
    return loss


class PerceptualVGG(torch.nn.Module):
    def __init__(self, vgg):
        super().__init__()
        # self.vgg = vgg19(pretrained=True)
        if isinstance(vgg, torch.nn.DataParallel):
            self.vgg_layers = vgg.module.features
        else:
            self.vgg_layers = vgg.features

        self.input_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )

    def forward(self, x):
        out = {"input": x}
        # normalize in between 0 and 1
        x = scale_img(x)
        # normalize appropriate for vgg
        x = torch.stack([self.input_transform(el) for el in torch.unbind(x)])

        for name, submodule in self.vgg_layers._modules.items():
            # x = submodule(x)
            if name in VGGTargetLayers:
                x = submodule(x)
                out[VGGTargetLayers[name]] = x
            else:
                with torch.no_grad():
                    x = submodule(x)

        return VGGOutput(**out)


class PerceptualLossInstances(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        vgg = vgg19(pretrained=True)
        vgg.eval()
        self.custom_vgg = PerceptualVGG(vgg)

        self.vgg_feat_weights = config.setdefault(
            "vgg_feat_weights", (len(VGGTargetLayers) + 1) * [1.0]
        )
        assert len(self.vgg_feat_weights) == len(VGGTargetLayers) + 1

    def forward(self, image, target):
        loss = vgg_loss(self.custom_vgg, target, image, vgg_feat_weights=self.vgg_feat_weights)
        return loss
