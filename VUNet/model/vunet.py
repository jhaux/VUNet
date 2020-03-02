import torch
from torch import nn
from torch.nn import ModuleDict, ModuleList, Conv2d
from edflow import get_logger
from edflow.util import retrieve
from .modules import (
    VUnetResnetBlock,
    Upsample,
    Downsample,
    NormConv2d,
    SpaceToDepth,
    DepthToSpace,
)
import numpy as np
from torch.distributions import MultivariateNormal

from VUNet.model.util import calc_input_channels, cat_inputs


class VUnetEncoder(nn.Module):
    def __init__(
        self,
        n_stages,
        nf_in=3,
        nf_start=64,
        nf_max=128,
        n_rnb=2,
        conv_layer=NormConv2d,
        dropout_prob=0.0,
    ):
        super().__init__()
        self.in_op = conv_layer(nf_in, nf_start, kernel_size=1)
        nf = nf_start
        self.blocks = ModuleDict()
        self.downs = ModuleDict()
        self.n_rnb = n_rnb
        self.n_stages = n_stages
        for i_s in range(self.n_stages):
            # prepare resnet blocks per stage
            if i_s > 0:
                self.downs.update(
                    {
                        f"s{i_s+1}": Downsample(
                            nf, min(2 * nf, nf_max), conv_layer=conv_layer
                        )
                    }
                )
                nf = min(2 * nf, nf_max)

            for ir in range(self.n_rnb):
                stage = f"s{i_s+1}_{ir+1}"
                self.blocks.update(
                    {
                        stage: VUnetResnetBlock(
                            nf, conv_layer=conv_layer, dropout_prob=dropout_prob
                        )
                    }
                )

    def forward(self, x):
        out = {}
        h = self.in_op(x)
        for ir in range(self.n_rnb):
            h = self.blocks[f"s1_{ir+1}"](h)
            out[f"s1_{ir+1}"] = h

        for i_s in range(1, self.n_stages):

            h = self.downs[f"s{i_s+1}"](h)

            for ir in range(self.n_rnb):
                stage = f"s{i_s+1}_{ir+1}"
                h = self.blocks[stage](h)
                out[stage] = h

        return out


class ZConverter(nn.Module):
    def __init__(self, n_stages, nf, device, conv_layer=NormConv2d, dropout_prob=0.0):
        super().__init__()
        self.n_stages = n_stages
        self.device = device
        self.blocks = ModuleList()
        for i in range(3):
            self.blocks.append(
                VUnetResnetBlock(
                    nf, use_skip=True, conv_layer=conv_layer, dropout_prob=dropout_prob
                )
            )
        self.conv1x1 = conv_layer(nf, nf, 1)
        self.up = Upsample(in_channels=nf, out_channels=nf, conv_layer=conv_layer)
        self.channel_norm = conv_layer(2 * nf, nf, 1)

        self.d2s = DepthToSpace(block_size=2)
        self.s2d = SpaceToDepth(block_size=2)

    def forward(self, x_f):
        params = {}
        zs = {}
        h = self.conv1x1(x_f[f"s{self.n_stages}_2"])

        for n, i_s in enumerate(range(self.n_stages, self.n_stages - 2, -1)):
            stage = f"s{i_s}"

            h = self.blocks[2 * n](h, x_f[stage + "_1"])

            params[stage] = h
            if params[stage].shape[-1] != 1:
                params[stage] = self.s2d(params[stage])

            if x_f[stage + "_2"].shape[-1] > 1:
                h = self.s2d(h)
                z = self._latent_sample(h)
                z = self.d2s(z)
            else:
                z = self._latent_sample(h)
            zs[stage] = z

            # post
            if n == 0:
                gz = torch.cat([x_f[stage + "_1"], z], dim=1)
                gz = self.channel_norm(gz)
                h = self.blocks[2 * n + 1](h, gz)
                h = self.up(h)

        return params, zs

    def _latent_sample(self, mean):
        sample_mean = torch.squeeze(torch.squeeze(mean, dim=-1), dim=-1)

        sampled = MultivariateNormal(
            loc=torch.zeros_like(sample_mean, device=self.device),
            covariance_matrix=torch.eye(sample_mean.shape[-1], device=self.device),
        ).sample()

        return (sampled + sample_mean).unsqueeze(dim=-1).unsqueeze(dim=-1)


class VUnetDecoder(nn.Module):
    def __init__(
        self,
        n_stages,
        nf=128,
        nf_out=3,
        n_rnb=2,
        conv_layer=NormConv2d,
        spatial_size=256,
        final_act=True,
        dropout_prob=0.0,
    ):
        super().__init__()
        assert (2 ** (n_stages - 1)) == spatial_size
        self.final_act = final_act
        self.blocks = ModuleDict()
        self.ups = ModuleDict()
        self.n_stages = n_stages
        self.n_rnb = n_rnb
        for i_s in range(self.n_stages - 2, 0, -1):
            # for final stage, bisect number of filters
            if i_s == 1:
                # upsampling operations
                self.ups.update(
                    {
                        f"s{i_s+1}": Upsample(
                            in_channels=nf, out_channels=nf // 2, conv_layer=conv_layer,
                        )
                    }
                )
                nf = nf // 2
            else:
                # upsampling operations
                self.ups.update(
                    {
                        f"s{i_s+1}": Upsample(
                            in_channels=nf, out_channels=nf, conv_layer=conv_layer,
                        )
                    }
                )

            # resnet blocks
            for ir in range(self.n_rnb, 0, -1):
                stage = f"s{i_s}_{ir}"
                self.blocks.update(
                    {
                        stage: VUnetResnetBlock(
                            nf,
                            use_skip=True,
                            conv_layer=conv_layer,
                            dropout_prob=dropout_prob,
                        )
                    }
                )

        # final 1x1 convolution
        self.final_layer = conv_layer(nf, nf_out, kernel_size=1)

        # conditionally: set final activation
        if self.final_act:
            self.final_act = nn.Tanh()

    def forward(self, x, skips):
        """

        Parameters
        ----------
        x : torch.Tensor
            Latent representation to decode.
        skips : dict
            The skip connections of the VUnet

        Returns
        -------
        out : torch.Tensor
            An image as described by :attr:`x` and :attr:`skips`
        """
        out = x
        for i_s in range(self.n_stages - 2, 0, -1):
            out = self.ups[f"s{i_s+1}"](out)

            for ir in range(self.n_rnb, 0, -1):
                stage = f"s{i_s}_{ir}"
                out = self.blocks[stage](out, skips[stage])

        out = self.final_layer(out)
        if self.final_act:
            out = self.final_act(out)
        return out


# IMPORTANT: upsampling always uses the same number of filters in and out, The number changes before in the second resnet block!!
class VUnetBottleneck(nn.Module):
    def __init__(
        self,
        n_stages,
        nf,
        device,
        n_rnb=2,
        n_auto_groups=4,
        conv_layer=NormConv2d,
        dropout_prob=0.0,
    ):
        super().__init__()
        self.device = device
        self.blocks = ModuleDict()
        self.channel_norm = ModuleDict()
        self.conv1x1 = conv_layer(nf, nf, 1)
        self.up = Upsample(in_channels=nf, out_channels=nf, conv_layer=conv_layer)
        self.depth_to_space = DepthToSpace(block_size=2)
        self.space_to_depth = SpaceToDepth(block_size=2)
        self.n_stages = n_stages
        self.n_rnb = n_rnb
        # number of autoregressively modeled groups
        self.n_auto_groups = n_auto_groups
        for i_s in range(self.n_stages, self.n_stages - 2, -1):
            self.channel_norm.update({f"s{i_s}": conv_layer(2 * nf, nf, 1)})
            for ir in range(self.n_rnb):
                self.blocks.update(
                    {
                        f"s{i_s}_{ir+1}": VUnetResnetBlock(
                            nf,
                            use_skip=True,
                            conv_layer=conv_layer,
                            dropout_prob=dropout_prob,
                        )
                    }
                )

        self.auto_blocks = ModuleList()
        # model the autoregressively groups rnb
        for i_a in range(4):
            if i_a < 1:
                self.auto_blocks.append(
                    VUnetResnetBlock(
                        nf, conv_layer=conv_layer, dropout_prob=dropout_prob
                    )
                )
                self.param_converter = conv_layer(4 * nf, nf, kernel_size=1)
            else:
                self.auto_blocks.append(
                    VUnetResnetBlock(
                        nf,
                        use_skip=True,
                        conv_layer=conv_layer,
                        dropout_prob=dropout_prob,
                    )
                )

    def forward(self, x_e, z_post, mode="train"):
        """

        Parameters
        ----------
        x_e : torch.Tensor
            The output from the encoder E_theta
        z_post : torch.Tensor
            The output from the encoder F_phi
        mode : str
            Determines the mode of the bottleneck, must be in
            ["train","appearance_transfer","sample_appearance"]

        Returns
        -------
        h : torch.Tensor
            the output of the last layer of the bottleneck which is
            subsequently used by the decoder.
        posterior_params : torch.Tensor
            The flattened means of the posterior distributions p(z|ŷ,x) of the
            two bottleneck stages.
        prior_params : dict(str: torch.Tensor)
            The flattened means of the prior distributions p(z|ŷ) of the two
            bottleneck stages.
        z_prior : torch.Tensor
            The current samples of the two stages of the prior distributions of
            both two bottleneck stages, flattened.
        """
        p_params = {}
        z_prior = {}

        use_z = mode == "train" or mode == "appearance_transfer"

        h = self.conv1x1(x_e[f"s{self.n_stages}_2"])
        for i_s in range(self.n_stages, self.n_stages - 2, -1):
            stage = f"s{i_s}"
            spatial_size = x_e[stage + "_2"].shape[-1]

            h = self.blocks[stage + "_2"](h, x_e[stage + "_2"])

            if spatial_size == 1:
                p_params[stage] = h
                # posterior_params[stage] = z_post[stage + "_2"]
                prior_samples = self._latent_sample(p_params[stage])
                z_prior[stage] = torch.squeeze(
                    torch.squeeze(prior_samples, dim=-1), dim=-1
                )
                # posterior_samples = self._latent_sample(posterior_params[stage])
            else:

                if use_z:
                    z_flat = (
                        self.space_to_depth(z_post[stage])
                        if z_post[stage].shape[2] > 1
                        else z_post[stage]
                    )
                    sec_size = z_flat.shape[1] // 4
                    z_groups = torch.split(
                        z_flat, [sec_size, sec_size, sec_size, sec_size], dim=1
                    )

                param_groups = []
                sample_groups = []

                param_features = self.auto_blocks[0](h)
                param_features = self.space_to_depth(param_features)
                # convert to fitting depth
                param_features = self.param_converter(param_features)

                for i_a in range(len(self.auto_blocks)):
                    param_groups.append(param_features)

                    prior_samples = self._latent_sample(param_groups[-1])

                    sample_groups.append(prior_samples)

                    if i_a + 1 < len(self.auto_blocks):
                        if use_z:
                            feedback = z_groups[i_a]
                        else:
                            feedback = prior_samples

                        param_features = self.auto_blocks[i_a](param_features, feedback)

                p_params_stage = torch.cat(param_groups, dim=1)
                prior_samples = self.__merge_groups(sample_groups)
                p_params[stage] = p_params_stage
                z_prior[stage] = (
                    self.space_to_depth(prior_samples).squeeze(dim=-1).squeeze(dim=-1)
                )

            if use_z:
                z = (
                    self.depth_to_space(z_post[stage])
                    if z_post[stage].shape[-1] != h.shape[-1]
                    else z_post[stage]
                )
            else:
                z = prior_samples

            h = torch.cat([h, z], dim=1)
            h = self.channel_norm[stage](h)
            h = self.blocks[stage + "_1"](h, x_e[stage + "_1"])

            if i_s == self.n_stages:
                h = self.up(h)

        return h, p_params, z_prior

    def __split_groups(self, x):
        # split along channel axis
        sec_size = x.shape[1] // 4
        return torch.split(
            self.space_to_depth(x), [sec_size, sec_size, sec_size, sec_size], dim=1,
        )

    def __merge_groups(self, x):
        # merge groups along channel axis
        return self.depth_to_space(torch.cat(x, dim=1))

    def _latent_sample(self, mean):
        sample_mean = torch.squeeze(torch.squeeze(mean, dim=-1), dim=-1)

        sampled = MultivariateNormal(
            loc=torch.zeros_like(sample_mean, device=self.device),
            covariance_matrix=torch.eye(sample_mean.shape[-1], device=self.device),
        ).sample()

        return (sampled + sample_mean).unsqueeze(dim=-1).unsqueeze(dim=-1)


class VUnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logger = get_logger(self)
        self.config = config

        final_act = retrieve(config, "model_pars/final_act", default=False)
        nf_max = retrieve(config, "model_pars/nf_max", default=128)
        nf_start = retrieve(config, "model_pars/nf_start", default=64)
        spatial_size = retrieve(config, "model_pars/spatial_size", default=256)
        dropout_prob = retrieve(config, "model_pars/dropout_prob", default=0.0)
        f_in_channels = retrieve(config, "model_pars/img_channels", default=3)
        e_in_channels = retrieve(config, "model_pars/pose_channels", default=3)

        self.output_channels = 3

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # define required parameters
        n_stages = 1 + int(np.round(np.log2(spatial_size)))

        # if final activation shall be utilized, choose common pytorch convolution as conv layer, else custom Module that follows the original implementation
        conv_layer_type = Conv2d if final_act else NormConv2d

        # image processing encoder to produce the prosterior p( z | x,ŷ )
        self.f_phi = VUnetEncoder(
            n_stages=n_stages,
            nf_in=f_in_channels,
            nf_start=nf_start,
            nf_max=nf_max,
            conv_layer=conv_layer_type,
            dropout_prob=dropout_prob,
        )

        # stickman processing encoder to produce the prior p(z|ŷ)
        self.e_theta = VUnetEncoder(
            n_stages=n_stages,
            nf_in=e_in_channels,
            nf_start=nf_start,
            nf_max=nf_max,
            conv_layer=conv_layer_type,
            dropout_prob=dropout_prob,
        )

        # zconverter
        self.zc = ZConverter(
            n_stages=n_stages,
            nf=nf_max,
            device=device,
            conv_layer=conv_layer_type,
            dropout_prob=dropout_prob,
        )

        # bottleneck
        self.bottleneck = VUnetBottleneck(
            n_stages=n_stages,
            nf=nf_max,
            device=device,
            conv_layer=conv_layer_type,
            dropout_prob=dropout_prob,
        )

        # decoder
        self.decoder = VUnetDecoder(
            n_stages=n_stages,
            nf=nf_max,
            nf_out=self.output_channels,
            conv_layer=conv_layer_type,
            spatial_size=spatial_size,
            final_act=final_act,
            dropout_prob=dropout_prob,
        )
        self.saved_tensors = None

    def forward(self, inputs, mode="train"):
        '''
        Parameters
        ----------
        inputs : dict
            A dictionary containing two keys ``pose`` and ``appearance``.
            Behind each key must be a 3D torch Tensor.
        mode : str
            Defines the mode in which the Bottlence is used. Must be one of
            ``train, appearance_transfer, sample_appearance``. Default is
            ``train``.

        Returns
        -------
        out_img : torch.Tensor
            The generated image of a person in the pose of ``stickman`` and
            with the appearance of ``appearance`` from ``inputs``.

        Attributes
        ----------
        saved_tensors : dict
            Contains the q_means and p_means.
        '''

        # encoded shape image
        x_e = self.e_theta(inputs['pose'])
        # encoded appearance image
        x_f = self.f_phi(inputs['appearance'])
        # sample z
        q_means, zs = self.zc(x_f)
        # with prior and posterior distribution; don't use prior samples within this training
        if mode == "train":
            out_b, p_means, ps = self.bottleneck(x_e, zs, mode)
        elif mode == "appearance_transfer":
            out_b, p_means, ps = self.bottleneck(x_e, q_means, mode)
        elif mode == "sample_appearance":
            out_b, p_means, ps = self.bottleneck(x_e, {}, mode)
        else:
            raise ValueError(
                'The mode of vunet has to be one of ["train",'
                + '"appearance_transfer","sample_appearance"], but is '
                + mode
            )

        # decode
        out_img = self.decoder(out_b, x_e)

        self.saved_tensors = dict(q_means=q_means, p_means=p_means)
        return out_img
