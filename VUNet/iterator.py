import torch
import torch.nn as nn
import torch.optim as optim
import time

import numpy as np
from edflow import TemplateIterator, get_logger
from edflow.util import retrieve, walk
from edflow.data.util import adjust_support

from VUNet.utils.loss_utils import (
    update_loss_weights_inplace,
    L1LossInstances,
    MSELossInstances,
    aggregate_kl_loss,
    PerceptualLossInstances,
)


def np2pt(array):
    '''Converts a numpy array to torch Tensor. If possible pushes to the
    GPU.'''
    tensor = torch.tensor(array, dtype=torch.float32)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    tensor = tensor.permute(0, 3, 1, 2)
    tensor = tensor.contiguous()
    return tensor


def pt2np(tensor):
    '''Converts a torch Tensor to a numpy array.'''
    array = tensor.detach().cpu().numpy()
    array = np.transpose(array, (0, 2, 3, 1))
    return array


class Iterator(TemplateIterator):
    '''The central training and evaluation Manager. Its method :meth:`step_op`
    defines, how batches of data are processed.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])

        self.L1LossInstances = L1LossInstances()
        self.MSELossInstances = MSELossInstances()
        if "perceptual" in self.config["losses"]:
            self.PerceptualLossInstances = PerceptualLossInstances(
                self.config["losses"]["perceptual"]
            )

        if torch.cuda.is_available():
            self.model.cuda()
            self.L1LossInstances.cuda()
            self.MSELossInstances.cuda()
            if "perceptual" in self.config["losses"]:
                self.PerceptualLossInstances.cuda()

    def save(self, checkpoint_path):
        '''Defines how to save the model. Called everytime a log output is
        produced and when ``ctrl-c``, i.e. ``KeybordInterrupt`` is invoked.
        '''
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        '''Defines how to load a model. Called when running edflow with the
        ``-p`` or ``-c`` parameter.'''
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def criterion(self, inputs, predictions):
        '''Combines all losses with weights as defined in the config. Add new
        losses here.
        '''
        # update kl weights
        update_loss_weights_inplace(self.config["losses"], self.get_global_step())

        # calculate losses
        instance_losses = {}

        if "color_L1" in self.config["losses"].keys():
            instance_losses["color_L1"] = self.L1LossInstances(
                predictions["image"], inputs["target"]
            )

        if "color_L2" in self.config["losses"].keys():
            instance_losses["color_L2"] = self.MSELossInstances(
                predictions["image"], inputs["target"]
            )

        if "color_gradient" in self.config["losses"].keys():
            instance_losses["color_gradient"] = self.L1LossInstances(
                torch.abs(
                    predictions["image"][..., 1:] - predictions["image"][..., :-1]
                ),
                torch.abs(inputs["target"][..., 1:] - inputs["target"][..., :-1]),
            ) + self.L1LossInstances(
                torch.abs(
                    predictions["image"][..., 1:, :] - predictions["image"][..., :-1, :]
                ),
                torch.abs(inputs["target"][..., 1:, :] - inputs["target"][..., :-1, :]),
            )

        if "KL" in self.config["losses"].keys() and "q_means" in predictions:
            instance_losses["KL"] = aggregate_kl_loss(
                predictions["q_means"], predictions["p_means"]
            )

        if "perceptual" in self.config["losses"]:
            instance_losses["perceptual"] = self.PerceptualLossInstances(
                predictions["image"], inputs["target"]
            )

        instance_losses["total"] = sum(
            [
                self.config["losses"][key]["weight"] * instance_losses[key]
                for key in instance_losses.keys()
            ]
        )

        # reduce to batch granularity
        batch_losses = {k: v.mean() for k, v in instance_losses.items()}

        losses = dict(instances=instance_losses, batch=batch_losses)

        return losses

    def prepare_inputs_inplace(self, inputs):
        '''Casts all input to torch Tensor and pushes them to the gpu.'''
        before = time.time()

        inputs = walk(inputs, np2pt, inplace=True)

        if retrieve(self.config, "debug_timing", default=False):
            self.logger.info("prepare of data needed {} s".format(time.time() - before))

    def prepare_logs(self, inputs, predictions, losses, model, granularity):
        '''Logs need to be differentiated into ``images`` and ``scalars``. This
        function casts everything we want to log to numpy and stores it
        correctly in the output log dictionary.
        '''
        # sample variational part
        output_sample = model(inputs["pt"], mode="sample_appearance")
        output_sample = {'image': output_sample}
        output_sample.update(model.saved_tensors)

        losses_sample = self.criterion(inputs["pt"], output_sample)
        sample_images = {
            "images_prediction_sample": pt2np(output_sample["image"]),
        }

        # concatenate logs
        logs = {
            "images": {
                "appearance": inputs["np"]["appearance"],
                "target": inputs["np"]["target"],
                "pose": inputs["np"]["pose"],
                "images_prediction": pt2np(predictions['image']),
            },
            "scalars": {
                **losses[granularity]
            },
        }

        # convert to numpy
        def conditional_convert2np(log_item):
            if isinstance(log_item, torch.Tensor):
                log_item = log_item.detach().cpu().numpy()
            return log_item

        walk(logs, conditional_convert2np, inplace=True)

        return logs

    def step_op(self, model, stickman, appearance, target, **kwargs):
        '''Evaluated for each batch in the suppied datasets

        Parameters
        ----------
        model : torch.nn.module
            The model to train. Usually ``Vunet.model.vunet.VUNet``.
        stickman : np.ndarray
            The pose input to VUNet. Should be an rgb stickman image, but at
            least be of same spatial size as :attr:`target` and with
            ``pose_channels`` channels as specified in the config under
            ``model_pars``.
        appearance : np.ndarray
            The appearance input to VUNet. Usually an RGB image.
        target : np.ndarray
            The groundtruth to the generated image.

        Returns
        -------
            op_dict : dict
                A dictionary specifying a ``train_op``, ``log_op`` and
                ``eval_op`` to be executed during various stages while training
                or evaluating.
        '''

        # Test if the model should be pushed to the gpu
        if not hasattr(self, '_model_is_cuda'):
            if torch.cuda.is_available():
                self.model.cuda()
                self._model_is_cuda = True
            else:
                self._model_is_cuda = False

        # prepare inputs for preprocessing
        inputs = {'pt': {'pose': stickman, 'appearance': appearance, 'target': target}}

        # set model to train / eval mode
        is_train = self.get_split() == "train"
        model.train(is_train)

        # prepare inputs
        self.prepare_inputs_inplace(inputs)
        # remember numpy versions
        inputs['np'] = {'pose': stickman, 'appearance': appearance, 'target': target}

        # compute model
        predictions = model(inputs['pt'])
        predictions = {'image': predictions}
        predictions.update(model.saved_tensors)

        # compute loss
        losses = self.criterion(inputs['pt'], predictions)

        loss = losses['batch']['total']

        def train_op():
            before = time.time()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if retrieve(self.config, "debug_timing", default=False):
                self.logger.info("train step needed {} s".format(time.time() - before))

        def log_op():
            with torch.no_grad():
                logs = self.prepare_logs(inputs, predictions, losses, model, 'batch')

            # log to tensorboard
            if self.config["integrations"]["tensorboardX"]["active"]:
                if self.get_global_step() == 0 and is_train:
                    # save model
                    self.tensorboardX_writer.add_graph(model, inputs["pt"])
                    self.logger.info("Added model graph to tensorboard")

            return logs

        def eval_op():
            with torch.no_grad():
                logs = self.prepare_logs(
                    inputs, predictions, losses, model, 'instances'
                )

            return {
                **logs["images"],
            }

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

    @property
    def callbacks(self):
        return {"eval_op": {"cb": eval_callback}}


def eval_callback(root, data_in, data_out, config):
    pass
