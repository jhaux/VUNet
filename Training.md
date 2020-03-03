# Training

To start a training of VUNet you need two things:

1. A config with all relevant parameters. Take a look at the [supplied
   configs](VUNet/configs/).
2. The code that defines, how a parameter update takes place given the model
   and input data. This code is defined in the `step_op` method of the
   `Iterator` found in [iterator.py](VUNet/iterator.py)

:information_source: A training is started with the command `edflow -n
<experiment name> -b <config1> <config2> -t [--potential_additional_parameter
value]`. The `-t` flag tells `edflow`, that we want to train our model. Should
it not be passed, `edflow` is in evaluation mode, which will produce only
evaluation outputs.


## Configs

All parameters we potentially want to vary between experiments or that we want
to explicitly log for documentation purposes, are stored in a central config
object. This config object, basically a nested dict, is instantiated when
invoking the `edflow` command. It will be passed to all relevant objects, like
the datasets, the iterator (see below) and the model, upon their construction.

The config is defined by all base configs in the `yaml` format passed to `edflow`
via the `-b` flag, as well as any number of additional arguments passed to
`edflow` in the form of `--parameter value`. You can also change parameters
defined in the base configs, by passing their name and new value in that way.
Nestedness can also be reflected in the parameter name by writing it out in a
path like fashion, e.g. `--losses/color_L1/weight 2`.

Before starting the actual training and after instantiation of model, iterator
and datasets, the actually used config will be displayed, so that you can make
sure, everything is as expected.


## The Iterator

The Iterator is the central training and evaluation object. It defines, how our
model is loaded, stored and updated, as well as how logs and evaluations are
created. As its name implicates the Iterator iterates over all given data.
While doing so it applies the method `step_op` on the data.
The `step_op` returns a set of dynamically created functions, which are called
at various stages of the training.

Also depending on the state of the training, `step_op` is called on the
train split of the data, which is defined in the config under `datasets/train`
updating the model parameters as defined in the `train_op` function and logging
from time to time. When logging the model is also applied on the validation
split of the data (found in the config at `datasets/validation`) without
updating the model. Logging is done as requested in the dynamically created
`log_op` function.

At the end of each epoch the model is applied to the whole sorted validation
split and all outputs of the `eval_op` function are written out in such a
manner, that they can be easily loaded again later for further evaluation.

The `train_op` is called at every step, if `edflow` was invoked with the `-t`
parameter. The `log_op` is only called every few steps. Finally the `eval_op`
is only called at the end of each epoch. Take a look at the documentation of
`edflow.eval.pipeline` for more information.


### A closer look at the `step_op`

Additionally to defining the three functions `train_op`, `log_op` and
`eval_op`, the `step_op` also takes care of various other things. Let's go
through it step by step:

```python
    def step_op(self, model, stickman, appearance, target, **kwargs):
```

The arguments expected by the `step_op` method are defined by what the dataset
returns per example, except for the very first argument, which is the model, we
want to train. In our case the model is `VUNet.VUNet` and the dataset returns
a `dict` with at least the three entries `stickman` for  the pose, `appearance`
and `target` as a training reference.

```python
        # Test if the model should be pushed to the gpu
        if not hasattr(self, '_model_is_cuda'):
            if torch.cuda.is_available():
                self.model.cuda()
                self._model_is_cuda = True
            else:
                self._model_is_cuda = False
```

As a first step, we check if the model is already pushed to a GPU and if not,
do so right away.

```python
        # prepare inputs for preprocessing
        inputs = {'pt': {'pose': stickman, 'appearance': appearance, 'target': target}}

        # set model to train / eval mode
        is_train = self.get_split() == "train"
        model.train(is_train)

        # prepare inputs
        self.prepare_inputs_inplace(inputs)
        # remember numpy versions
        inputs['np'] = {'pose': stickman, 'appearance': appearance, 'target': target}
```

After that we prepare the inputs to be `pytorch` tensors and set the model to
train mode, depending on the state of the training. Remember: E.g. during
logging the state will change to eval model, when iterating over the validation
set.

```python
        # compute model
        predictions = model(inputs['pt'])
        predictions = {'image': predictions}
        predictions.update(model.saved_tensors)

        # compute loss
        losses = self.criterion(inputs['pt'], predictions)

        loss = losses['batch']['total']
```

Now it is time to actually apply the model to the data and compute the losses
of the generated outputs w.r.t. to the targets. How we calculate the losses
will be discussed in further detail below. 

Up to now we defined operations that are executed at each step during training.
Now we have a look at those steps, that can be turned on and of as needed
during training and evaluation to optimize the execution performance.

```python
        def train_op():
            before = time.time()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if retrieve(self.config, "debug_timing", default=False):
                self.logger.info("train step needed {} s".format(time.time() - before))
```

The train op simply defines how the loss is used to calculate the parameter
updates. This is completely according to `pytorch` way of things.

```python
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
```

The `log_op`, as explained above will only be called every now and then. Here
we make sure that all log values are `numpy` arrays or python scalars. We also
ensure, that the returned `logs` dictionary adheres to the required format,
which is `logs = {'images: {'name': image_batch, 'name2': image_batch2, ...},
'scalars': {'name': value, 'name2: value2, ...}}`.

```python
        def eval_op():
            with torch.no_grad():
                logs = self.prepare_logs(
                    inputs, predictions, losses, model, 'instances'
                )

            return {
                **logs["images"],
            }
```

Finally we define the `eval_op`. It will turn a batch of model outputs and
losses into entries of a dataset, which is written into the projects `eval/`
folder. Note that we need to make sure, that the outputs of this function do
all need to keep the batch dimensions!

```python
        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}
```

The three dynamically created functions are then passed on to `edflow`, which
handles the execution.


### Playing with the losses

To add or change losses, we need to take a look at the method `criterion`:

```python
    def criterion(self, inputs, predictions):
```

It expects all inputs in the form of `pytorch` tensors, as well as the model
predictions, containing the generated images and the latent representations for
pose and appearance.

```python
        '''
        # update kl weights
        update_loss_weights_inplace(self.config["losses"], self.get_global_step())
```

As we want to vary some loss weights, namely ramp up the KL-loss during
training, we first update those weights depending on the current training step.

```python
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
```

Next, all relevant losses are calculated as given in the config under the key
`losses`. 
:information_source: To add a custom loss, simply calculate it here and store it in the
`instance_losses` dictionary. Note, that your calculated loss should still have
a correct batch dimension.

```python
        instance_losses["total"] = sum(
            [
                self.config["losses"][key]["weight"] * instance_losses[key]
                for key in instance_losses.keys()
            ]
        )
```

Knowing the absolute amounts of the losses, we then weigh and sum them
according to the weights, we define in the config under
`losses/<loss_name>/weight`. This total loss is the one, we optimize for during
training.

```python
        # reduce to batch granularity
        batch_losses = {k: v.mean() for k, v in instance_losses.items()}
        losses = dict(instances=instance_losses, batch=batch_losses)

        return losses
```

During training, we only want to log losses for the whole batch, i.e. scalar
values, but during evaluation it can be interesting to know losses for each
seen example independently. Thus we return the losses at both granularities.


## Conclusion

You now know the core elements for training VUNet and making changes to the
training process:

- The config defines all relevant parameters, which can be accessed during
  training via `self.config` inside the iterator.
- The `Iterator` defines how we want to train and evaluate the model. Inside
  `step_op` we define, how a batch of data is handled.
- To change the weight of a loss, simply change the corresponding parameter in
  the `losses` entry in the config, to add a loss, modify the `criterion`
  method of the iterator.
