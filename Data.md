# Data

As a person image generator VUNet needs two inputs, one that defines the
pose of the displayed person, one that defines its appearance.
Additionally for training we need to know what this combination of pose and
appearance looks like to be able to define a reconstruction loss.

This repo comes together with a small split of the custom `Prjoti` dataset,
which contains images of a person walking, along with keypoint annotations.
These keypoint annotations are converted into stickman images upon loading
single examples.


## Explore the data

Let us first of all have a look at the data by running the following command:

```bash
cd <VUNet root>/VUnet/configs/
edexplore -b prjoti.yaml --dataset VUNet.data.prjoti.Prjoti_VUNet_train
```
Note that you need to have `edflow` installed to run `edexplore`. This happens
automatically when you install this repo following
[README.md](README.md#installation).

This will start an interactive browser application, where you can inspect all
examples of the dataset.

You can see a single example, which contains an appearance image, a
stickman representation and a target image, which VUNet should generate.


## Define your own data

To use you own dataset, you need to make sure it follows the `edflow` dataset
conventions. Your dataset class should

1. inherit from the `edflow.data.dataset_mixin.DatasetMixin` class
2. implement a `get__example(self, idx)` method, which returns a `dict` given
   the example index `idx`.
3. define the `__len__` method.

Take a look at the `VUNet.data.prjoti.Prjoti_VUNet` class, to see these methods
implemented.

Also make sure that the returned example of your custom dataset has entries of
the same form as `VUNet.data.prjoti.Prjoti_VUNet`, i.e. you need to define the
keys `stickman`, `appearance`, `target`.


### `MetaDataset`: edflow's dataloading class

Should you plan on implementing a completely new data loader, consider taking a
look at the `edflow.data.believers.meta.MetaDataset` class and its
documentation. It defines highly efficient data loading routines and follows a
few concepts, which allows for easy loading of e.g. images. Take a look at the
`Prjoti` dataset and the folder its data is contained in to learn more about
the structure of a `MetaDataset`. The three core components of a `MetaDataset`
are

1. a `meta.yaml` which contains all dataset relevant information like a
   description, loading instructions and the like,
2. a directory `labels`, which contains `numpy.memorymap`s of informations for
   all examples in the dataset
3. possibly the actual load-heavy data, like images at some directory.

For load heavy data like images you will define an array of paths, which point
to all images and store is as label array. Inside the `meta.yaml` you then
explain the dataset class, that it is supposed to turn the paths into actual
images in the form of `numpy` arrays.
