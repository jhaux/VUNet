from edflow.data.believers.meta import MetaDataset
from edflow.data.dataset_mixin import DatasetMixin
from edflow.data.agnostics.subdataset import SubDataset
from edflow.data.util import adjust_support
from edflow.util import PRNGMixin
import os
import numpy as np

from VUNet.data.stickman import kp2stick
from VUNet.data.keypoint_models import OPENPOSE_18


class Prjoti(MetaDataset):
    def __init__(self, config):
        root = config["data_root"]

        if not os.path.exists(os.path.join(root, 'meta.yaml')):
            from VUNet.data.download import prjoti_installer
            root = prjoti_installer(root)
            config['data_root'] = root

        super().__init__(root)


class Prjoti_VUNet(DatasetMixin, PRNGMixin):
    def __init__(self, config):
        self.base = Prjoti(config)
        self.labels = self.base.labels

    def get_example(self, idx):
        target = self.base[idx]['crop']()

        pose = self.base.labels['kps_rel'][idx]
        pose = pose * np.array(target.shape[:2])[None]
        stickman = kp2stick(pose, size=[256, 256], kp_model=OPENPOSE_18)
        stickman = adjust_support(stickman, '-1->1', '0->255')

        app_idx = self.prng.choice(len(self.base))
        appearance = self.base[app_idx]['crop']

        return {'stickman': stickman, 'appearance': appearance, 'target': target}

    def __len__(self):
        return len(self.base)


class Prjoti_VUNet_train(DatasetMixin):
    def __init__(self, config):
        self.P = Prjoti_VUNet(config)
        self.data = SubDataset(self.P, np.arange(0, int(0.9 * len(self.P))))


class Prjoti_VUNet_val(DatasetMixin):
    def __init__(self, config):
        self.P = Prjoti_VUNet(config)
        self.data = SubDataset(self.P, np.arange(int(0.9 * len(self.P)), len(self.P)))
