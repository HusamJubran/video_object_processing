import sys
from os import path as osp
sys.path.append(osp.dirname(__file__))
#sys.path.append("/home/kqsun/Tasks/magicpony_tmp/preprocessing_data/")
from externals.RAFT.core.raft import RAFT
import torch
import torch.nn.functional as F


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class FlowModel():
    def __init__(self, model, device):
        args = AttrDict({'model': model, 'small': False, 'mixed_precision': False, 'alternate_corr': False})
        self.model = self.load_model(args, device)
        self.device = device

    @staticmethod
    def load_model(args, device):
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))
        model = model.module
        model.to(device)
        model.eval()
        return model

    def preprocess_image(self, image):
        image = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device)
        image = image[None]
        padder = InputPadder(image.shape)
        return padder.pad(image)[0], padder

    def compute_flow(self, frame, next_frame, iters=20):
        frame, padder = self.preprocess_image(frame)
        next_frame, padder = self.preprocess_image(next_frame)
        _, flow = self.model(frame, next_frame, iters=iters, test_mode=True)
        return padder.unpad(flow)[0].permute(1, 2, 0).detach().cpu().numpy()