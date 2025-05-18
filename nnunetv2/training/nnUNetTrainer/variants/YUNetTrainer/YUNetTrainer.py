import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class YUNetTrainer(nnUNetTrainer):
    def _set_batch_size_and_oversample(self):
        super()._set_batch_size_and_oversample()
        self.batch_size = 4
