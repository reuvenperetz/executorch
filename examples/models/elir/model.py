# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
from torch.utils.data import DataLoader

from ELIR.datasets.celeba import CelebADataset
from ELIR.models.elir import Elir
from ..model_base import EagerModelBase


class ELIRModel(EagerModelBase):
    def __init__(self):
        pass

    def get_eager_model(self) -> Elir:
        from ELIR.models.elir import Elir
        model_params = {'fm_cfg': {'k_steps': 3, 'sigma_s': 0.1, 'latent_shape': [16, 32, 32], 'seed': 2025},
                        'fmir_cfg': {'name': 'lunet',
                                     'params': {'ch_mult': [1, 2, 1, 2], 'n_mid_blocks': 1, 'in_channels': 16,
                                                'hid_channels': 128, 'out_channels': 16,
                                                't_emb_dim': 160, 'overparametrization': False},
                                     'trainable': False},
                        'mmse_cfg': {'name': 'rrdbnet',
                                     'params': {'c_inout': 16, 'c_hid': 96, 'n_rrdb': 3, 'overparametrization': False},
                                     'trainable': False},
                        'enc_cfg': {'name': 'tiny_enc',
                                    'trainable': False},
                        'dec_cfg': {'name': 'tiny_dec',
                                    'trainable': False}}
        model = Elir(**model_params)
        return model

    def get_example_inputs(self):
        return (torch.randn(1, 3, 512, 512), )
        dataset = CelebADataset(os.path.join("/Users/reuper01/PycharmProjects/ELIR/ELIR/datasets/celebA/", "test"),
                                "sr",
                                32)
        # Loaders
        loader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            drop_last=False)
        x = (next(iter(loader))[0],)
        return x
