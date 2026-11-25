# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Hidden-matrix Muon with Adam for projections, biases and normalization."""

import torch
from torch import nn


class MuonAdam:
    """Keep input/output projections, biases and normalization parameters on Adam."""

    def __init__(self, model, networks, lr, betas, fused):
        matrices = []
        for network in networks:
            layers = [layer for layer in network.modules() if isinstance(layer, nn.Linear)]
            matrices.extend(layer.weight for layer in layers[1:-1])
        if not matrices:
            raise ValueError("APG Muon requires at least two hidden layers")
        matrix_ids = {id(parameter) for parameter in matrices}
        remaining = [parameter for parameter in model.parameters() if id(parameter) not in matrix_ids]
        self.muon = torch.optim.Muon(matrices, lr=lr, weight_decay=0, adjust_lr_fn="match_rms_adamw")
        self.adam = torch.optim.Adam(remaining, lr=lr, betas=betas, fused=fused)

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adam.param_groups

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adam.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state):
        self.muon.load_state_dict(state["muon"])
        self.adam.load_state_dict(state["adam"])


def make_optimizer(model, networks, optimizer, learning_rate, *, betas, device):
    if optimizer == "muon":
        return MuonAdam(model, networks, learning_rate, betas, device.type == "cuda")
    return torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, fused=device.type == "cuda")
