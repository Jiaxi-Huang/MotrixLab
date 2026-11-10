# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""APG Torch checkpoint to deterministic ONNX policy adapter."""

import io
from collections.abc import Mapping

import numpy as np

from motrix_rl.deploy.api import OnnxExportRequest, OnnxModelArtifact, OnnxPolicyExporter
from motrix_rl.deploy.onnx_validation import validate_onnx_policy
from motrix_rl.normalize import EmpiricalNormalization


class ApgOnnxExporter(OnnxPolicyExporter):
    """Restore an APG actor and export its deterministic inference path.

    The exported graph input is the flattened policy-observation history
    stack ``(batch, history * obs_dim)``; the deploy side maintains the stack
    (episode starts broadcast-fill, as in play). The graph folds the
    observation normalization and the ``action_limit`` scale, so its action
    output is the environment command directly.
    """

    def export(self, request: OnnxExportRequest) -> OnnxModelArtifact:
        try:
            import onnx
            import torch
            from torch import nn
        except ImportError as error:
            raise RuntimeError("APG ONNX export requires the 'motrix-rl[onnx]' extra") from error

        from motrix_rl_builtin.apg.networks import MlpActor

        checkpoint = torch.load(request.checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"APG checkpoint must be a mapping: {request.checkpoint}")
        dims = checkpoint["task"]
        actor_state = _module_state(checkpoint, "actor", request.checkpoint)
        obs_dim = int(dims["obs_dim"])
        action_dim = int(dims["action_dim"])

        actor = MlpActor(
            obs_dim,
            action_dim,
            request.task_config.algo.actor.hidden,
            output_activation=request.task_config.algo.actor.output_activation,
        )
        actor.load_state_dict(actor_state, strict=True)
        actor.eval()

        normalizer = EmpiricalNormalization(shape=int(dims["raw_obs_dim"]), device="cpu")
        normalizer.load_state_dict(_module_state(checkpoint, "obs_norm", request.checkpoint), strict=True)
        normalizer.eval()
        action_limit = float(dims["action_limit"])
        history = int(dims["history"])

        class DeterministicPolicy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.actor = actor
                self.normalizer = normalizer

            def forward(self, observations: torch.Tensor) -> torch.Tensor:
                frames = observations.reshape(observations.shape[0], history, -1)
                normalized = self.normalizer(frames, update=False)
                return action_limit * self.actor(normalized.flatten(1))

        policy = DeterministicPolicy().eval()
        buffer = io.BytesIO()
        torch.onnx.export(
            policy,
            torch.zeros(1, obs_dim, dtype=torch.float32),
            buffer,
            export_params=True,
            opset_version=request.opset,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch_size"}, "actions": {0: "batch_size"}},
            dynamo=False,
        )
        model_proto = onnx.load_model_from_string(buffer.getvalue())
        onnx.helper.set_model_props(model_proto, _metadata(request, obs_dim, action_dim))
        onnx.checker.check_model(model_proto)
        model_bytes = model_proto.SerializeToString()

        def reference(observations: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                return policy(torch.from_numpy(observations)).numpy()

        report = validate_onnx_policy(
            model_bytes,
            reference,
            observation_size=obs_dim,
            action_size=action_dim,
            config=request.parity,
            source="motrix/torch/apg",
        )
        return OnnxModelArtifact(model_bytes=model_bytes, report=report)


def _module_state(checkpoint: Mapping, name: str, path) -> Mapping:
    state = checkpoint.get(name)
    if not isinstance(state, Mapping):
        raise ValueError(f"APG checkpoint has no {name!r} state_dict: {path}")
    return state


def _metadata(request: OnnxExportRequest, observation_size: int, action_size: int) -> dict[str, str]:
    metadata = request.run.metadata
    return {
        "env_name": metadata.env_name,
        "rllib": metadata.rllib,
        "train_backend": metadata.train_backend,
        "algo": metadata.algo,
        "obs_dim": str(observation_size),
        "action_dim": str(action_size),
        "source_checkpoint": str(request.checkpoint),
    }
