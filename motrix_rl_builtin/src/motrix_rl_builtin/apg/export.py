# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import io

from motrix_rl.deploy.api import OnnxModelArtifact, OnnxPolicyExporter
from motrix_rl.deploy.onnx_validation import validate_onnx_policy


class ApgOnnxExporter(OnnxPolicyExporter):
    def export(self, request):
        import onnx
        import torch

        from motrix_rl_builtin.apg.train import restore_policy

        checkpoint = torch.load(request.checkpoint, map_location="cpu", weights_only=True)
        policy = restore_policy(checkpoint)
        buffer = io.BytesIO()
        torch.onnx.export(
            policy,
            torch.zeros(1, policy.input_dim),
            buffer,
            opset_version=request.opset,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={"obs": {0: "batch_size"}, "actions": {0: "batch_size"}},
            dynamo=False,
        )
        model = onnx.load_model_from_string(buffer.getvalue())
        onnx.helper.set_model_props(
            model,
            {
                "algo": "apg",
                "rllib": "motrix",
                "train_backend": "torch",
                "env_name": request.run.metadata.env_name,
                "obs_dim": str(policy.input_dim),
                "action_dim": str(checkpoint["action_dim"]),
                "history_layout": "frame-major, oldest-first",
                "history_length": str(policy.actor.history_length),
                "proprio_dim": str(policy.actor.proprio_dim),
                "goal_dim": str(policy.actor.goal_dim),
            },
        )
        onnx.checker.check_model(model)
        model_bytes = model.SerializeToString()

        def reference(obs):
            with torch.inference_mode():
                return policy(torch.from_numpy(obs)).numpy()

        report = validate_onnx_policy(
            model_bytes,
            reference,
            observation_size=policy.input_dim,
            action_size=checkpoint["action_dim"],
            config=request.parity,
            source="motrix/torch/apg",
        )
        return OnnxModelArtifact(model_bytes=model_bytes, report=report)
