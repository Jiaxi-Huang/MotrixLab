# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""LAFAN1 → SONIC corpus conversion: the retarget and BVH→SMPL pathways.

This package is independent of the WBT-facing ``lafan_converter`` (which bakes
single retargeted clips for replay / whole-body tracking). It produces SONIC
training corpora: one MotrixLab motion NPZ v1 per clip carrying the
``ext_smpl_joints`` / ``ext_smpl_root_quat`` reference channels, directly
consumable by :meth:`motrix_envs.motion.sonic.SonicMotionClip.from_corpus`.

Two pathways meet in every output clip:

- ``retarget`` — the retargeted G1 side: LAFAN1_Retargeting_Dataset G1 CSVs
  (root pose + joint angles) resampled to 50 Hz and baked through MotrixSim
  forward kinematics (``retarget.py``).
- ``bvh`` → ``smpl`` — the human reference side: raw LAFAN1 BVH rotations
  mapped onto the SMPL joint hierarchy (``bvh.py``), then materialized into
  the SONIC SMPL frame contract (``smpl.py``).

``scripts/motion/download_lafan4sonic.py`` materializes the paired raw
sources; ``scripts/motion/convert_lafan4sonic.py`` runs the conversion.
"""
