# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""APG provider registration and metrics-family contracts (P3.2, P-A/T-F)."""

from hydra.core.config_store import ConfigStore

from motrix_rl import frameworks
from motrix_rl.runner import _METRICS_FAMILIES


def test_motrix_framework_aggregates_apg() -> None:
    import motrix_rl_builtin.plugin  # noqa: F401

    agents = frameworks.supported_agents("motrix")
    assert "apg" in agents and "fastsac" in agents
    provider = frameworks.get_agent_provider("motrix", "apg", "torch")
    assert provider is not None
    assert provider.agent_name == "apg"
    assert provider.train_backend == "torch"
    assert provider.checkpoint_format == "pt"


def test_apg_schema_installed_in_config_store() -> None:
    frameworks.get_framework("motrix")  # plugin load installs provider schemas

    store = ConfigStore.instance()
    node = store.repo.get("algo_base", {}).get("_motrix_apg_schema.yaml")
    assert node is not None


def test_apg_total_env_steps_accounts_fractional_collection() -> None:
    from motrix_rl_builtin.apg.config import ApgCfg

    cfg = ApgCfg()
    cfg.trainer.iterations = 30000
    cfg.trainer.utd = 0.4
    provider = frameworks.get_agent_provider("motrix", "apg", "torch")
    assert provider is not None
    assert provider.total_env_steps(cfg, 2048) == 30000 * 2.5 * 2048


def test_apg_registers_worldmodel_metrics_family() -> None:
    assert _METRICS_FAMILIES["apg"] == "rl.worldmodel"
