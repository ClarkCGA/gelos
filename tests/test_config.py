import importlib
import pytest


def _reload_config():
    import gelos.config as config

    return importlib.reload(config)


def test_config_uses_data_root(tmp_path, monkeypatch):
    data_root = tmp_path / "data_root"
    project_root = tmp_path / "project_root"
    config_root = project_root / "gelos" / "configs"

    monkeypatch.setenv("GELOS_DATA_ROOT", str(data_root))
    monkeypatch.setenv("GELOS_PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("GELOS_CONFIG_DIR", str(config_root))

    config = _reload_config()

    assert config.RAW_DATA_DIR == data_root / "raw"
    assert config.PROCESSED_DATA_DIR == data_root / "processed"
    assert config.EXTERNAL_DATA_DIR == data_root / "external"
    assert config.CONFIG_DIR == config_root


def test_config_missing_paths_raises(monkeypatch):
    for var in [
        "GELOS_DATA_ROOT",
        "GELOS_RAW_DIR",
        "GELOS_INTERIM_DIR",
        "GELOS_PROCESSED_DIR",
        "GELOS_EXTERNAL_DIR",
        "GELOS_PROJECT_ROOT",
        "GELOS_CONFIG_DIR",
        "GELOS_MODELS_DIR",
        "GELOS_REPORTS_DIR",
        "GELOS_FIGURES_DIR",
    ]:
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ValueError, match="Missing required GELOS path environment variables"):
        _reload_config()
