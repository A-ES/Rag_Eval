from pathlib import Path

from config import Settings


def test_settings_defaults_point_to_project_data_dirs() -> None:
    settings = Settings()

    assert settings.raw_data_dir == settings.project_root / "data" / "raw"
    assert settings.processed_data_dir == settings.project_root / "data" / "processed"
    assert isinstance(settings.project_root, Path)
