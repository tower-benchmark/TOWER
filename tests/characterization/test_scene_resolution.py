import pytest

from scripts.maintainer.golden_support import GOLDEN, read
from tests.characterization.scene_resolution import scene_resolution_table


@pytest.mark.archive
def test_all_600_scene_resolutions(v1_export) -> None:
    assert scene_resolution_table(v1_root=v1_export) == read(GOLDEN / "scene-resolve-sha256.json")
