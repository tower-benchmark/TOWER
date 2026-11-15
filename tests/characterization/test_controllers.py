from scripts.maintainer.golden_support import GOLDEN, read
from tests.characterization.controller_scenarios import controller_streams
from tests.characterization.golden_compare import assert_equivalent


def test_controller_modification_and_event_streams() -> None:
    assert_equivalent(read(GOLDEN / "controllers.json"), controller_streams())
