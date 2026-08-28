import pytest

from tests.fixtures.synthetic import SyntheticTruth, make_clip


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory) -> SyntheticTruth:
    """One rendered synthetic clip shared by the whole test session."""
    path = tmp_path_factory.mktemp("clips") / "synthetic.mp4"
    return make_clip(str(path))
