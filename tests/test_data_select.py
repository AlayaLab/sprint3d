"""Selection rules of the 10k set (data/select): caption analysis and the Toys4K name rule (CPU, no data)."""
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("numpy")

SELECT = Path(__file__).resolve().parents[1] / "data" / "select"


@pytest.fixture(scope="module")
def flags():
    sys.path.insert(0, str(SELECT))
    try:
        spec = importlib.util.spec_from_file_location("select_flags", SELECT / "flags.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(SELECT))


def test_toys4k_name_rule(flags):
    for name, tags, desc in [("Toys4K giraffe", [], ""), ("giraffe", ["toy4k"], ""), ("x", [], "from the TOYS 4K set"),
                             ("toy 4k", None, None)]:
        assert flags.mentions_toys4k(name, tags, desc), (name, tags, desc)
    for name, tags, desc in [("toy car 4k texture", [], ""), ("toys", ["4 k"], ""), ("Toyota 4K", [], "")]:
        assert not flags.mentions_toys4k(name, tags, desc), (name, tags, desc)


def test_head_noun_analysis(flags):
    head, first, noun_and, commas = flags.head_analysis("A wooden chair with a red cushion.")
    assert (head, first, noun_and, commas) == ("chair", "a", False, 0)
    assert flags.head_analysis("Black and white vase.")[:3] == ("vase", "black", False)  # adjective conjunction
    assert flags.head_analysis("Table and chairs.")[2] is True


def test_caption_flags(flags):
    f, short, long_ = flags.cap_flags(["Medieval village with houses.", "A medieval village on a hill with many houses."],
                                      "sketchfab")
    assert f["scene_head"] and short == "Medieval village with houses."
    f, _, _ = flags.cap_flags(["Two chairs.", "Two wooden chairs side by side."], "sketchfab")
    assert f["quant_plural"]
    f, _, _ = flags.cap_flags(["Kitchen island.", "A kitchen island with drawers."], "sketchfab")
    assert not f["scene_head"]  # compound exception
    f, _, _ = flags.cap_flags([], "hssd")
    assert f == {"no_caption": True}
