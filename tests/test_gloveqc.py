import numpy as np
import pytest

from demo import score
from gloveqc import qc
from gloveqc.synth import write


@pytest.fixture(scope="module")
def shift(tmp_path_factory):
    root = tmp_path_factory.mktemp("shift")
    labels = write(root, seed=26)
    return root, labels, {r.ep_id: r for r in qc.run_all(root)}


def test_all_planted_faults_found_no_false(shift):
    _, labels, reps = shift
    tp, fp, fn, rob_ok, rob_n = score(list(reps.values()), labels)
    assert (fp, fn) == (0, 0) and tp == 16 and rob_ok == rob_n


def test_offset_and_drift_recovered(shift):
    _, _, reps = shift
    assert reps["G-03-e06"].sync["offset_ms"] == pytest.approx(180, abs=10)
    assert reps["G-03-e07"].sync["drift_ms_per_min"] == pytest.approx(60, abs=8)
    assert reps["G-03-e07"].sync["after_ms"] < 33


def test_light_grip_pinky_is_not_called_dead(shift):
    _, _, reps = shift
    assert "TAXEL_DEAD" not in reps["G-03-e02"].codes      # cereal box: pinky carries only ~1 N


def test_retargeting_tags(shift):
    _, _, reps = shift
    wide = [r for r in reps.values() if r.meta["object"] == "wide bin" and r.verdict != "REJECT"][0]
    assert "parallel_gripper" not in wide.robots and "three_finger" in wide.robots


def test_rollup_tells_ops_what_to_do(shift):
    _, _, reps = shift
    roll = " | ".join(qc.glove_rollup(list(reps.values())))
    assert "G-07: middle taxel zero creeping" in roll and "recalibrate" in roll
    assert "G-11: ring taxel dead in 4 episodes from G-11-e05" in roll


def test_theil_sen_ignores_one_bad_window():
    c = np.arange(5, 120, 10.0)
    o = np.full_like(c, 0.002)
    o[-1] = -0.05
    icpt, slope, worst = qc.robust_fit(c, o, 120)
    assert slope == 0 and worst < 0.033


def test_more_seeds_no_errors(tmp_path):
    labels = write(tmp_path / "s", seed=3)
    reps = qc.run_all(tmp_path / "s")
    tp, fp, fn, rob_ok, rob_n = score(reps, labels)
    assert (fp, fn) == (0, 0) and rob_ok == rob_n
