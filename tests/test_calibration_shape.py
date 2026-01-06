"""Can the instrument find a bug it is *told* is there?

The calibration milestone is gated on this: an instrument that cannot find a
known bug cannot find an unknown one, and nothing below it is admissible.

Two faults are injected here, and they are different bugs.

The first is an **inversion** -- a covariance that is honest on every scalar
while pointing the wrong way. The scalings are *solved* so the mean NEES lands
exactly on 1.0, leaving the scalar perfectly fooled, while the reported
`sigma_long/sigma_lat` is driven onto 0.809 against a true 3.030.

The second is **body honest, tail overconfident**, and it is the fault a trained
model actually has. It is easy to mistake for an inversion: "0.809 against
3.030" looks like one, but 0.809 is a median of per-frame ratios and 3.030 is a
ratio of RMS, and with a heavy tail those describe different populations.
Measured like for like both sides sit below 1.0 and there is no inversion at
all. What there is instead is an along-track tail the covariance never widens
for.

Both are injected from arithmetic rather than loaded from a checkpoint. A
checkpoint would make these tests depend on a file, an architecture and a
training run; the arithmetic depends on none of them.
"""

from __future__ import annotations

import math

import torch

from mapposeformer.metrics import Calibration

#: The true marginal sigmas: longitudinal error dominates, which is this
#: problem's signature -- lane geometry aliases along the road.
TRUE_SD = torch.tensor([0.30, 0.10, 0.02])

#: Enough frames that the KS statistic is stable to about a thousandth, which
#: is an order finer than any threshold asserted here.
N = 20_000


def _report(scale_long: float, scale_lat: float) -> dict[str, float]:
    """Errors drawn from :data:`TRUE_SD`, with a covariance mis-scaled per axis.

    ``gt`` is identity and ``pred`` is the error itself, so ``pose_error``
    returns exactly what was drawn and the test is measuring the instrument
    rather than the geometry.
    """
    torch.manual_seed(0)
    e = torch.randn(N, 3) * TRUE_SD
    reported = TRUE_SD * torch.tensor([scale_long, scale_lat, 1.0])
    cov = torch.diag_embed(reported.square()).expand(N, 3, 3).contiguous()
    calib = Calibration()
    calib.update(e.clone(), torch.zeros(N, 3), cov)
    return calib.as_dict()


def _nees_neutral_inversion(true_aniso: float, reported_aniso: float):
    """Axis scalings that invert the reported ratio at unchanged mean NEES.

    Mean NEES is ``(1/s_long^2 + 1/s_lat^2 + 1) / 3``, so holding it at 1.0
    requires ``1/s_long^2 + 1/s_lat^2 = 2``. With ``r = s_long / s_lat`` fixed
    by the anisotropy being asked for, that closes.
    """
    r = reported_aniso / true_aniso
    s_long = math.sqrt((1.0 + r * r) / 2.0)
    return s_long, s_long / r


def test_calibrated_covariance_is_reported_as_quiet():
    """No false positives: a correct covariance must not be accused."""
    d = _report(1.0, 1.0)
    for axis in ("long", "lat", "yaw"):
        assert d[f"pit_ks_{axis}"] < 0.05, axis
        assert 0.95 < d[f"tightness_{axis}"] < 1.05, axis
    # The anisotropy of TRUE_SD itself, recovered from both sides.
    assert abs(d["aniso_reported"] - 3.0) < 0.05
    assert abs(d["aniso_measured"] - 3.0) < 0.05


def test_the_scalar_is_fooled_by_a_nees_neutral_inversion():
    """The premise of the whole milestone, stated as a test.

    If NEES could see this, none of the machinery below would be worth having.
    """
    s_long, s_lat = _nees_neutral_inversion(3.030, 0.809)
    d = _report(s_long, s_lat)
    ratio = d["anees_median"] / Calibration.CHI2_MEDIAN
    # The band `Calibration.format` uses to print the word "calibrated".
    assert 0.8 < ratio < 1.25, f"NEES median {d['anees_median']} was not fooled"


def test_the_shape_statistics_catch_what_the_scalar_missed():
    """Opposite directions, and an anisotropy on the wrong side of 1.0."""
    s_long, s_lat = _nees_neutral_inversion(3.030, 0.809)
    d = _report(s_long, s_lat)

    # Too tight along track, too loose across it -- the directions are the
    # discriminating part. A merely over-wide covariance moves both the same
    # way, and this test would not distinguish it from an inversion otherwise.
    assert d["tightness_long"] > 1.25
    assert d["tightness_lat"] < 0.8

    reported, measured = d["aniso_reported"], d["aniso_measured"]
    assert (reported - 1.0) * (measured - 1.0) < 0, "inversion not flagged"
    assert abs(reported - 0.809) < 0.02, reported

    # Loud on at least one axis. Not both -- see the next test for why that
    # would be an impossible thing to ask for.
    assert max(d["pit_ks_long"], d["pit_ks_lat"]) >= 0.25


def test_the_fault_does_not_smear_onto_the_untouched_axis():
    """An instrument that lights up everywhere cannot say what to repair."""
    s_long, s_lat = _nees_neutral_inversion(3.030, 0.809)
    d = _report(s_long, s_lat)
    assert d["pit_ks_yaw"] < 0.05
    assert 0.95 < d["tightness_yaw"] < 1.05


def test_demanding_both_axes_be_loud_would_reject_every_such_inversion():
    """Why the gate is "either axis" and not "both", as an executable argument.

    Demanding PIT-KS >= 0.25 on long *and* lat looks stricter and is vacuous.
    Under the NEES-neutral constraint ``1/s_long^2 + 1/s_lat^2 = 2``, pushing
    the reported ratio further below 1.0 sends ``s_long`` towards ``1/sqrt(2)``
    and no further -- so the longitudinal axis is at most ``sqrt(2)`` too tight,
    and its KS statistic has a ceiling well under 0.25. A both-axes gate would
    reject the entire class of bug it was written for, and this fails if that
    ever stops being true.
    """
    # The limit of the family: an arbitrarily strong inversion.
    s_long, _ = _nees_neutral_inversion(3.030, 1e-6)
    assert abs(s_long - 1.0 / math.sqrt(2.0)) < 1e-3

    d = _report(1.0 / math.sqrt(2.0), 1e6)
    assert d["pit_ks_long"] < 0.25, (
        f"longitudinal KS reached {d['pit_ks_long']}, so a "
        "'both axes loud' gate may be satisfiable after all"
    )


def test_a_body_honest_tail_overconfident_covariance_is_caught():
    """The failure a trained model *actually* has, which is not an inversion.

    Measured on a trained checkpoint: 99.1% of frames report
    `sigma_long < sigma_lat` and 59.9% of frames genuinely have
    `|e_long| < |e_lat|`, so the orientation is right. What is wrong is the
    tail. Along-track error is 0.211 m at the 99th percentile and 4.049 m at
    the 99.9th, and the reported sigma tracks the first and not the second --
    on the worst 1% of frames the error averaged 2.249 m against a claimed
    0.308 m.

    Every median statistic calls that model honest, and it *is* honest, on 99%
    of frames. So this fixture is built the same way: a clean body with a thin,
    violent tail that the covariance does not widen for.
    """
    torch.manual_seed(0)
    e = torch.randn(N, 3) * TRUE_SD
    # One frame in 500 aliases along track and lands 40x out. The covariance
    # is left untouched, which is exactly the fault being modelled.
    tail = torch.rand(N) < 0.002
    e[tail, 0] *= 40.0
    cov = torch.diag_embed(TRUE_SD.square()).expand(N, 3, 3).contiguous()
    calib = Calibration()
    calib.update(e.clone(), torch.zeros(N, 3), cov)
    d = calib.as_dict()

    # The body statistics are fooled -- they are describing the 99.8% that is
    # fine, and they are right about it.
    assert d["pit_ks_long"] < 0.05
    assert 0.95 < d["tightness_long"] < 1.05
    assert 0.8 < d["anees_median"] / Calibration.CHI2_MEDIAN < 1.25

    # The tail statistic is not.
    assert d["tail_z_long"] > 2.0, d["tail_z_long"]
    # And it stays quiet on the axes that were never touched, so the report
    # says which axis to repair rather than merely that something is wrong.
    assert d["tail_z_lat"] < 1.5
    assert d["tail_z_yaw"] < 1.5
