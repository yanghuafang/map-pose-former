"""The pose solver, which has no parameters and so can be checked exactly.

Every other part of this model is judged by whether its error is small. This
one has a right answer: given correspondences that came from a known pose, it
must return that pose to float precision. A tolerance here would be hiding
something.
"""

import math

import pytest
import torch
from torch import Tensor

from mapposeformer import geometry as G
from mapposeformer.solve import (
    MIN_RESIDUAL_M2,
    curvature_covariance,
    point_normals,
    projections,
    solve_pose,
    solve_pose_directional,
    solve_pose_irls,
)


def _correspondences(pose, n=40, seed=0):
    """Map points, and the detections that ``pose`` carries onto them.

    ``pose`` is the correction the solver should recover: detections live in
    the ego frame, the map in the prior frame, and applying the correction to
    the former lands it on the latter.
    """
    gen = torch.Generator().manual_seed(seed)
    det = torch.rand(1, n, 2, generator=gen) * 60 - 30
    mp = G.transform_points(pose, det)
    assign = torch.eye(n).unsqueeze(0)
    return det, mp, assign


def test_exact_correspondences_recover_the_pose_exactly():
    for xy_yaw in ([0.0, 0.0, 0.0], [1.5, -0.6, 0.02], [-4.0, 2.0, -0.05]):
        pose = torch.tensor([xy_yaw])
        det, mp, assign = _correspondences(pose)
        got, mass = solve_pose(det, mp, assign)
        assert torch.allclose(got, pose, atol=1e-5), f"{got} != {pose}"
        assert math.isclose(mass.item(), 40.0)


def test_duplicating_a_correspondence_is_the_same_as_weighting_it():
    """The assignment is evidence, not a permutation.

    Nothing requires the rows to sum to one, so the solver has to treat a
    weight of two exactly as it treats the same pair listed twice. This is
    what lets a soft assignment mean anything.
    """
    pose = torch.tensor([[1.0, -0.5, 0.03]])
    det, mp, assign = _correspondences(pose, n=12)
    doubled = assign.clone()
    doubled[0, 3, 3] = 2.0
    a, _ = solve_pose(det, mp, doubled)

    det2 = torch.cat([det, det[:, 3:4]], dim=1)
    mp2 = torch.cat([mp, mp[:, 3:4]], dim=1)
    b, _ = solve_pose(det2, mp2, torch.eye(13).unsqueeze(0))
    assert torch.allclose(a, b, atol=1e-5)


def test_a_frame_with_no_evidence_returns_identity_and_no_mass():
    """0/0 is a NaN, and a NaN reaching the filter is a diverged scene.

    The solver cannot decide whether an empty frame should be refused -- that
    is a decision about evidence, taken upstream on a real threshold. What it
    must do is fail loudly in the output it does return.
    """
    pose = torch.tensor([[1.0, 0.5, 0.01]])
    det, mp, assign = _correspondences(pose, n=8)
    got, mass = solve_pose(det, mp, torch.zeros_like(assign))
    assert torch.equal(got, torch.zeros(1, 3))
    assert mass.item() == 0.0


def test_one_gross_outlier_drags_the_least_squares_solve():
    """The motivation for IRLS, asserted rather than assumed.

    If this ever stops being true the robust path is dead weight, so the test
    that justifies it is the one that shows the plain solve failing.
    """
    pose = torch.tensor([[1.2, -0.4, 0.02]])
    det, mp, assign = _correspondences(pose, n=30)
    mp[0, 7] += torch.tensor([25.0, -18.0])  # one confidently wrong match

    plain, _ = solve_pose(det, mp, assign)
    robust, _ = solve_pose_irls(det, mp, assign, iters=3, sigma_m=1.0)

    plain_err = (plain[0, :2] - pose[0, :2]).norm()
    robust_err = (robust[0, :2] - pose[0, :2]).norm()
    assert plain_err > 0.5, f"outlier did not bite: {plain_err}"
    assert robust_err < 0.02, f"IRLS did not reject it: {robust_err}"


def test_irls_leaves_clean_correspondences_alone():
    """Robustness must not cost accuracy when there is nothing to reject."""
    pose = torch.tensor([[0.8, 0.3, -0.01]])
    det, mp, assign = _correspondences(pose, n=30)
    got, _ = solve_pose_irls(det, mp, assign, iters=3)
    assert torch.allclose(got, pose, atol=1e-4)


def test_the_solve_is_batched_and_independent():
    """One batch element's correspondences must not move another's pose."""
    poses = torch.tensor([[1.0, 0.0, 0.0], [-2.0, 1.0, 0.04]])
    dets, mps = [], []
    for i, p in enumerate(poses):
        d, m, _ = _correspondences(p.unsqueeze(0), n=16, seed=i)
        dets.append(d)
        mps.append(m)
    det, mp = torch.cat(dets), torch.cat(mps)
    assign = torch.eye(16).unsqueeze(0).expand(2, -1, -1)
    got, _ = solve_pose(det, mp, assign)
    assert torch.allclose(got, poses, atol=1e-5)


def test_gradients_reach_the_assignment():
    """The solver is the only path from the matcher to the pose loss.

    Nothing downstream of here has parameters, so if this gradient is missing
    the matcher trains on nothing at all.
    """
    pose = torch.tensor([[1.0, -0.5, 0.02]])
    det, mp, assign = _correspondences(pose, n=10)
    assign = (assign + 0.05).requires_grad_(True)
    got, _ = solve_pose(det, mp, assign)
    got.norm().backward()
    assert assign.grad is not None
    assert torch.isfinite(assign.grad).all()
    assert assign.grad.abs().sum() > 0


def test_autocast_does_not_reach_the_coordinate_arithmetic():
    """``geometry.exact_arithmetic`` guards this solve, so bf16 must not bite.

    bf16 keeps 8 mantissa bits, so a 40 m coordinate lands on a 0.25 m
    lattice -- and this function multiplies weights by coordinates twice, in
    the centroids and in the cross-covariance. CPU autocast is off by default,
    so a test that cares has to ask for bf16 explicitly, which is what this is.
    """
    pose = torch.tensor([[1.5, -0.6, 0.02]])
    det, mp, assign = _correspondences(pose)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        got, _ = solve_pose(det, mp, assign)
    assert got.dtype is torch.float32, "the fp32 island leaked"
    assert torch.allclose(got, pose, atol=1e-5)


# --- point-to-line ---------------------------------------------------------
# The closed form above reports the same uncertainty shape whatever the
# landmarks are. These check that the directional solve does not.


def _road(n_poles=0, span=60.0, per_lane=20):
    """Three lane lines along x, plus optional roadside poles."""
    xs = torch.linspace(-span / 2, span / 2, per_lane)
    lanes = [
        torch.stack([xs, torch.full_like(xs, y)], -1) for y in (-3.5, 0.0, 3.5)
    ]
    pts = torch.stack(lanes).unsqueeze(0)
    pmask = torch.ones(1, 3, per_lane, dtype=torch.bool)
    is_line = torch.ones(1, 3, dtype=torch.bool)
    if n_poles:
        gen = torch.Generator().manual_seed(0)
        px = torch.rand(n_poles, generator=gen) * span - span / 2
        py = torch.where(torch.rand(n_poles, generator=gen) > 0.5, 8.0, -8.0)
        pole = torch.zeros(1, n_poles, per_lane, 2)
        pole[0, :, 0] = torch.stack([px, py], -1)
        pmask_p = torch.zeros(1, n_poles, per_lane, dtype=torch.bool)
        pmask_p[0, :, 0] = True
        pts = torch.cat([pts, pole], 1)
        pmask = torch.cat([pmask, pmask_p], 1)
        is_line = torch.cat(
            [is_line, torch.zeros(1, n_poles, dtype=torch.bool)], 1
        )
    return pts, pmask, is_line


def _directional(pts, pmask, is_line, delta, noise=0.0, seed=0):
    """Detections that `delta` lands on the map, matched point to point."""

    flat = pts.reshape(1, -1, 2)
    det = G.transform_points(G.inverse(delta), flat)
    if noise:
        gen = torch.Generator().manual_seed(seed)
        det = det + noise * torch.randn(det.shape, generator=gen)
    n = flat.shape[1]
    assign = (torch.eye(n) * pmask.reshape(1, -1)).unsqueeze(0)
    proj = projections(point_normals(pts, pmask), is_line)
    return solve_pose_directional(det, flat, assign, proj, iters=8)


def test_a_lane_line_normal_is_perpendicular_to_the_line():

    pts, pmask, _ = _road()
    n = point_normals(pts, pmask)
    # Lines run along x, so every normal is +/- y.
    assert torch.allclose(n[0, :, :, 0].abs(), torch.zeros(3, 20), atol=1e-5)
    assert torch.allclose(n[0, :, :, 1].abs(), torch.ones(3, 20), atol=1e-5)


def test_the_directional_solve_recovers_the_pose_when_the_road_allows_it():
    """With poles present, along-track is observable and the answer is exact."""
    pts, pmask, is_line = _road(n_poles=8)
    for xy_yaw in ([0.0, 0.0, 0.0], [0.4, -0.3, 0.01], [-0.6, 0.2, -0.015]):
        delta = torch.tensor([xy_yaw])
        got, mass, _, _, _ = _directional(pts, pmask, is_line, delta)
        assert mass.item() > 0
        assert torch.allclose(got, delta, atol=1e-3), f"{got} != {delta}"


def test_lane_lines_alone_leave_along_track_unobservable():
    """The result point-to-point cannot express, and this is the point of M3.

    Three parallel lines pin lateral offset and heading and say nothing about
    position along the road. A rank-deficient Hessian is the honest report;
    the closed form's `2 * mass * I` is not.
    """
    pts, pmask, is_line = _road(n_poles=0)
    _, _, hess, _, _ = _directional(pts, pmask, is_line, torch.zeros(1, 3))
    ev = torch.linalg.eigvalsh(hess[0])
    assert ev[0] / ev[-1] < 1e-9, f"expected a null direction, got {ev}"

    # And the null direction is *along the road*, not some other axis.
    null = torch.linalg.eigh(hess[0]).eigenvectors[:, 0]
    assert null[:2].abs().argmax().item() == 0, null


def test_adding_poles_makes_the_road_observable_again():
    """Along-track uncertainty should fall as along-track evidence arrives."""
    ratios = []
    for n_poles in (2, 8):
        pts, pmask, is_line = _road(n_poles=n_poles)
        _, _, hess, _, _ = _directional(pts, pmask, is_line, torch.zeros(1, 3))
        cov = torch.linalg.inv(hess[0])
        ratios.append((cov[0, 0] / cov[1, 1]).sqrt().item())
    assert ratios[0] > ratios[1] > 1.0, ratios


def test_point_to_point_reports_the_same_shape_whatever_the_road_is():
    """The contrast, measured rather than asserted.

    Identity projections make the directional solve point-to-point, and then
    the uncertainty shape stops depending on the landmarks entirely -- eight
    poles added to a bare road do not move it.
    """
    from mapposeformer.solve import solve_pose_directional

    shapes = []
    for n_poles in (0, 8):
        pts, pmask, _ = _road(n_poles=n_poles)
        flat = pts.reshape(1, -1, 2)
        n = flat.shape[1]
        assign = (torch.eye(n) * pmask.reshape(1, -1)).unsqueeze(0)
        eye = torch.eye(2).expand(1, pts.shape[1], pts.shape[2], 2, 2)
        _, _, hess, _, _ = solve_pose_directional(
            flat, flat, assign, eye, iters=3
        )
        cov = torch.linalg.inv(hess[0])
        shapes.append((cov[0, 0] / cov[1, 1]).sqrt().item())
    assert abs(shapes[0] - shapes[1]) < 0.02, shapes
    assert all(abs(s - 1.0) < 0.05 for s in shapes), shapes


def test_the_covariance_shrinks_as_evidence_arrives():
    """The 1/n every estimator is supposed to have.

    Easy to lose: the cost is a sum over correspondences and the Hessian grows
    with the mass too, so using the raw sum makes the two cancel and the
    covariance stops depending on how much evidence there was.
    """

    # With exact correspondences there is no residual to average down, and
    # the covariance is correctly zero however many points there are. The
    # question only means anything once the detections are noisy.
    # Poles scale with the lane points. Holding them fixed would hold the
    # along-track information fixed too, and along track dominates the sum --
    # the covariance would then fall by 1.9x where the evidence rose by 4x,
    # which is right but measures the pole count rather than the 1/n.
    sizes = []
    for per_lane, poles in ((10, 8), (40, 32)):
        pts, pmask, is_line = _road(n_poles=poles, per_lane=per_lane)
        _, _mass, hess, cost, dof = _directional(
            pts, pmask, is_line, torch.zeros(1, 3), noise=0.2
        )
        cov = curvature_covariance(hess, cost, dof)
        sizes.append(cov.diagonal(dim1=-2, dim2=-1)[0, :2].sum().item())
    assert sizes[1] < sizes[0] / 3, sizes


def test_a_worse_fit_reports_a_wider_covariance():
    """Nothing is fitted, so this has to come out of the residuals alone.

    Both noise levels sit above ``MIN_RESIDUAL_M2``, the detector's own
    per-point noise. Below that floor the covariance stops reading the
    residuals, on purpose: a fit that looks perfect is reporting less noise
    than the sensor has, and an information matrix built from it inverts to a
    negative-definite covariance in fp32.
    """

    pts, pmask, is_line = _road(n_poles=8)
    flat = pts.reshape(1, -1, 2)
    n = flat.shape[1]
    assign = (torch.eye(n) * pmask.reshape(1, -1)).unsqueeze(0)
    proj = projections(point_normals(pts, pmask), is_line)

    sizes = []
    for noise in (0.15, 0.6):
        gen = torch.Generator().manual_seed(0)
        det = flat + noise * torch.randn(flat.shape, generator=gen)
        _, _mass, hess, cost, dof = solve_pose_directional(
            det, flat, assign, proj, iters=8
        )
        cov = curvature_covariance(hess, cost, dof)
        sizes.append(cov.diagonal(dim1=-2, dim2=-1)[0, :2].sum().item())
    assert sizes[1] > 10 * sizes[0], sizes


def test_the_prior_is_what_makes_a_bare_road_reportable():
    """Lane lines alone give singular information; fusing the prior fixes it.

    And the result must still be elongated along the road -- a covariance that
    came back isotropic would mean the prior had erased the measurement rather
    than completed it.
    """

    pts, pmask, is_line = _road(n_poles=0)
    _, _mass, hess, cost, dof = _directional(
        pts, pmask, is_line, torch.zeros(1, 3)
    )
    prior = torch.diag(
        1.0 / torch.tensor([1.5, 0.6, torch.deg2rad(torch.tensor(1.0))]) ** 2
    ).unsqueeze(0)

    cov = curvature_covariance(hess, cost, dof, prior)
    assert torch.isfinite(cov).all()
    long_sigma = cov[0, 0, 0].sqrt()
    lat_sigma = cov[0, 1, 1].sqrt()
    # Along track the prior is all there is, so its 1.5 m survives intact.
    assert abs(long_sigma.item() - 1.5) < 0.05, long_sigma
    assert lat_sigma < long_sigma / 5, (long_sigma, lat_sigma)


def test_rejected_outliers_do_not_inflate_the_covariance():
    """The scale is ``cost / (dof - 3)``, and both have to see the same weights.

    IRLS rejects a gross correspondence, so the pose does not move. If the
    cost is built from the raw assignment while the dof and the Hessian are
    robust, that same rejected correspondence lands in the numerator and is
    removed from the denominator -- the uncertainty is inflated twice over for
    evidence the solve already decided to ignore.

    Measured on a synthetic road: eight gross outliers among sixty-eight
    correspondences left the pose error at 0.0083 m and grew reported
    sigma_lat from 0.0070 m to 0.3363 m, a factor of 48.
    """

    pts, pmask, is_line = _road(n_poles=8)
    flat = pts.reshape(1, -1, 2)
    n = flat.shape[1]
    proj = projections(point_normals(pts, pmask), is_line)
    gen = torch.Generator().manual_seed(0)
    clean = flat + 0.05 * torch.randn(flat.shape, generator=gen)
    assign = (torch.eye(n) * pmask.reshape(1, -1)).unsqueeze(0)

    sigmas, errors = [], []
    for bad in (0, 8):
        det = clean.clone()
        det[0, :bad] += torch.tensor([9.0, -7.0])
        pose, _mass, hess, cost, dof = solve_pose_directional(
            det, flat, assign, proj, iters=6, sigma_m=1.0
        )
        cov = curvature_covariance(hess, cost, dof)
        sigmas.append(float(cov[0, 1, 1].sqrt()))
        errors.append(float(pose[0, :2].norm()))

    # The pose is unharmed -- that is what the robust weight is for.
    assert abs(errors[1] - errors[0]) < 0.005, errors
    # So the reported uncertainty must not blow up either.
    assert sigmas[1] < 2.0 * sigmas[0], (
        f"outliers the solve rejected still inflated sigma: {sigmas}"
    )


#: The prior the model always supplies -- 1.5 m along, 0.6 m across, 1 deg --
#: as an information matrix. `curvature_covariance` is never called without
#: one outside these tests, and on a rank-deficient Hessian it is not a
#: refinement but the thing that makes the inverse exist at all.
_PRIOR = torch.diag(
    torch.tensor([1 / 1.5**2, 1 / 0.6**2, 1 / (math.pi / 180) ** 2])
).unsqueeze(0)


def _gave_up_frame() -> tuple[Tensor, Tensor, Tensor]:
    """The worst frame of the point-token model's validation split, as literals.

    Kept inline rather than as a checkpoint fixture so the numbers can be read
    and argued with. Recover more with ``tools/probe_cov.py --save``.

    @return ``(hessian, cost, dof)``. The recorded `mass` was 0.9999988 -- one
        surviving correspondence -- and that run used line residuals, which
        constrain one direction each, so the dof is that same 1.0. Below three
        it is clamped, which is the honest reading: three pose parameters
        cannot be fitted to one constrained direction.
    """
    hessian = torch.tensor(
        [
            [0.06770067662000656, -0.36168748140335083, -6.608574390411377],
            [-0.36168748140335083, 1.9322969913482666, 35.305973052978516],
            [-6.608574390411377, 35.305973052978516, 645.09326171875],
        ]
    ).unsqueeze(0)
    return hessian, torch.tensor([0.0]), torch.tensor([1.0])


def test_the_covariance_survives_a_solve_that_gave_up():
    """It goes to a Kalman filter, which factorises it.

    These numbers are not invented. They are the worst frame of 1 168 in the
    point-token checkpoint's validation split, recovered with
    ``tools/probe_cov.py`` -- the frame whose pose diverged to 22 m. Without
    the residual floor, 38.0% of that checkpoint's frames produce a covariance
    with a negative eigenvalue, this one at -87.48.

    The cause is the robust weighting giving up, not a good fit. With the pose
    metres wrong Geman-McClure rejects nearly every correspondence: `mass`
    falls to 1.0 out of hundreds, the survivor fits exactly so `cost` is 0,
    and one correspondence constrains one direction -- leaving a Hessian that
    is all but rank 1. Such a Hessian has no inverse, and none of this is a
    numerical accident; the prior is what makes the problem well posed, which
    is why `curvature_covariance` is never called in anger without one.
    """

    hessian, cost, dof = _gave_up_frame()
    # The Hessian really is near-singular: one correspondence, one direction.
    eig = torch.linalg.eigvalsh(hessian.double())
    assert eig[0, 0] / eig[0, -1] < 1e-6, eig

    cov = curvature_covariance(hessian, cost, dof, _PRIOR)
    assert torch.isfinite(cov).all(), cov
    # What the filter does, and what an indefinite covariance cannot survive.
    torch.linalg.cholesky(cov)
    assert (torch.linalg.eigvalsh(cov.double()) > 0).all(), cov

    # And it must be symmetric, because a filter will not check: any asymmetry
    # propagates into the state through `gain @ r @ gain.T`.
    assert torch.allclose(cov, cov.transpose(-1, -2), atol=0), cov


def test_the_residual_floor_is_what_keeps_the_prior_alive():
    """The floor is not an epsilon, and calling it one is how it breaks.

    Measured on the frame above, floor and dtype do not contribute equally --
    the floor does all of it:

    | floor | dtype | min eigenvalue |
    |---|---|---|
    | 1e-8 | float32 | -8.748e+01 |
    | 1e-8 | float64 | -6.213e-10 |
    | 0.12^2 | float32 | +3.885e-05 |

    Float64 alone does not save it. The reason is scale separation: at a 1e-8
    floor the information matrix reaches 3.2e10 while the prior contributes
    3.3e3, seven orders below, so **813 units of the prior are rounded away**
    and the regulariser meant to make a singular Hessian invertible simply
    stops being there. At 0.12^2 the two are 2.2e4 against 3.3e3 and 2.4e-4 is
    lost. A floor small enough to look harmless is exactly a floor large
    enough to delete the prior.
    """

    hessian, cost, dof = _gave_up_frame()
    scale_floor = float(MIN_RESIDUAL_M2)
    raw = float(cost) / max(float(dof) - 3.0, 1.0)

    def information(floor: float) -> torch.Tensor:
        return hessian.float() / (2.0 * max(raw, floor)) + _PRIOR.float()

    # How much of the prior survives being added to the measurement term.
    for floor, want_alive in ((1e-8, False), (scale_floor, True)):
        info = information(floor)
        recovered = info - hessian.float() / (2.0 * max(raw, floor))
        lost = float((recovered - _PRIOR.float()).abs().max())
        alive = lost < 0.01 * float(_PRIOR.max())
        assert alive is want_alive, (floor, lost)


def test_a_healthy_solve_is_left_alone_by_the_floor():
    """The floor must not move a covariance a trained model produces.

    Measured over 1 168 validation frames, the *smallest* residual scale a
    trained model produces is 5.01e-2 m^2 against a floor of 1.44e-2 -- a
    factor of 3.5, and the floor binds on 0.00% of frames. Three trained
    checkpoints each report the same worst eigenvalue with the floor and
    without it, to every digit printed, so every number in `RESULTS.md` is a
    number the floor did not move.
    """

    pts, pmask, is_line = _road(n_poles=8)
    flat = pts.reshape(1, -1, 2)
    n = flat.shape[1]
    proj = projections(point_normals(pts, pmask), is_line)
    assign = (torch.eye(n) * pmask.reshape(1, -1)).unsqueeze(0)

    # 0.25 m of noise puts the residual scale where real frames sit. Five
    # centimetres does not: it lands at 3.0e-3, *under* the floor, which is
    # worth knowing -- the floor is inert on this data, not inert in general.
    gen = torch.Generator().manual_seed(0)
    det = flat + 0.25 * torch.randn(flat.shape, generator=gen)
    _, _mass, hess, cost, dof = solve_pose_directional(
        det, flat, assign, proj, iters=6, sigma_m=1.0
    )
    scale = cost / (dof - 3.0).clamp_min(1.0)
    assert float(scale) > MIN_RESIDUAL_M2, float(scale)

    cov = curvature_covariance(hess, cost, dof)
    unfloored = torch.linalg.inv(
        hess.double() / (2.0 * scale.double()).view(-1, 1, 1)
    )
    assert torch.allclose(cov.double(), unfloored, rtol=1e-6, atol=1e-12), (
        cov,
        unfloored,
    )


def test_dof_counts_constrained_directions_not_correspondences():
    """Why the variance divisor is not `mass`.

    A point on a polyline is pulled perpendicular to it and is free to slide
    along it, so it constrains **one** direction. A pole constrains two. Both
    are worth exactly one correspondence, so `mass` cannot tell them apart --
    which is what makes `s^2 = cost / mass` wrong.

    The consequence is that the two residual modes' calibrations cannot be
    compared. Point-to-point makes every projection the identity, so its true
    dof is twice its correspondence count, and dividing by `mass` inflates its
    variance estimate by about two -- widening its covariance and depressing
    its NEES against a mode whose dof and mass happen to agree.
    """

    # Three lane lines: every residual is rank 1, so dof == mass.
    pts, pmask, is_line = _road(n_poles=0)
    _, mass, _, _, dof = _directional(pts, pmask, is_line, torch.zeros(1, 3))
    assert float(dof) == pytest.approx(float(mass), rel=1e-4), (dof, mass)

    # The same road under point-to-point residuals: every projection is the
    # identity, so each correspondence is worth two directions.
    flat = pts.reshape(1, -1, 2)
    n = flat.shape[1]
    assign = (torch.eye(n) * pmask.reshape(1, -1)).unsqueeze(0)
    eye_proj = (
        torch.eye(2).expand(1, pts.shape[1], pts.shape[2], 2, 2).contiguous()
    )
    _, mass_p, _, _, dof_p = solve_pose_directional(
        flat, flat, assign, eye_proj, iters=3
    )
    assert float(dof_p) == pytest.approx(2 * float(mass_p), rel=1e-4), (
        dof_p,
        mass_p,
    )

    # And with poles added to a line road the dof lands strictly between the
    # correspondence count and twice it.
    pts, pmask, is_line = _road(n_poles=8)
    _, mass_m, _, _, dof_m = _directional(
        pts, pmask, is_line, torch.zeros(1, 3)
    )
    assert float(mass_m) < float(dof_m) < 2 * float(mass_m), (dof_m, mass_m)
