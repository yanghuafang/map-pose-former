"""The pose, given the correspondences: weighted Procrustes, then refinement.

This is the part of the model with no parameters. Once something has decided
how much each detected point belongs to each map point, the pose that best
explains those correspondences is not learned -- it is the minimiser of

    E(R, t) = Σ_ij a_ij ‖R dᵢ + t − mⱼ‖²

and that minimiser has a formula. Solving it rather than regressing it buys
three things worth more than the flexibility it gives up: it cannot overfit,
it quantizes exactly because there are no weights to quantize, and the pose
becomes the answer to a stated question rather than the output of a black box.
When it is wrong, the assignment is wrong, and that is a thing you can look at.

**The derivation**, because the code below is unreadable without it. Write
`w = Σ a_ij` and put the weighted centroids at

    d̄ = (Σ a_ij dᵢ) / w      m̄ = (Σ a_ij mⱼ) / w

Setting ∂E/∂t = 0 gives `t = m̄ − R d̄` for any R, so translation is whatever
carries one centroid onto the other and only the rotation is left. Substituting
it back and dropping the terms R cannot change leaves

    maximise  tr(R H),   H = Σ a_ij (dᵢ − d̄)(mⱼ − m̄)ᵀ

a 2×2 matrix. With R = [[c, −s], [s, c]], tr(R H) = c(H₀₀ + H₁₁) + s(H₀₁ − H₁₀),
which is a single sinusoid in θ, maximised at

    θ = atan2(H₀₁ − H₁₀, H₀₀ + H₁₁)

No SVD, no iteration, no sign ambiguity to repair -- the 3-D Procrustes solve
needs all three and SE(2) needs none of them.

**H costs one matmul, not a loop.** Σ_ij a_ij dᵢ mⱼᵀ is `Dᵀ A M`, which
contracts (2×N_d)(N_d×N_m)(N_m×2) down to 2×2. The full assignment is never
materialised as residuals unless the robust reweighting asks for them.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from mapposeformer import geometry as G

# : Mass below which a frame has no evidence and the solve is not attempted.
# : Not a tuning knob -- it is the divide-by-zero guard. A frame this empty is
# : refused upstream on a real threshold, which is a decision about evidence
# : rather than about arithmetic.
MIN_MASS = 1e-6

#: The smallest mean squared residual this data can produce: the detector's own
#: per-point noise of 0.12 m, squared. It is the floor on the covariance's
#: scale, because a floor of 1e-8 m^2 -- a tenth of a millimetre -- makes the
#: information matrix about 1e11 and its fp32 inverse comes back negative
#: definite on 0.5% of frames.
MIN_RESIDUAL_M2 = 0.12**2


def solve_pose(
    det_pts: Tensor,
    map_pts: Tensor,
    assign: Tensor,
) -> tuple[Tensor, Tensor]:
    """The pose minimising the weighted point-to-point cost, in closed form.

    @param det_pts ``(B, D, 2)`` detected points, in the ego frame.
    @param map_pts ``(B, M, 2)`` map points, in the prior frame.
    @param assign ``(B, D, M)`` non-negative correspondence weights. Rows need
        not sum to one: the total is the evidence, and a row of zeros is a
        detection that matched nothing.

    @return ``(pose, mass)``. ``pose`` is ``(B, 3)`` as ``(x, y, yaw)``, the
        correction carrying detections onto the map. ``mass`` is ``(B,)``, the
        summed assignment -- how much evidence the pose rests on. A batch
        element with no mass gets the identity pose, so the caller sees a
        refusal rather than a NaN.
    """
    with G.exact_arithmetic(det_pts.device.type):
        det, mp, a = det_pts.float(), map_pts.float(), assign.float()

        mass = a.sum(dim=(1, 2))
        safe = mass.clamp_min(MIN_MASS).unsqueeze(-1)

        # Centroids, each weighted by how much of the assignment touches it.
        det_bar = torch.einsum("bdm,bdk->bk", a, det) / safe
        map_bar = torch.einsum("bdm,bmk->bk", a, mp) / safe

        # The 2x2 cross-covariance, expanded so that centring never
        # materialises D*M copies of the points: the cross term of the
        # expansion is exactly mass * det_bar * map_bar^T.
        cross = torch.einsum("bdk,bdm,bml->bkl", det, a, mp)
        H = cross - mass.view(-1, 1, 1) * det_bar.unsqueeze(
            2
        ) * map_bar.unsqueeze(1)

        yaw = torch.atan2(H[:, 0, 1] - H[:, 1, 0], H[:, 0, 0] + H[:, 1, 1])

        # Translation carries one centroid onto the other, and nothing more.
        rot_bar = G.transform_points(
            torch.stack(
                [torch.zeros_like(yaw), torch.zeros_like(yaw), yaw], -1
            ),
            det_bar.unsqueeze(1),
        ).squeeze(1)
        xy = map_bar - rot_bar

        pose = torch.cat([xy, yaw.unsqueeze(-1)], dim=-1)
        # No evidence, no claim. Identity rather than whatever 0/0 produced.
        return torch.where((mass > MIN_MASS).unsqueeze(-1), pose, 0.0), mass


def solve_pose_irls(
    det_pts: Tensor,
    map_pts: Tensor,
    assign: Tensor,
    iters: int = 3,
    sigma_m: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """``solve_pose`` again, distrusting what it could not fit.

    A least-squares solve has no defence against a confident wrong match: one
    correspondence off by 10 m contributes a hundred times what a correct one
    at 1 m does, so a single outlier drags the pose. Iteratively reweighted
    least squares fixes this the cheap way -- solve, look at the residuals,
    shrink the weight on whatever did not fit, and solve again.

    The weight is Geman-McClure's, ``(σ²/(σ² + r²))²``, normalised to 1 at zero
    residual. It is *redescending*: a correspondence far enough out contributes
    almost nothing rather than merely less, which is what a wrong match
    deserves. The price is that the first solve has to be close enough for the
    residuals to mean something, which here it is -- the prior is metres out,
    not tens of metres.

    @param iters Reweighting passes after the first solve. Three is where the
        pose stops moving on this data; the cost is one 2×2 solve each.
    @param sigma_m Residual scale in metres: the distance at which a
        correspondence has lost half its weight is ``σ·√(√2 − 1)``.

    @return As ``solve_pose``.
    """
    pose, mass = solve_pose(det_pts, map_pts, assign)
    if iters <= 0:
        return pose, mass

    with G.exact_arithmetic(det_pts.device.type):
        det, mp, a0 = det_pts.float(), map_pts.float(), assign.float()
        s2 = sigma_m * sigma_m
        weights = a0
        for _ in range(iters):
            moved = G.transform_points(pose, det)  # (B, D, 2)
            r2 = torch.cdist(moved, mp).square()  # (B, D, M)
            weights = a0 * (s2 / (s2 + r2)).square()
            pose, mass = solve_pose(det, mp, weights)
    return pose, mass


def point_normals(pts: Tensor, pmask: Tensor) -> Tensor:
    """Unit normal to the polyline at each of its points.

    The tangent is the central difference between a point's neighbours, which
    is the direction the line runs *there* rather than the direction the
    element runs on average -- a 12 m chunk of a curve is not straight, and
    the residual below is only as good as this.

    @param pts ``(B, M, P, 2)``, @param pmask ``(B, M, P)``.

    @return ``(B, M, P, 2)``, zero where the point is padding.
    """
    nxt = torch.roll(pts, -1, dims=2)
    prv = torch.roll(pts, 1, dims=2)
    # The ends have one neighbour, so they take the one-sided difference.
    nxt = torch.where(torch.roll(pmask, -1, 2).unsqueeze(-1), nxt, pts)
    prv = torch.where(torch.roll(pmask, 1, 2).unsqueeze(-1), prv, pts)
    tangent = nxt - prv
    length = tangent.norm(dim=-1, keepdim=True)
    # A degenerate tangent gets +x, chosen before the divide rather than
    # patched after it: 0/0 is NaN in the backward pass even where `where`
    # discards it.
    safe = torch.where(length > 1e-6, tangent, tangent.new_tensor([1.0, 0.0]))
    unit = safe / safe.norm(dim=-1, keepdim=True)
    normal = torch.stack([-unit[..., 1], unit[..., 0]], dim=-1)
    return normal * pmask.unsqueeze(-1)


def projections(normals: Tensor, is_line: Tensor) -> Tensor:
    """One 2x2 matrix per map point saying which directions it constrains.

    A pole constrains both axes, so its matrix is the identity. A point on a
    lane line constrains only the perpendicular offset -- sliding along the
    line costs nothing -- so its matrix is ``n n^T``, rank one.

    This is the whole of the point-to-line change. The solve below contracts
    against these matrices and never asks which kind of landmark it has, and
    point-to-point is exactly the case where every matrix is the identity.

    @param normals ``(B, M, P, 2)``, @param is_line ``(B, M)``.

    @return ``(B, M, P, 2, 2)``.
    """
    outer = normals.unsqueeze(-1) * normals.unsqueeze(-2)
    eye = torch.eye(2, device=normals.device, dtype=normals.dtype)
    return torch.where(is_line[:, :, None, None, None], outer, eye)


def _cost_at(pose, det, mp, a, proj_m, proj_d, target, weight) -> Tensor:
    """The weighted sum of squared residuals at ``pose``.

    **Weighted by the same robust weights the pose was solved with.** The
    scale of the covariance is ``cost / (dof - 3)``, and the dof and the Hessian
    are both robust, so a raw cost puts a rejected outlier in the numerator
    while removing it from the denominator -- inflating the uncertainty twice
    over for a correspondence the solve already decided to ignore. Measured on
    a synthetic road: eight gross outliers among sixty-eight correspondences
    left the pose error at 0.0083 m and grew the reported sigma_lat from
    0.0070 m to 0.3363 m, a factor of 48.

    Expanded rather than formed. The direct sum wants every (detection, map
    point) difference, which is a hundred megabytes at a full batch, and the
    three terms below are each O(D) or O(M) given quantities the solve has.
    """
    moved = G.transform_points(pose, det)
    quad = torch.einsum(
        "bdk,bdkl,bdl->b", moved, proj_d * weight[..., None, None], moved
    )
    cross = torch.einsum("bdk,bdk->b", moved, target * weight[..., None])
    own = torch.einsum("bmk,bmkl,bml->bm", mp, proj_m, mp)
    own_d = torch.einsum("bdm,bm->bd", a, own)
    return quad - 2 * cross + torch.einsum("bd,bd->b", weight, own_d)


class Solve(NamedTuple):
    """What the directional solve worked out, and what each field is for.

    A tuple rather than four bare returns because the fourth and fifth are
    easy to confuse and mean very different things: ``mass`` counts
    *correspondences* and answers "was there any evidence", while ``dof``
    counts *constrained directions* and is the divisor a variance estimate
    needs.
    """

    pose: Tensor
    #: Robust-weighted correspondence count. The refusal gate reads this.
    mass: Tensor
    #: ``(B, 3, 3)`` curvature in ``(x, y, yaw)``, excluding the prior.
    hessian: Tensor
    #: Weighted sum of squared residuals at the solution -- a sum, not a mean.
    cost: Tensor
    #: Residual degrees of freedom. A point on a polyline is pulled in one
    #: direction and a pole in two, so a correspondence is worth 1 or 2 here
    #: while being worth exactly 1 in ``mass``.
    dof: Tensor


def solve_pose_directional(
    det_pts: Tensor,
    map_pts: Tensor,
    assign: Tensor,
    proj: Tensor,
    iters: int = 5,
    sigma_m: float = 1.0,
    prior_information: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Least squares against the projections, by Gauss-Newton.

    Point-to-point has a closed form because every residual constrains both
    axes equally. Point-to-line does not, so this linearises instead -- the
    same move point-to-plane ICP makes, and for the same reason. Rotation is
    the only nonlinearity, ``d(R p)/dtheta`` at the current estimate is the
    perpendicular ``(-y, x)``, and what is left is a 3x3 solve.

    The Hessian it accumulates is the thing M3 is really after. Under
    point-to-point it is ``2 * mass * I`` in translation whatever the
    landmarks are; under these projections a road of parallel lane lines
    produces a Hessian that is genuinely ill-conditioned along the road, which
    is what the errors have been saying all along.

    @param proj ``(B, M, P, 2, 2)`` from ``projections``.
    @param sigma_m Robust scale, as in ``solve_pose_irls``. The weight is
        applied per detected *point* -- each one is a residual -- and has to
        be annealed for the same reason: redescending weights reject
        everything while the pose is still bad.
    @param prior_information Optional ``(B, 3, 3)`` used to make each *step*
        solvable. A road with no along-track landmark is singular rather than
        merely weak, so something has to make it invertible, and the prior is
        the honest candidate -- it is information the system genuinely has.
        The returned Hessian deliberately excludes it: what comes out of here
        is what the *measurement* knows, and fusing is the caller's business.

    @return ``(pose, mass, hessian, cost)``. ``hessian`` is ``(B, 3, 3)`` in
        ``(x, y, yaw)``, the curvature the covariance is read from; ``cost``
        is ``(B,)``, the weighted sum of squared residuals at the solution --
        a *sum*, not a mean, so anything reading it has to divide by the
        returned ``dof``, less the three pose parameters that were fitted.
    """
    if iters < 1:
        # The Hessian is built inside the loop, so zero passes would return
        # it as exact zeros -- and `curvature_covariance` then inverts the
        # prior alone and hands that back as a measured covariance. There is
        # no "no refinement" setting here: the first pass *is* the least
        # squares against these projections, and the seed is point-to-point.
        raise ValueError(
            f"solve_pose_directional needs iters >= 1, got {iters}: the "
            "Hessian the covariance is read from is accumulated in the loop"
        )
    with G.exact_arithmetic(det_pts.device.type):
        det, mp, a = det_pts.float(), map_pts.float(), assign.float()
        B = det.shape[0]
        N = proj.float().reshape(B, -1, 2, 2)
        mass = a.sum(dim=(1, 2))

        # Per detected point: the projections its matches impose, and the
        # target those projections pull it towards.
        proj_d = torch.einsum("bdm,bmkl->bdkl", a, N)
        pulled = torch.einsum("bmkl,bml->bmk", N, mp)
        target = torch.einsum("bdm,bmk->bdk", a, pulled)

        own = torch.einsum("bmk,bmkl,bml->bm", mp, N, mp)
        own_d = torch.einsum("bdm,bm->bd", a, own)
        point_mass = a.sum(-1)
        s2 = sigma_m * sigma_m

        pose, _ = solve_pose(det, mp, a)
        hessian = torch.zeros(B, 3, 3, device=det.device, dtype=det.dtype)
        eye = torch.eye(3, device=det.device, dtype=det.dtype)
        weight = torch.ones_like(point_mass)
        for _ in range(iters):
            moved = G.transform_points(pose, det)

            # Geman-McClure on each detected point's own mean squared
            # residual. One point is one residual, so that is the granularity
            # the weight belongs at.
            quad_d = torch.einsum("bdk,bdkl,bdl->bd", moved, proj_d, moved)
            cross_d = torch.einsum("bdk,bdk->bd", moved, target)
            r2 = (quad_d - 2 * cross_d + own_d).clamp_min(
                0
            ) / point_mass.clamp_min(MIN_MASS)
            weight = (s2 / (s2 + r2)).square()
            wj = proj_d * weight[..., None, None]
            wt = target * weight[..., None]
            # J = [I, (R d) rotated a quarter turn]; 2x3 per detected point.
            jac = torch.zeros(B, det.shape[1], 2, 3, device=det.device)
            jac[..., 0, 0] = 1.0
            jac[..., 1, 1] = 1.0
            jac[..., 0, 2] = -moved[..., 1]
            jac[..., 1, 2] = moved[..., 0]

            # Twice the Gauss-Newton products, so `hessian` is the cost's
            # actual second derivative -- which is what the curvature
            # covariance `2 * c_min * H^-1` is defined against. The step is a
            # ratio of the two, so the factor cancels there.
            hessian = 2 * torch.einsum("bdki,bdkl,bdlj->bij", jac, wj, jac)
            resid = torch.einsum("bdkl,bdl->bdk", wj, moved) - wt
            grad = 2 * torch.einsum("bdki,bdk->bi", jac, resid)

            full = hessian
            if prior_information is not None:
                full = full + prior_information
            # Levenberg damping. Lane lines alone make this exactly singular,
            # and a solve against a singular matrix is an exception or a
            # garbage step depending on the backend. Damping turns the
            # unobservable direction into no movement along it, which is the
            # right answer: with no along-track evidence, do not move along
            # track.
            scale = (
                full.diagonal(dim1=-2, dim2=-1)
                .abs()
                .amax(-1)
                .clamp_min(1.0)
                .view(-1, 1, 1)
            )
            step = -torch.linalg.solve(
                full + 1e-6 * scale * eye, grad.unsqueeze(-1)
            ).squeeze(-1)

            # The perturbation is additive on (x, y, yaw), which is what the
            # Jacobian above differentiates, so the update is addition rather
            # than an SE(2) composition.
            pose = pose + step
            pose = torch.cat([pose[:, :2], G.wrap_angle(pose[:, 2:])], dim=-1)

        cost = _cost_at(pose, det, mp, a, N, proj_d, target, weight)
        # The evidence that survived reweighting, which is what a refusal
        # should read -- as in ``solve_pose_irls``.
        mass = (point_mass * weight).sum(-1)

        # Residual degrees of freedom, which is *not* the correspondence
        # count. Each projection's trace is the number of directions it
        # constrains -- 1 for a polyline's `n n^T`, 2 for a pole's identity --
        # and `proj_d` is already the assignment-weighted sum of them, so its
        # trace is the dof that detection point contributes.
        trace_d = torch.einsum("bdkk->bd", proj_d)
        dof = (trace_d * weight).sum(-1)

        alive = (mass > MIN_MASS).unsqueeze(-1)
        return Solve(
            torch.where(alive, pose, 0.0),
            mass,
            hessian,
            cost.clamp_min(0),
            dof,
        )


def measurement_information(
    hessian: Tensor, cost: Tensor, dof: Tensor
) -> Tensor:
    """What this frame's landmarks alone say about the pose, as information.

    ``H / 2 s^2`` with ``s^2 = cost / (dof - 3)``: the inverse of the covariance
    the measurement would have on its own. Information rather than covariance
    because **it is routinely singular and that is not an error**. Three lane
    dividers and no along-track landmark determine the lateral offset and the
    heading and say nothing whatever about position along the road, so the
    covariance does not exist while the information is perfectly well defined
    and simply has a zero eigenvalue pointing down the road.

    A filter should add this to its own information rather than be handed a
    covariance, for two reasons. It never has to invert a singular matrix --
    and the covariance it would otherwise be given has the *prior* already
    fused into it, which for a filter whose own state is that prior counts the
    same information twice.

    @return ``(B, 3, 3)`` in ``(x, y, yaw)``, float64, symmetric, PSD.
    """
    # A real measurement-noise floor, not an epsilon, and the floor is the
    # whole of the repair it was added for. Dividing a near-singular Hessian
    # by 1e-8 m^2 sends this matrix to about 3e10, and a prior contributing
    # 3e3 is then seven orders below it -- so adding the prior in float32
    # rounds 813 units of it away and the regulariser that made the problem
    # well posed stops being there. Measured on the worst frame of a
    # point-token checkpoint, float64 alone still returned a negative
    # eigenvalue (-6.2e-10); the floor alone fixed it even in float32
    # (+3.9e-05). 38.0% of that checkpoint's frames were affected; every
    # healthy checkpoint had none, and its smallest residual scale sits 3.5x
    # above this floor, so nothing a trained model produces is touched.
    # `dof - 3`, not `dof`: three pose parameters were fitted to these same
    # residuals, so a sum of squares divided by the raw count under-states the
    # variance. It is the ordinary reduced chi-square correction, and on this
    # data it is not a small one. The robust mass at convergence is 11.2 and
    # 12.2 on two seeds of a 4-layer model and 13.8 on a 2-layer one, and
    # `dof` lies between `mass` and `2 mass`, so subtracting 3 moves s^2 by
    # 12-37% -- against the 1.5% it would move at the hundreds of
    # correspondences a soft assignment never actually produces. Every other
    # converged run sits below a mass of 1, where `clamp_min(1.0)` takes the
    # divisor over outright and s^2 is just `cost`: a floor, not a chi-square
    # correction. Either way it is what makes the number a variance estimate
    # rather than a mean square.
    scale = (cost / (dof - 3.0).clamp_min(1.0)).clamp_min(MIN_RESIDUAL_M2)
    information = hessian.double() / (2.0 * scale.double()).view(-1, 1, 1)
    return 0.5 * (information + information.transpose(-1, -2))


def curvature_covariance(
    hessian: Tensor,
    cost: Tensor,
    dof: Tensor,
    prior_information: Tensor | None = None,
) -> Tensor:
    """The pose covariance, read off the curvature of the cost.

    Near the minimum the cost is ``c_min + d^T H d / 2``, so the pose is
    uncertain by however far ``d`` can move before the cost rises by the noise
    in the cost itself. Estimate that noise by how badly the correspondences
    fit -- the weighted mean squared residual ``s^2 = cost / (dof - 3)`` --
    and the covariance falls out as ``2 s^2 H^-1``. Nothing is fitted; there
    is no scale head and no temperature to tune.

    **Sum against mean is the trap here.** ``cost`` is a sum over
    correspondences and ``H`` grows with the mass too, so using the sum
    directly gives a covariance that does not shrink as evidence arrives --
    it cancels exactly. Dividing by the mass first restores the ``1/n`` every
    estimator is supposed to have.

    @param prior_information Optional ``(B, 3, 3)`` inverse prior covariance.
        Lane lines with no along-track landmark make the measurement
        information singular, so without this there is no covariance to
        report at all -- and the prior is not a regulariser here, it is
        information the system genuinely has.

    @return ``(B, 3, 3)`` covariance in ``(x, y, yaw)``.
    """
    information = measurement_information(hessian, cost, dof)
    if prior_information is not None:
        information = information + prior_information.double()

    # Solved in float64 and symmetrised. An information matrix built from a
    # sum of rank-1 outer products is symmetric in exact arithmetic and only
    # nearly so in floating point, and `inv` does not know that.
    information = 0.5 * (information + information.transpose(-1, -2))

    # Factorised on the CPU, deliberately. These are 3x3 matrices, so cuSOLVER
    # has nothing to offer on them -- and it has something to take: creating a
    # cuSOLVER handle needs workspace, and on a machine already running several
    # training jobs it fails outright with CUSOLVER_STATUS_INTERNAL_ERROR,
    # which took down the measurement of a whole experiment. A batch of 64 3x3
    # Choleskys is microseconds either way; the difference is that one of them
    # cannot fail because a neighbour is using the card.
    device = information.device
    info_cpu = information.cpu()
    eye = torch.eye(3, dtype=info_cpu.dtype).expand_as(info_cpu)
    cov = torch.cholesky_solve(eye, torch.linalg.cholesky(info_cpu))
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    return cov.to(device=device, dtype=hessian.dtype)
