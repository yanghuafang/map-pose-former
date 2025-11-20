"""The whole network: encode, match, solve.

    map elements ─┐
                  ├ encode ─ match ─ assign ─ Procrustes ─ delta
    detections ───┘

Three parts and one of them has no parameters. The encoder turns each polyline
or point landmark into a token in its own frame; the matcher decides which
detection is which map element and which point is which point; the solver
turns those correspondences into a pose by solving, not regressing. Everything
the network learns, it learns about *association*, because association is the
hard part -- given correct correspondences the pose was never in question.

**History is width, not depth.** A past frame's detections are warped into the
current ego frame through the odometry that measured the motion, and then
joined to this frame's detections as more elements. There is no recurrence and
no second pass: the matcher simply has three frames' worth of evidence to place
against one map, which is what makes an intermittent along-track landmark
useful after it has gone out of view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from mapposeformer import geometry as G
from mapposeformer.model.encoder import (
    ElementEncoder,
    PointEncoder,
    embeddings,
    local_points,
)
from mapposeformer.model.matcher import Matcher
from mapposeformer.solve import (
    curvature_covariance,
    measurement_information,
    point_normals,
    projections,
    solve_pose_directional,
)

#: Per-detection side channel: the detector's confidence and the two axes of
#: the noise it claims. The map has no equivalent -- it is surveyed.
DETECTION_EXTRAS = 3


@dataclass
class ModelParams:
    """The network's shape, and how the pose is extracted from it."""

    dim: int = 128
    #: Four, not two. Depth is the one axis where more is unambiguously
    #: better here: two layers costs 9.8 points of recall at 18 pooled seed
    #: standard deviations, the widest margin any structural comparison in
    #: this project produced.
    layers: int = 4
    heads: int = 2
    #: Width of each attention head. Stated, not derived, so the shipped
    #: shape reads off the defaults without arithmetic -- ``dim`` 128 over 2
    #: heads derives the same 64, but a reader should not have to work that
    #: out. Stating it also makes head *count* separable from head *width*: a
    #: derived width is 128, 64 and 32 for 1, 2 and 4 heads, so varying the
    #: head count would also vary the per-head score-matrix rank cap -- the
    #: very quantity the rank argument in ``ROADMAP.md`` is about.
    #:
    #: 0 still derives ``dim // heads``, for sweeps that want that.
    head_dim: int = 64
    #: Rotary frequency bands; 0 derives ``min(head_dim // 6, 5)``.
    #:
    #: Five is an *interior* optimum, not a ceiling reached by accident: 3
    #: scores 0.9400 and 10 scores 0.9280, against 5's 0.9585.
    #: ``RotaryFrames`` rotates ``6 * bands`` channels and passes the rest
    #: through, so positional resolution is paid for out of content capacity
    #: -- and at 5 the shortest wavelength is 1.875 m against a 1.714 m map
    #: point pitch, while at 10 it is 0.059 m, far below anything the map can
    #: resolve, with 60 of 64 channels spent reaching it.
    #:
    #: Stated rather than derived, for the same reason as ``head_dim``. 0
    #: still derives ``min(head_dim // 6, 5)``, which is what lets a model
    #: narrower than 30 channels a head be built at all.
    rope_bands: int = 5
    #: How the geometry enters the attention scores: ``relative`` (fully
    #: SE(2)-equivariant), ``rope`` (translation only), or ``absolute``.
    #:
    #: ``rope`` by default, and the equivariance it carries is load-bearing:
    #: ``absolute`` has to *learn* the invariance and loses 9.2 points of
    #: recall doing it (7.6 sd). At point resolution two points on parallel
    #: lane lines at the same station have identical content features, so
    #: their frames are the only thing telling them apart.
    geometry: str = "rope"
    #: ``line`` gives each polyline point a rank-1 residual constraining only
    #: its perpendicular offset, which is what lets the Hessian be
    #: anisotropic; ``point`` constrains both axes and reproduces the
    #: isotropic ``2*mass*I``. A flag rather than an edit, so the two can be
    #: run as configurations differing by exactly one thing.
    #:
    #: ``point`` by default, which is the opposite of what the anisotropy
    #: argument predicts and is what the measurement says. At point
    #: resolution ``line`` is the more *accurate* of the two and its
    #: covariance is unusable -- NEES 0.04 to 0.20 against an honest 0.789,
    #: and at four layers it runs 7.7x to 17.7x too wide with the major axis
    #: 65 to 79 degrees out. ``point`` holds 0.68 to 0.76. An honest
    #: covariance is what this model is for, so it buys the axis it can keep.
    residual: str = "point"
    #: ``element`` lets a detection's element pick one map element for all of
    #: its points; ``point`` lets each point pick its own. The map is chunked
    #: at a fixed 12 m and a frustum is not, so a detected road boundary spans
    #: four map elements and its points genuinely belong to different ones --
    #: which the element-level answer cannot express.
    #:
    #: **Only has effect when ``tokens`` is ``element``.** At ``tokens=point``
    #: every detected point already chooses its own map point, so there is no
    #: element-level answer to refine and this field is inert -- the forward
    #: pass returns before reading it. It is kept because it is the cheaper
    #: half of the resolution question: it buys per-point assignment while
    #: still pooling the tokens, and measuring it separates "pooling costs
    #: resolution" from "pooling costs capacity".
    point_assign: str = "element"
    #: ``element`` pools a polyline's points into one token and matches those;
    #: ``point`` matches at point resolution.
    #:
    #: Element tokens were adopted to make attention 64x cheaper. Measured,
    #: that saving is not the constraint -- the assignment is 768 x 576 either
    #: way, the model is loader-bound, and element tokens ran 10% *slower*.
    #: What pooling costs is resolution: 12.7 points of recall at 6.8 pooled
    #: seed standard deviations. ``point`` by default; ``element`` remains so
    #: the saving can be re-measured rather than taken on trust.
    tokens: str = "point"
    #: Geometry in the attention scores. ``relative`` needs an N x N bias
    #: tensor, which over point tokens measures 2.4x the step time for 2.5%
    #: accuracy, so point tokens default to ``rope``: the rotary encoding
    #: carries relative position with no such tensor. Left settable, because
    #: that trade depends on the shape it is measured at.
    point_geometry: str = "rope"
    #: IRLS passes after the first solve. Zero is the plain least squares.
    refine_iters: int = 3
    #: Residual scale the robust weight is measured against, in metres. This
    #: is the *final* value: see the note on annealing in the class docstring.
    robust_sigma_m: float = 1.0
    #: Where the anneal starts. The prior is wrong by up to 4.5 m along track,
    #: so a fresh model's residuals are metres and a 1 m sigma rejects all of
    #: them -- including the correct correspondences it has yet to learn.
    robust_sigma_start_m: float = 6.0
    #: The prior's own uncertainty, which the covariance is fused with. Three
    #: lane lines and nothing else leave the measurement information singular,
    #: so without this there is no covariance to report at all -- and it is
    #: not a regulariser, it is information the system genuinely has.
    #: ``config._validate`` checks these against the prior the data draws.
    prior_sigma_long_m: float = 1.5
    prior_sigma_lat_m: float = 0.6
    prior_sigma_yaw_deg: float = 1.0


class MapPoseFormer(nn.Module):
    """The network.

    **The robust weight has to be annealed, not switched on.** Geman-McClure
    is redescending: the property that makes it reject an outlier also makes
    it reject nearly everything while the pose is still bad. At initialisation
    the correspondences are random and the pose is metres out, so a 1 m sigma
    leaves 0.006 of the assignment mass where a 12 m sigma leaves 0.35 -- a
    60-fold cut, measured on an untrained model.

    The gradient survives that; it is the *evidence* that does not. What is
    left is whichever few correspondences happened to land close, so the pose
    is decided by an accident rather than by the match. Graduated
    non-convexity is the standard answer -- start wide enough to reject
    nothing, narrow as the pose becomes worth trusting -- and being a
    schedule it lives in the trainer, so ``forward`` takes the value rather
    than reading a constant.

    @param p See ``ModelParams``.
    """

    def __init__(self, p: ModelParams) -> None:
        super().__init__()
        self.p = p
        cls_embed, attr_embed = embeddings(p.dim)
        if p.tokens == "point":
            self.map_encoder = PointEncoder(p.dim, cls_embed, attr_embed)
            self.det_encoder = PointEncoder(
                p.dim, cls_embed, attr_embed, extra_dim=DETECTION_EXTRAS
            )
        elif p.tokens == "element":
            self.map_encoder = ElementEncoder(p.dim, cls_embed, attr_embed)
            self.det_encoder = ElementEncoder(
                p.dim, cls_embed, attr_embed, extra_dim=DETECTION_EXTRAS
            )
        else:
            raise ValueError(f"unknown tokens {p.tokens!r}")
        self.matcher = Matcher(
            p.dim,
            p.layers,
            p.heads,
            p.geometry if p.tokens == "element" else p.point_geometry,
            per_point=p.tokens == "element" and p.point_assign == "point",
            head_dim=p.head_dim,
            rope_bands=p.rope_bands,
        )
        sigmas = torch.tensor(
            [
                p.prior_sigma_long_m,
                p.prior_sigma_lat_m,
                math.radians(p.prior_sigma_yaw_deg),
            ]
        )
        self.register_buffer(
            "prior_information",
            torch.diag(1.0 / sigmas.square()),
            persistent=False,
        )

    def detections(self, b: dict[str, Tensor]) -> tuple[Tensor, ...]:
        """This frame's detections and the warped history, as one set.

        ``hist_rel`` is ``gt⁻¹ ∘ gt_past`` -- where the vehicle was, as it is
        now -- so it is exactly the transform that brings a past detection
        into the current ego frame. It carries the odometry's drift with it,
        which is the point: pretending the past arrived noiselessly would make
        history an oracle rather than evidence.
        """
        pts, pmask = [b["det_pts"]], [b["det_pmask"]]
        cls, attr = [b["det_cls"]], [b["det_attr"]]
        conf, sigma = [b["det_conf"]], [b["det_sigma"]]

        H = b["hist_pts"].shape[1]
        for k in range(H):
            rel = b["hist_rel"][:, k]
            hp = b["hist_pts"][:, k]
            B, D, P, _ = hp.shape
            pts.append(
                G.transform_points(rel, hp.reshape(B, D * P, 2)).view(
                    B, D, P, 2
                )
            )
            pmask.append(b["hist_pmask"][:, k])
            cls.append(b["hist_cls"][:, k])
            attr.append(b["hist_attr"][:, k])
            conf.append(b["hist_conf"][:, k])
            sigma.append(b["hist_sigma"][:, k])

        extra = torch.cat(
            [torch.cat(conf, 1).unsqueeze(-1), torch.cat(sigma, 1)], dim=-1
        )
        return (
            torch.cat(pts, 1),
            torch.cat(pmask, 1),
            torch.cat(cls, 1),
            torch.cat(attr, 1),
            extra,
        )

    def _oriented_for_residual(self, oriented: Tensor) -> Tensor:
        """Which elements get a rank-1 residual, under ``residual``.

        One helper because both token paths need it. Skipping it on the point
        path makes `residual=point` run line residuals while reporting itself
        as point-to-point -- an ablation that silently measures nothing.
        """
        if self.p.residual == "point":
            return torch.zeros_like(oriented)
        if self.p.residual != "line":
            raise ValueError(f"unknown residual {self.p.residual!r}")
        return oriented

    def forward(
        self, b: dict[str, Tensor], sigma_m: float | None = None
    ) -> dict[str, Tensor]:
        """@param sigma_m Robust scale for this step; see the class docstring.
            Defaults to the configured final value.

        @return ``delta`` and ``mass``, plus what the losses supervise.
            ``mass`` is measured *after* reweighting, so it is the evidence
            that survived -- which is what a refusal should be read from.
        """
        det_pts, det_pmask, det_cls, det_attr, extra = self.detections(b)

        map_e = self.map_encoder(
            b["map_pts"], b["map_pmask"], b["map_cls"], b["map_attr"]
        )
        det_e = self.det_encoder(det_pts, det_pmask, det_cls, det_attr, extra)

        elements, scores, sig_d, sig_m, map_tok, det_tok = (
            self.matcher.elements(map_e, det_e)
        )
        if self.p.tokens == "point":
            # Matching at point resolution already *is* the assignment the
            # solve wants -- (D*P, M*P) -- so there is no second stage, and
            # with it goes the defect that a detection spanning four map
            # chunks could not tell its points apart.
            # `residual` has to be consulted here too. Without it
            # `tokens=point, residual=point` runs line residuals and reports
            # itself as point-to-point, which would make the single most
            # important ablation in this design unmeasurable.
            oriented = map_e[2].view(*b["map_pts"].shape[:2], -1)[..., 0]
            oriented = self._oriented_for_residual(oriented)
            proj = projections(
                point_normals(b["map_pts"], b["map_pmask"]), oriented
            )
            prior = self.prior_information.expand(det_pts.shape[0], 3, 3)
            delta, mass, hessian, cost, dof = solve_pose_directional(
                det_pts.reshape(det_pts.shape[0], -1, 2),
                b["map_pts"].reshape(det_pts.shape[0], -1, 2),
                elements,
                proj,
                iters=self.p.refine_iters,
                sigma_m=self.p.robust_sigma_m if sigma_m is None else sigma_m,
                prior_information=prior,
            )
            return {
                "delta": delta,
                "mass": mass,
                "cov": curvature_covariance(hessian, cost, dof, prior),
                # The same matrix cannot serve both readers. `cov` is the
                # posterior -- prior fused in -- which is what a NEES against
                # the truth should be measured with, because the estimate used
                # the prior. A Kalman filter wants the opposite: its own state
                # already *is* the prior, so handing it a posterior counts the
                # prior twice. It gets the information instead, which is also
                # the form that survives a frame with no along-track landmark.
                "information": measurement_information(hessian, cost, dof),
                "hessian": hessian,
                "cost": cost,
                "dof": dof,
                "prior_information": prior,
                "granularity": "point",
                "elements": elements,
                "scores": scores,
                "det_matchable": sig_d,
                "map_matchable": sig_m,
                "det_pts": det_pts,
                "det_pmask": det_pmask,
            }
        spread: Tensor = elements
        if self.p.point_assign == "point":
            spread = self.matcher.point_elements(
                map_tok,
                det_tok,
                local_points(det_pts, det_pmask, det_e[1])[..., :2],
                det_pmask,
                map_e[3],
                sig_d,
                sig_m,
            )
        elif self.p.point_assign != "element":
            raise ValueError(f"unknown point_assign {self.p.point_assign!r}")
        assign = self.matcher.points(
            spread,
            det_pts,
            det_pmask,
            det_e[1],
            b["map_pts"],
            b["map_pmask"],
            map_e[1],
        )

        B = det_pts.shape[0]
        # Which map elements constrain both axes and which only one. An
        # element with a single point -- a pole, a sign -- pins position
        # outright; a polyline pins only the perpendicular offset, which is
        # what lets the Hessian be anisotropic at all. `oriented` decides it
        # geometrically rather than by a class list, so an element with one
        # surviving point is treated as the point it is.
        oriented = self._oriented_for_residual(map_e[2])
        proj = projections(
            point_normals(b["map_pts"], b["map_pmask"]), oriented
        )
        prior = self.prior_information.expand(B, 3, 3)
        delta, mass, hessian, cost, dof = solve_pose_directional(
            det_pts.reshape(B, -1, 2),
            b["map_pts"].reshape(B, -1, 2),
            assign,
            proj,
            iters=self.p.refine_iters,
            sigma_m=self.p.robust_sigma_m if sigma_m is None else sigma_m,
            prior_information=prior,
        )
        return {
            "delta": delta,
            "mass": mass,
            "cov": curvature_covariance(hessian, cost, dof, prior),
            #: Measurement only, in information form; see the point path.
            "information": measurement_information(hessian, cost, dof),
            "hessian": hessian,
            "cost": cost,
            "dof": dof,
            # Stated, not implied. The loss picks its label scheme on this key,
            # and omitting it here would work only because a missing key falls
            # through to the element branch. Any other reader -- a dof logger,
            # a calibration probe -- would read the element path as unlabelled
            # and silently take a default.
            "granularity": "element",
            "prior_information": prior,
            "elements": elements,
            "scores": scores,
            "det_matchable": sig_d,
            "map_matchable": sig_m,
            "det_pts": det_pts,
            "det_pmask": det_pmask,
        }
