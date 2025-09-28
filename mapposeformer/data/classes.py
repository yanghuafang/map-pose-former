"""Landmark classes, and what each one can tell a localizer.

    class            lateral   longitudinal   heading
    LANE_DIVIDER     strong    ~none          strong
    ROAD_BOUNDARY    strong    ~none          strong
    PED_CROSSING     weak      strong         strong
    STOP_LINE        weak      strong         strong
    POLE             strong    strong         moderate
    TRAFFIC_SIGN     strong    strong         moderate

Copied, with its conclusions, from camera-map-localization. Lane geometry runs
*parallel* to travel, so sliding a hypothesis down the road costs almost
nothing: a model fed only lane detections has an unobservable degree of freedom
that training cannot fix. Upright and perpendicular features pin along-track
position.

That is a claim, and the synthetic dataset exists partly so it can be tested --
``docs/DATASET.md``, "Ablations".
"""

from __future__ import annotations

import enum


class LandmarkClass(enum.IntEnum):
    """Semantic class of a map element or a detection.

    The integer values are the embedding indices, so they are part of every
    checkpoint's contract: append, never reorder.
    """

    LANE_DIVIDER = 0
    ROAD_BOUNDARY = 1
    PED_CROSSING = 2
    STOP_LINE = 3
    POLE = 4
    TRAFFIC_SIGN = 5


NUM_CLASSES = len(LandmarkClass)

#: Classes that run along the road. Ablating these leaves lateral position and
#: heading unobservable.
ALONG_TRACK_BLIND = (LandmarkClass.LANE_DIVIDER, LandmarkClass.ROAD_BOUNDARY)

#: Classes that constrain along-track position. Ablating these is the
#: experiment that makes the table above visible in the metrics.
ALONG_TRACK_ANCHORS = (
    LandmarkClass.PED_CROSSING,
    LandmarkClass.STOP_LINE,
    LandmarkClass.POLE,
    LandmarkClass.TRAFFIC_SIGN,
)

#: Which classes nuScenes' map expansion actually carries. Poles and signs are
#: not among them, which is why the real-data milestone expects weaker
#: longitudinal observability than the synthetic one. Stated here rather than
#: discovered later in a metrics table.
NUSCENES_AVAILABLE = (
    LandmarkClass.LANE_DIVIDER,
    LandmarkClass.ROAD_BOUNDARY,
    LandmarkClass.PED_CROSSING,
    LandmarkClass.STOP_LINE,
)


class MarkType(enum.IntEnum):
    """Paint style of a lane element. Argoverse 2 calls this ``mark_type``.

    A dashed line is stripes, and a stripe *end* is along-track evidence -- the
    one thing lane geometry is otherwise blind to. Real vector maps throw those
    ends away and keep the attribute that implies them, so this dataset does the
    same: the map stores the polyline and the style; the detector sees paint.
    ``docs/DATASET.md`` has the experiment that makes possible.

    Integer values are embedding indices: append, never reorder.
    """

    NONE = 0
    """Not a painted line: poles, signs, crossings, stop lines."""
    SOLID = 1
    DASHED = 2


NUM_ATTRS = len(MarkType)

#: Classes that carry a paint style. Everything else is ``MarkType.NONE``.
PAINTED = (LandmarkClass.LANE_DIVIDER, LandmarkClass.ROAD_BOUNDARY)
