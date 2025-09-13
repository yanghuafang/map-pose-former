"""Landmark classes, and what each one can tell a localizer.

The table below is the reason this project exists in the shape it does, and it
is copied -- with its conclusions intact -- from camera-map-localization's
architecture notes. A localizer searching three degrees of freedom needs
landmarks that constrain all three, and map features are not interchangeable:

    class            lateral   longitudinal   heading
    LANE_DIVIDER     strong    ~none          strong
    ROAD_BOUNDARY    strong    ~none          strong
    PED_CROSSING     weak      strong         strong
    STOP_LINE        weak      strong         strong
    POLE             strong    strong         moderate
    TRAFFIC_SIGN     strong    strong         moderate

Lane geometry runs *parallel* to travel, so sliding a hypothesis down the road
costs almost nothing: a model fed only lane detections has an unobservable
degree of freedom, and no amount of training fixes it. The upright and
perpendicular features are what pin along-track position.

This is a claim, and the synthetic dataset exists partly so that it can be
tested rather than repeated -- see ``docs/DATASET.md``, "the ablation that
matters".
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
