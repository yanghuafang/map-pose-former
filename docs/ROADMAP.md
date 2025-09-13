# Roadmap

## The problem

A vehicle has a prior pose that is wrong by a metre or two, a map of surveyed
landmarks, and a perception system reporting landmarks it can see. Recover the
correction, and a covariance honest enough for a filter to trust.

The hard part is not the geometry. Given correct correspondences the pose is a
closed-form least-squares solve. The hard part is **association**: lane dashes
repeat every few metres and parallel lines are locally identical, so the nearest
map element to a detection is routinely the wrong one — and it is most often
wrong along the road, which is the axis the landmarks constrain worst.

## The bar for `main`

A change lands when its effect exceeds the seed variance M1 measures, on the
same protocol, with the arms differing by exactly one thing. Anything smaller
is unresolved, not negative.
