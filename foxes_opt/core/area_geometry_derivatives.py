from typing import cast

import numpy as np
from foxes.utils.geom2d import (
    AreaGeometry,
    AreaIntersection,
    AreaUnion,
    Circle,
    ClosedPolygon,
    InvertedAreaGeometry,
)


def _polygon_candidates(geometry: ClosedPolygon, point: np.ndarray) -> list[np.ndarray]:
    """Find all nearest points on polygon segments."""
    starts = geometry.points[:-1]
    segments = geometry.points[1:] - starts
    lengths_squared = np.einsum("pd,pd->p", segments, segments)
    nearest = starts.copy()
    valid = lengths_squared > 0.0
    nearest[valid] += (
        segments[valid]
        * np.clip(
            np.einsum("pd,pd->p", point - starts[valid], segments[valid])
            / lengths_squared[valid],
            0.0,
            1.0,
        )[:, None]
    )
    distances = np.linalg.norm(point - nearest, axis=1)
    active = np.isclose(distances, np.min(distances), rtol=1e-10, atol=1e-12)
    return [candidate for candidate in nearest[active]]


def _boundary_candidates(geometry: AreaGeometry, point: np.ndarray) -> list[np.ndarray]:
    """Collect nearest candidates from primitive geometry boundaries."""
    if isinstance(geometry, ClosedPolygon):
        return _polygon_candidates(geometry, point)
    if isinstance(geometry, Circle) and np.all(point == geometry.centre):
        offset = np.array([geometry.radius, 0.0])
        return [geometry.centre - offset, geometry.centre + offset]
    if isinstance(geometry, (AreaUnion, AreaIntersection)):
        return [
            candidate
            for child in geometry.geometries
            for candidate in _boundary_candidates(child, point)
        ]
    if isinstance(geometry, InvertedAreaGeometry):
        return _boundary_candidates(geometry._geometry, point)
    result = geometry.points_distance(point[None, :], return_nearest=True)
    __, nearest = cast(tuple[np.ndarray, np.ndarray], result)
    return [nearest[0]]


def _unique_nearest(
    geometry: AreaGeometry, points: np.ndarray, distances: np.ndarray
) -> np.ndarray:
    """Check whether each point has one distinct active nearest boundary point."""
    out = np.zeros(len(points), dtype=bool)
    for index, (point, distance) in enumerate(zip(points, distances)):
        try:
            candidates = _boundary_candidates(geometry, point)
        except (FloatingPointError, ValueError):
            continue
        active = [
            candidate
            for candidate in candidates
            if np.isclose(
                np.linalg.norm(point - candidate),
                distance,
                rtol=1e-10,
                atol=1e-12,
            )
        ]
        unique: list[np.ndarray] = []
        for candidate in active:
            if not any(np.allclose(candidate, other) for other in unique):
                unique.append(candidate)
        out[index] = len(unique) == 1
    return out


def signed_distance_derivatives(
    geometry: AreaGeometry,
    points: np.ndarray,
    point_derivatives: np.ndarray,
) -> np.ndarray:
    """Differentiate signed point-to-boundary distances."""
    out = np.zeros(len(points), dtype=np.float64)
    moving = np.any(point_derivatives != 0.0, axis=1)
    if not np.any(moving):
        return out
    gradients = signed_distance_gradients(geometry, points[moving])
    out[moving] = np.einsum("pd,pd->p", gradients, point_derivatives[moving])
    return out


def signed_distance_gradients(
    geometry: AreaGeometry,
    points: np.ndarray,
) -> np.ndarray:
    """Calculate signed point-to-boundary distance gradients."""
    out = np.full((len(points), 2), np.nan, dtype=np.float64)
    try:
        result = geometry.points_distance(points, return_nearest=True)
        distances, nearest = cast(tuple[np.ndarray, np.ndarray], result)
    except (FloatingPointError, ValueError):
        return out
    delta = points - nearest
    distance = np.linalg.norm(delta, axis=1)
    differentiable = (distance > 0.0) & _unique_nearest(geometry, points, distances)
    sign = np.where(geometry.points_inside(points), -1.0, 1.0)
    out[differentiable] = sign[differentiable, None] * (
        delta[differentiable] / distance[differentiable, None]
    )
    return out
