import numpy as np
from iwopy import Problem
from scipy.spatial.distance import cdist


def layout_data(
    problem: Problem,
    vars_int: np.ndarray,
    vars_float: np.ndarray,
    var: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return layout points, validity, and one Jacobian column."""
    derivative = getattr(problem, "layout_derivative", None)
    if not callable(derivative):
        return None
    points, valid = problem.apply_individual(vars_int, vars_float)
    return points, valid, derivative(vars_int, vars_float, var)


def _common_derivative(values: list[float]) -> float:
    """Return a common active derivative, or NaN when branches disagree."""
    derivatives = np.asarray(values, dtype=np.float64)
    if not len(derivatives) or np.any(np.isnan(derivatives)):
        return np.nan
    return float(derivatives[0]) if np.allclose(derivatives, derivatives[0]) else np.nan


def _distance_derivative(delta: np.ndarray, point_derivative: np.ndarray) -> float:
    """Differentiate one point-to-fixed-point distance."""
    distance = np.linalg.norm(delta)
    if distance > 0.0:
        return float(np.dot(delta / distance, point_derivative))
    return 0.0 if not np.any(point_derivative) else np.nan


def maximin_distance_derivative(
    probes: np.ndarray,
    points: np.ndarray,
    valid: np.ndarray,
    point_derivatives: np.ndarray,
) -> float:
    """Differentiate the largest probe-to-nearest-point distance."""
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        return np.nan
    distances = cdist(probes, points[valid_indices])
    nearest_distances = np.min(distances, axis=1)
    active_probes = np.flatnonzero(
        np.isclose(nearest_distances, np.max(nearest_distances))
    )
    probe_derivatives = []
    for probe in active_probes:
        active_points = np.flatnonzero(
            np.isclose(distances[probe], nearest_distances[probe])
        )
        derivatives = [
            _distance_derivative(
                points[valid_indices[point]] - probes[probe],
                point_derivatives[valid_indices[point]],
            )
            for point in active_points
        ]
        probe_derivatives.append(_common_derivative(derivatives))
    return _common_derivative(probe_derivatives)


def nearest_distance_derivatives(
    points: np.ndarray,
    point_derivatives: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate nearest-neighbour distances and one derivative column."""
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    derivatives = np.zeros(len(points), dtype=np.float64)
    for point in range(len(points)):
        neighbours = np.flatnonzero(np.isclose(distances[point], nearest[point]))
        candidates = [
            _distance_derivative(
                points[point] - points[neighbour],
                point_derivatives[point] - point_derivatives[neighbour],
            )
            for neighbour in neighbours
        ]
        derivatives[point] = _common_derivative(candidates)
    return nearest / scale, derivatives / scale


def extreme_derivative(
    values: np.ndarray, derivatives: np.ndarray, maximize: bool
) -> float:
    """Differentiate a minimum or maximum across active scalar branches."""
    extreme = np.max(values) if maximize else np.min(values)
    active = np.flatnonzero(np.isclose(values, extreme))
    return _common_derivative(derivatives[active].tolist())
