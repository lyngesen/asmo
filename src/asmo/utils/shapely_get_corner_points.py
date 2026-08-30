"""
Used to get corner points of a polygon by removing nearly collinear points.
source from
https://github.com/shapely/shapely/issues/1046
"""

import numpy as np
from shapely import Polygon


def get_angles(vec_1, vec_2):
    """
    return the angle, in degrees, between two vectors
    """

    dot = np.dot(vec_1, vec_2)
    # det = np.cross(vec_1, vec_2)
    # DeprecationWarning: Arrays of 2-dimensional vectors are deprecated. Use arrays of 3-dimensional vectors instead. (deprecated in NumPy 2.0)
    det = vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0]
    angle_in_rad = np.arctan2(det, dot)
    return np.degrees(angle_in_rad)


def simplify_by_angle(poly_in: Polygon, deg_tol=0):
    """Try to remove persistent coordinate points that remain after
    simplify, convex hull, or something, etc. with some trig instead

    The function `simplify_by_angle` aims to reduce the number of vertices in a polygon by removing points that create nearly straight lines. The geometric intuition is that if three consecutive points in a polygon form an angle that is very close to 180 degrees (i.e., the middle point is nearly collinear with the other two), then the middle point can be considered redundant and can be removed without significantly altering the shape of the polygon. By calculating the angles between successive vectors formed by the polygon's edges, the function identifies and retains only those vertices that contribute to significant changes in direction, effectively simplifying the polygon while preserving its overall shape.

    poly_in: shapely Polygon
    deg_tol: degree tolerance for comparison between successive vectors
    """
    try:
        ext_poly_coords = poly_in.exterior.coords[:]
    except AttributeError:
        ext_poly_coords = poly_in.coords[:]
    vector_rep = np.diff(ext_poly_coords, axis=0)
    num_vectors = len(vector_rep)
    angles_list = []
    for i in range(0, num_vectors):
        angles_list.append(
            np.abs(get_angles(vector_rep[i], vector_rep[(i + 1) % num_vectors]))
        )

    #   get mask satisfying tolerance
    thresh_vals_by_deg = np.where(np.array(angles_list) > deg_tol)

    new_idx = list(thresh_vals_by_deg[0] + 1)
    new_vertices = [ext_poly_coords[idx] for idx in new_idx]

    return new_vertices
