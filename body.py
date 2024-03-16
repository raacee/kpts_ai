import math
import numpy as np

"""Dictionary that maps from joint names to keypoint indices."""
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

"""Maps bones to a matplotlib color name."""
KEYPOINT_EDGES: [tuple] = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16)
]

"""List of tuples representing the angles to monitor articulations
They mean to be representing the angle at the second entry"""
ANGLES_TO_MONITOR = (
    ((5, 11), (11, 13)),
    ((6, 12), (12, 14)),
    ((5, 7), (5, 11)),
    ((6, 8), (6, 12))
)

"""Tuples representing all the edges in ANGLES_TO_MONITOR"""
EDGES_TO_MONITOR = (
    (5, 7),
    (6, 8),
    (5, 11),
    (6, 12),
    (11, 13),
    (12, 14),
)

"""Edges dict that give to each edge its counter part to measure angles in which it is implicated"""
EDGES_DICT = {
    (5, 11): [(11, 13), (5, 7)],
    (11, 13): [(5, 11), (5, 7)],
    (6, 12): [(12, 14), (6, 8)],
}


def calculate_angle(vec1, vec2):
    """Angle between the vectors passed as parameters and
    returns the angle between the vectors

    Args:
      vec1: First vector of shape (1,n)
      vec2: Second vector of shape (1,n)

    Returns:
      The angle between the vectors passed as parameters
    """
    value = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    try:
        return math.acos(
            value
        )
    except TypeError as err:
        print(value)
        raise err


def vec_from_edge(edge_coordinates):
    """Calculates the vector from the two points of the edge passed as a parameter

        Args:
          edge_coordinates: A tuple containing the coordinates of the edge points

        Returns:
          The angle between the vectors passed as parameters
    """
    return np.array(
        (
            (edge_coordinates[0, 0] - edge_coordinates[1, 0]),
            (edge_coordinates[0, 1] - edge_coordinates[1, 1])
        )
    )


def color_from_angle(angle):
    """
    Calculates the color of two edges according to the angle between them
    Args:
        angle: a number between 0 and 360 degrees

    Returns:
        A tuple of shape (3) representing the color of the edges

    """
    return 0, 255 - math.sin(angle) * 255, 255 * math.sin(angle)
