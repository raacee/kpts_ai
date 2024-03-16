from body import *
import numpy as np
import cv2 as cv


def edge_over_threshold(edge, scores, threshold):
    return scores[edge[0]] > threshold and scores[edge[1]] > threshold


def scale_keypoints(keypoints_with_scores, width=1, height=1):
    kpts_x = keypoints_with_scores[0, 0, :, 1]
    kpts_y = keypoints_with_scores[0, 0, :, 0]
    scores = keypoints_with_scores[0, 0, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y), scores],
        axis=-1
    )
    return kpts_absolute_xy


def draw_prediction_on_image(image, keypoints_with_scores, threshold=0.35):
    """Draws the keypoint predictions on image.

    Args:
      threshold: Threshold underwhich a keypoint prediction is ignored
      image: A numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
      keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.

    Returns:
      A numpy array with shape [out_height, out_width, channel] representing the
      image overlaid with keypoint predictions.
    """

    height, width, _ = image.shape

    keypoints_with_scores = scale_keypoints(keypoints_with_scores, width=width, height=height)

    radius = 2
    thickness = -1
    keypoints_locs = keypoints_with_scores[:, :2].astype(int)
    scores = keypoints_with_scores[:, 2]

    color = (255, 255, 0)

    # Draw the keypoints on the picture
    for keypoint_coords, score in zip(keypoints_locs, scores):
        if score > threshold:
            image = cv.circle(image, tuple(keypoint_coords[:2]), radius, color, thickness)

    drawn_edges = set()
    # Calculates angles between edges for edges to be monitored
    for edge1, edge2 in ANGLES_TO_MONITOR:
        # Create vectors for each edge in the pair
        if edge_over_threshold(edge1, scores, threshold) and edge_over_threshold(edge2, scores, threshold):
            edge_coordinates1 = np.array((keypoints_locs[edge1[0]], keypoints_locs[edge1[1]]))
            edge_coordinates2 = np.array((keypoints_locs[edge2[0]], keypoints_locs[edge2[1]]))
            vec1 = vec_from_edge(edge_coordinates1)
            vec2 = vec_from_edge(edge_coordinates2)

            angle = calculate_angle(vec1, vec2)
            color = color_from_angle(angle)

            image = cv.line(image, edge_coordinates1[0], edge_coordinates1[1], color, 1)
            image = cv.line(image, edge_coordinates2[0], edge_coordinates2[1], color, 1)

        drawn_edges.add(edge1)
        drawn_edges.add(edge2)

    # Draw the lines between each keypoints in edges
    color = (255, 0, 255)
    for edge in KEYPOINT_EDGES:
        if edge not in drawn_edges:
            if edge_over_threshold(edge, scores, threshold):
                start_point = keypoints_locs[edge[0]]
                end_point = keypoints_locs[edge[1]]
                image = cv.line(image, tuple(start_point), tuple(end_point), color, 1)
                drawn_edges.add(edge)

    return image
