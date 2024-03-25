from picamera2 import Picamera2
from gpiozero import LED
from model import get_keypoints
from body import ANGLES_TO_MONITOR, angle_from_keypoints
from draw import edge_over_threshold, scale_keypoints, draw_edges_angles, draw_keypoints, draw_edges_lines
import cv2 as cv
import math

camera = Picamera2()
threshold = 0.35
led = LED(17)


def main(show=True):
    """
    Main function for Raspberry Pi posture detection
    Args:
        show: boolean flag to show captured video in open cv window

    Returns:
        None
    """
    camera.start()
    if show:
        while True:
            # Capture frame-by-frame
            resized_frame, keypoints_locs, scores = led_detect(camera)
            resized_frame = draw_keypoints(resized_frame, keypoints_locs, scores, threshold)
            resized_frame, drawn_edges = draw_edges_angles(resized_frame, keypoints_locs, scores, threshold)
            resized_frame = draw_edges_lines(resized_frame, drawn_edges, keypoints_locs, scores, threshold)

            cv.imshow('main', cv.resize(resized_frame, (640, 480)))

            if cv.waitKey(1) == ord('q'):
                break
    else:
        while True:
            try:
                led_detect(camera)
            except KeyboardInterrupt:
                break

    # When everything done, release the capture
    camera.stop()
    cv.destroyAllWindows()


def led_detect(camera_element):
    """
    Blinks led according to angles measured between monitored edges
    Args:
        camera_element: Camera passed as a parameter that will be capturing images
    Returns:
        None
    """
    frame = camera_element.capture_array()
    frame = frame[:, :, :3]
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    resized_frame = cv.resize(frame, (192, 192))

    height, width, _ = resized_frame.shape

    keypoints_with_scores = get_keypoints(resized_frame.reshape(1, *resized_frame.shape))
    keypoints_with_scores_scaled = scale_keypoints(keypoints_with_scores, width=width, height=height)
    keypoints_locs = keypoints_with_scores_scaled[:, :2].astype(int)
    scores = keypoints_with_scores_scaled[:, 2]

    for edge1, edge2 in ANGLES_TO_MONITOR:
        # Create vectors for each edge in the pair
        if edge_over_threshold(edge1, scores, threshold) and edge_over_threshold(edge2, scores, threshold):
            angle, _, _, _, _ = angle_from_keypoints(keypoints_with_scores_scaled, edge1, edge2)
            if math.sin(angle) > 0.5:
                led.blink(0.1, 0.1, 15)
            else:
                led.off()

    return resized_frame, keypoints_locs, scores


if __name__ == "__main__":
    main()
