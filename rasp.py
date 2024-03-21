from gpiozero import LED
from picamera2 import Picamera2

from light import blink_fast
from model import get_keypoints
from body import ANGLES_TO_MONITOR, vec_from_edge, calculate_angle, angle_from_keypoints
from draw import edge_over_threshold, scale_keypoints
import numpy as np
import cv2 as cv

camera = Picamera2()
threshold = 0.35
led = LED(17)

camera.start()
while True:
    # Capture frame-by-frame
    frame = camera.capture_array()

    resized_frame = cv.resize(frame, (192, 192))
    keypoints_with_scores = scale_keypoints(get_keypoints(resized_frame.reshape(1, *resized_frame.shape)))

    keypoints_locs = keypoints_with_scores[:, :2].astype(int)
    scores = keypoints_with_scores[:, 2]

    for edge1, edge2 in ANGLES_TO_MONITOR:
        # Create vectors for each edge in the pair
        if edge_over_threshold(edge1, scores, threshold) and edge_over_threshold(edge2, scores, threshold):
            angle = angle_from_keypoints(keypoints_with_scores, edge1, edge2)
            if angle > 180:
                blink_fast(0.1)
            else:
                led.off()

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
camera.stop()
cv.destroyAllWindows()
