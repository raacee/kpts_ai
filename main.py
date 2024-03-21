from draw import draw_prediction_on_image
from model import get_keypoints
from picamera2 import Picamera2
import cv2 as cv
import time


def main():
    camera = Picamera2()
    camera.start()
    time.sleep(1)

    while True:
        # Capture frame-by-frame
        frame = camera.capture_array()

        resized_frame = cv.resize(frame, (192, 192))
        keypoints_with_scores = get_keypoints(resized_frame.reshape(1, *resized_frame.shape))
        image_w_kpts = draw_prediction_on_image(resized_frame, keypoints_with_scores)

        cv.imshow('main', cv.resize(image_w_kpts, (640, 480)))
        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    camera.stop()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
