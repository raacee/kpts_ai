import cv2 as cv
from draw import draw_prediction_on_image
from model import get_keypoints


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        resized_frame = cv.resize(frame, (192, 192))
        keypoints_with_scores = get_keypoints(resized_frame.reshape(1, *resized_frame.shape))
        image_w_kpts = draw_prediction_on_image(resized_frame, keypoints_with_scores)
        image_w_kpts = image_w_kpts.reshape((192, 192, 3))

        cv.imshow('main', cv.resize(image_w_kpts, (640, 480)))
        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
