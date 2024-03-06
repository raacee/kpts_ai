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

        # Make predictions and draw on image
        frame = cv.resize(frame, (192, 192))
        frame = frame.reshape((1, *frame.shape))
        kpts = get_keypoints(frame)
        image = draw_prediction_on_image(frame, kpts)

        # Display the resulting frame
        cv.imshow('frame', image)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
