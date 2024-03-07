import cv2 as cv
from model import get_keypoints
from draw import draw_prediction_on_image


def test():
    image_path = 'test.jpeg'
    img = cv.imread(image_path)
    keypoints = get_keypoints(
        cv.resize(img, (192, 192)
                  ).reshape(1, 192, 192, 3)
    )
    img = draw_prediction_on_image(img, keypoints)
    cv.imshow('test', img)
    cv.waitKey(10000)


if __name__ == "__main__":
    test()
