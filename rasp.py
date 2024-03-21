from picamera2 import Picamera2
from light import blink_fast, led
from model import get_keypoints
from body import ANGLES_TO_MONITOR, angle_from_keypoints
from draw import edge_over_threshold, scale_keypoints
import cv2 as cv
import math

def main(show=False):
	camera = Picamera2()
	threshold = 0.35

	camera.start()
	while True:
		# Capture frame-by-frame
		frame = camera.capture_array()
		frame = frame[:,:,:3]
		frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

		resized_frame = cv.resize(frame, (192, 192))
		keypoints_with_scores = scale_keypoints(get_keypoints(resized_frame.reshape(1, *resized_frame.shape)))

		keypoints_locs = keypoints_with_scores[:, :2].astype(int)
		scores = keypoints_with_scores[:, 2]

		for edge1, edge2 in ANGLES_TO_MONITOR:
			# Create vectors for each edge in the pair
			if edge_over_threshold(edge1, scores, threshold) and edge_over_threshold(edge2, scores, threshold):
				angle = angle_from_keypoints(keypoints_with_scores, edge1, edge2)
				if math.sin(angle) > 0.5:
					led.blink(0.1, 0.1, 15)
				else:
					led.off()
		
		if show:
			frame = draw_keypoints(image, keypoints_locs, scores, threshold)
			frame, drawn_edges = draw_edges_angles(image, keypoints_locs, scores)
			frame = draw_edges_lines(image, drawn_edges, keypoints_locs, scores, threshold)
			cv.imshow('main', cv.resize(image_w_kpts, (640, 480)))
			
			if cv.waitKey(1) == ord('q'):
				break

	# When everything done, release the capture
	camera.stop()
	cv.destroyAllWindows()


if __name__ == "__main__":
	main()
