import tflite_runtime.interpreter as tflite
import numpy as np

input_size = 192

# Initialize the TFLite interpreter
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()


def get_keypoints(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """

    # TF Lite format expects tensor type of uint8.
    input_image = input_image.astype(np.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    return interpreter.get_tensor(output_details[0]['index'])


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
