# Real-time Object Detection using TensorFlow and PiCamera

# Inspired by source: https://www.instructables.com/Object-Detection-on-Raspberry-Pi/

import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys

# Constants for image width and height
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Camera type (use PiCamera or USB webcam)
camera_source = 'PiCamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of PiCamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_source = 'USB'

sys.path.append('..')

# Importing utility functions for object detection
from utils import label_map_util
from utils import visualization_utils as vis_util

# Pre-trained model details
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
CURRENT_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CURRENT_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CURRENT_PATH, 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Load label map and create category index
label_map_data = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map_data, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load TensorFlow detection graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    tf_session = tf.Session(graph=detection_graph)

# Get required tensors from the detection graph
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Frame rate calculation variables
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize the camera
if camera_source == 'PiCamera':
    camera = PiCamera()
    camera.resolution = (IMAGE_WIDTH, IMAGE_HEIGHT)
    camera.framerate = 10
    raw_capture = PiRGBArray(camera, size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    raw_capture.truncate(0)

    for frame1 in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):

        t1 = cv2.getTickCount()

        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        (boxes, scores, classes, num) = tf_session.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)

        cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (30, 50), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        if cv2.waitKey(1) == ord('q'):
            break

        raw_capture.truncate(0)

    camera.close()
    camera.release()

cv2.destroyAllWindows()
