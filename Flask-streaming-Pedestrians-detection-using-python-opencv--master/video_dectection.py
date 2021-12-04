import sys
import time
import numpy as np
import tensorflow as tf
import cv2

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils_color as vis_util


PATH_TO_CKPT = './model/100K-steps.pb'

PATH_TO_LABELS = './model/label_map.pbtxt'

NUM_CLASSES = 3

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

cap = cv2.VideoCapture("test.mp4")
out = None

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.compat.v1.Session(graph=detection_graph, config=config) as sess:
    frame_num = 1490
    while frame_num:
      frame_num -= 1
      ret, image = cap.read()
      if ret == 0:
          break

      if out is None:
          [h, w] = image.shape[:2]
          fourcc = cv2.VideoWriter_fourcc(*'mp4v')
          out = cv2.VideoWriter("facebook.mp4", fourcc, 30, (w, h))


      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      start_time = time.time()
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      elapsed_time = time.time() - start_time
      print('inference time cost: {}'.format(elapsed_time))
      #print(boxes.shape, boxes)
      #print(scores.shape,scores)
      #print(classes.shape,classes)
      #print(num_detections)
      vis_util.visualize_boxes_and_labels_on_image_array(

          image,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)
      out.write(image)


    cap.release()
    out.release()